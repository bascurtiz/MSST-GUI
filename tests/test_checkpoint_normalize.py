"""Regression: >4 GiB (ZIP64) checkpoints must load deterministically.

torch 2.11's C++ zip reader on Windows crashes natively (access violation
in get_storage_from_record) on ZIP64 archives written by older torch /
community training tools — the layout of large mel-band checkpoints such as
mbr_4stemxl1_aname (5.97 GB). torch.load(mmap=True) reads them but its
mapped storages intermittently fault during the state-dict copy, so mmap is
not reliable either. ensure_readable_checkpoint rewrites a ZIP64 archive
once with Python's zipfile (which torch reads fine at any size) to a cached
sibling and returns that path. These checks exercise the normalize / cache /
rebuild logic at small scale by lowering the size threshold.
"""
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import utils.model_utils as mmu  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def main():
    # Threshold sanity: anything >= 4 GiB is a ZIP64 archive.
    check(mmu._BIG_CHECKPOINT_BYTES == 2 ** 32,
          "ZIP64 threshold must be 2**32 bytes")

    with tempfile.TemporaryDirectory() as tmp:
        # Small files must pass through untouched (no sibling, no rewrite).
        small = os.path.join(tmp, "small.ckpt")
        with open(small, "wb") as f:
            f.write(b"not really a checkpoint")
        check(mmu.ensure_readable_checkpoint(small) == small,
              "sub-threshold file must be returned unchanged")
        check(not os.path.exists(small + ".plain"),
              "sub-threshold file must not gain a normalized sibling")

        # Force the ZIP64 branch with a real torch-saved archive.
        sd = {f"layer.{i}.weight": torch.randn(64, 64) for i in range(4)}
        src = os.path.join(tmp, "big.ckpt")
        torch.save(sd, src)
        old_threshold = mmu._BIG_CHECKPOINT_BYTES
        old_mtime = os.path.getmtime(src)
        os.utime(src, (1_000_000_000, 1_000_000_000))  # fixed old mtime
        try:
            mmu._BIG_CHECKPOINT_BYTES = 0  # normalize everything

            # 1. First call normalizes and returns the sibling path.
            norm = mmu.ensure_readable_checkpoint(src)
            check(norm == src + ".plain", "normalized path must be '<src>.plain'")
            check(os.path.exists(norm), "normalized sibling must exist")
            with zipfile.ZipFile(norm) as zf:
                check(zf.comment == b"fb:%d:%d" % (os.path.getsize(src),
                                                   1_000_000_000),
                      "sibling must carry the source size+mtime marker")

            # 2. torch plain-loads the normalized archive with equal values.
            rel = torch.load(norm, weights_only=False, map_location="cpu")
            for k, v in sd.items():
                check(torch.equal(rel[k], v), f"value mismatch after normalize: {k}")

            # 3. Second call reuses the cache (no rewrite).
            mtime = os.path.getmtime(norm)
            check(mmu.ensure_readable_checkpoint(src) == norm,
                  "cached sibling must be reused")
            check(os.path.getmtime(norm) == mtime,
                  "reuse must not rewrite the sibling")

            # 4. A re-downloaded checkpoint (new mtime) triggers a rebuild.
            os.utime(src, (2_000_000_000, 2_000_000_000))
            norm2 = mmu.ensure_readable_checkpoint(src)
            check(norm2 == norm, "rebuild must target the same sibling path")
            with zipfile.ZipFile(norm2) as zf:
                check(zf.comment == b"fb:%d:%d" % (os.path.getsize(src),
                                                   2_000_000_000),
                      "marker must update after source change")
            rel2 = torch.load(norm2, weights_only=False, map_location="cpu")
            for k, v in sd.items():
                check(torch.equal(rel2[k], v), f"value mismatch after rebuild: {k}")

            # 5. No stray temp files left behind.
            leftovers = [n for n in os.listdir(tmp)
                         if ".tmp." in n or n.endswith(".plain.tmp")]
            check(not leftovers, f"stray temp files: {leftovers}")
        finally:
            mmu._BIG_CHECKPOINT_BYTES = old_threshold

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
