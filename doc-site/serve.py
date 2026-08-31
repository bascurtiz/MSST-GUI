#!/usr/bin/env python3
"""serve.py — preview the generated site locally (stdlib only).

  python serve.py                    # serve ./site on port 8080
  python serve.py --port 9000 --dir /path/to/site
  python serve.py --watch 30 --doc DOC_ID --source api  # also re-generate
                                                       # every 30 minutes

Only the Python standard library is used.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="site", help="directory to serve (default: site)")
    ap.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    ap.add_argument("--watch", type=int, default=0,
                    help="re-generate every N minutes (0 = off; needs --doc)")
    ap.add_argument("--doc", default=None, help="document ID for --watch")
    ap.add_argument("--source", default=None, help="source for --watch (api|export)")
    ap.add_argument("--auth-file", default="auth.json",
                    help="OAuth token file (default: auth.json)")
    args = ap.parse_args()

    if args.watch and not args.doc:
        ap.error("--watch needs --doc (and optionally --source)")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=args.dir, **kw)

        def log_message(self, fmt, *m):
            pass  # keep the console quiet

    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        sys.stderr.write(f"cannot bind {args.host}:{args.port} — {e}\n")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print(f"Serving {os.path.abspath(args.dir)}  at  {url}")

    if args.watch:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdoc_site.py")

        def regenerate():
            while True:
                time.sleep(args.watch * 60)
                cmd = [sys.executable, script, "--doc", args.doc, "--out", args.dir,
                       "--auth-file", args.auth_file]
                if args.source:
                    cmd += ["--source", args.source]
                print(f"[{time.strftime('%H:%M:%S')}] regenerating…")
                subprocess.call(cmd)

        threading.Thread(target=regenerate, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
