#!/usr/bin/env python3
"""gdoc_site.py — turn a big Google Doc into a fast, tiny static site.

The Google Doc stays the source of truth. This script pulls it and renders a
static site: one small page per top-level section, a sidebar table of
contents (grouped by document tab), client-side search, working internal
links and "edit this section" links back to Google Docs.

Sources
-------
api      Google Docs API via OAuth (preserves document tabs). Default when
         auth.json exists. Uses documents.get with includeTabsContent=true.
         One-time setup: python gdoc_site.py --auth --client-json <file>.
         (Google no longer accepts API keys for the Docs API.)
export   Public HTML export (no auth; tabs are flattened into one continuous
         document). Default when auth.json is absent.
file     Read a saved documents.get JSON response (handy for testing).
         Use --file path.

Usage
-----
  python gdoc_site.py --auth --client-json client_secret_XXXX.json   # once
  python gdoc_site.py --doc DOC_ID [--source api] [--out site]      # tabs
  python gdoc_site.py --doc DOC_ID [--out site]                     # export
  python gdoc_site.py --doc DOC_ID --source file --file resp.json

Only the Python standard library is used.
"""

import argparse
import html as htmlmod
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

DOCS_API = "https://docs.googleapis.com/v1/documents/{doc}?includeTabsContent=true"
EXPORT_URL = "https://docs.google.com/document/d/{doc}/export?format=html"
USER_AGENT = "Mozilla/5.0 (gdoc-site mirror generator)"

# --- OAuth 2.0 (Google no longer accepts API keys for the Docs API) ---------
TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_SCOPE = "https://www.googleapis.com/auth/documents.readonly"
AUTH_URL = (
    "https://accounts.google.com/o/oauth2/v2/auth"
    "?client_id={client_id}&redirect_uri={redirect_uri}"
    "&response_type=code&scope={scope}"
    "&access_type=offline&prompt=consent"
    "&code_challenge={challenge}&code_challenge_method=S256")

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def esc(s):
    """Escape text for HTML body content."""
    return htmlmod.escape(s or "", quote=False)


def attr(s):
    """Escape text for an HTML attribute value."""
    return htmlmod.escape(s or "", quote=True)


def text_of_runs(runs):
    parts = []
    for r in runs:
        if r.get("br"):
            parts.append(" ")
        elif r.get("img"):
            parts.append(" [image] ")
        else:
            parts.append(r.get("text", ""))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def is_decorative_heading(t):
    t = t.strip()
    if not t:
        return True
    if re.fullmatch(r"[\s\-–—_=~*·•.]+", t):
        return True
    return False


def slugify(t):
    t = t.lower()
    t = re.sub(r"[^\w\s-]", "", t, flags=re.UNICODE)
    t = re.sub(r"[\s_]+", "-", t).strip("-")
    return t or "section"


def unique_slug(base, used, max_len=60):
    base = base[:max_len].rstrip("-")
    slug = base
    i = 2
    while slug in used:
        suffix = f"-{i}"
        slug = base[: max_len - len(suffix)].rstrip("-") + suffix
        i += 1
    used.add(slug)
    return slug


# ---------------------------------------------------------------------------
# link resolution
# ---------------------------------------------------------------------------

def unwrap_google_url(href):
    """Google's HTML export wraps external links in a /url?q= redirect."""
    u = urllib.parse.urlsplit(href)
    if u.netloc in ("www.google.com", "google.com") and u.path in ("/url", "/url/"):
        q = urllib.parse.parse_qs(u.query)
        if "q" in q:
            return q["q"][0]
    return href


def raw_link_from_href(href):
    """Turn an href from the export HTML into a raw link descriptor."""
    if not href:
        return None
    href = href.strip()
    if href.startswith("#"):
        return ("heading", href[1:], None)
    if href.startswith("?"):
        q, _, frag = href.partition("#")
        params = urllib.parse.parse_qs(q.lstrip("?"))
        tab = (params.get("tab") or [None])[0]
        if frag.startswith("heading="):
            return ("heading", frag.split("=", 1)[1], tab)
        if tab:
            return ("tab", tab)
        return None
    return ("url", unwrap_google_url(href))


def parse_api_link(link):
    """Turn a Link object from the Docs API into a raw link descriptor."""
    if "heading" in link:
        h = link["heading"]
        return ("heading", h.get("id"), h.get("tabId"))
    if "headingId" in link:
        return ("heading", link["headingId"], None)
    if "bookmark" in link:
        return ("bookmark", link["bookmark"].get("tabId"))
    if "bookmarkId" in link:
        return ("bookmark", None)
    if "tabId" in link:
        return ("tab", link["tabId"])
    if link.get("url"):
        return ("url", link["url"])
    return None


CHUNK_THRESHOLD = 120_000  # estimated HTML bytes per page before splitting


def _runs_cost(runs):
    n = 0
    for r in runs:
        n += 40 + len(r.get("text", ""))
        if r.get("link"):
            n += 100  # <a href="…">…</a> overhead
        if r.get("img"):
            n += 120
    return n


def estimate_blocks(blocks):
    """Rough estimate of the rendered HTML size of a list of blocks."""
    n = 0
    for b in blocks:
        t = b["type"]
        if t in ("para", "heading"):
            n += _runs_cost(b.get("runs", []))
        elif t == "list":
            for it in b.get("items", []):
                n += _runs_cost(it.get("runs", []))
        elif t == "table":
            for row in b.get("rows", []):
                for cell in row:
                    n += estimate_blocks(cell)
        elif t == "toc":
            n += estimate_blocks(b.get("blocks", []))
        elif t == "html":
            n += len(b.get("html", ""))
    return n


def chunk_blocks(blocks, threshold=CHUNK_THRESHOLD):
    """Split a list of blocks into pages at sub-headings when it gets big.

    Returns a list of (blocks, depth): each page starts with the heading that
    names it (the first page starts with the section's own heading). depth is
    the heading level of the page's title minus 2 (0 for a section page).
    """
    if estimate_blocks(blocks) <= threshold:
        return [(blocks, 0)]
    counts = {}
    for b in blocks:
        if (b["type"] == "heading"
                and not is_decorative_heading(text_of_runs(b["runs"]))):
            counts[b["level"]] = counts.get(b["level"], 0) + 1
    # h6 is too fine-grained to page on (it is used for list-like entries);
    # splitting below h5 just creates one-line pages.
    lvl = next((lv for lv in (3, 4, 5) if counts.get(lv, 0) >= 2), None)
    if lvl is None:
        return [(blocks, 0)]  # no useful sub-structure: keep as one page
    pages = []
    cur = None
    for b in blocks:
        if (b["type"] == "heading" and b["level"] == lvl
                and not is_decorative_heading(text_of_runs(b["runs"]))):
            if cur is not None:
                pages.append(cur)
            cur = [b]
        else:
            if cur is None:
                cur = []
            cur.append(b)
    if cur is not None:
        pages.append(cur)
    out = []
    for pg in pages:
        for sub_blocks, _ in chunk_blocks(pg, threshold):
            depth = 0
            if sub_blocks and sub_blocks[0]["type"] == "heading":
                depth = max(sub_blocks[0]["level"] - 2, 0)
            out.append((sub_blocks, depth))
    return out


def build_subs(blocks):
    """Sub-headings inside a page (everything except the page's own title)."""
    subs = []
    for i, b in enumerate(blocks):
        if i == 0 and b["type"] == "heading":
            continue  # the page's own title heading
        if b["type"] == "heading":
            title = text_of_runs(b["runs"])
            if not is_decorative_heading(title):
                subs.append({"level": b["level"], "id": b.get("heading_id"),
                             "title": title})
    return subs


class Site:
    """Holds the parsed document and knows how to resolve links."""

    def __init__(self, doc_id, title, tabs, source):
        self.doc_id = doc_id
        self.title = title
        self.source = source
        self.tabs = [t for t in tabs if t.get("blocks")]
        self.sections = []
        self.heading_map = {}       # heading id -> section
        self.heading_tab_map = {}   # (tab id, heading id) -> section
        self.tab_first = {}         # tab id -> slug of first section
        self.generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        self._split_and_slug()

    # -- structure ----------------------------------------------------------

    @staticmethod
    def _boundary_level(blocks):
        # Prefer the smallest heading level that actually appears as a real
        # (non-decorative) heading. For Google Docs, HEADING_1 maps to our
        # level 2, so most docs split there; docs that only use Heading 2+
        # still split correctly.
        for lvl in (2, 3, 4, 5, 6):
            for b in blocks:
                if (b["type"] == "heading" and b["level"] == lvl
                        and not is_decorative_heading(text_of_runs(b["runs"]))):
                    return lvl
        return 0

    def _split_and_slug(self):
        # 1) cut each tab into sections at its top-level headings
        raw = []
        for tab in self.tabs:
            boundary = self._boundary_level(tab["blocks"])
            cur = None
            for b in tab["blocks"]:
                if b["type"] == "heading" and b["level"] == boundary:
                    title = text_of_runs(b["runs"])
                    if is_decorative_heading(title):
                        continue  # spacer / separator headings are dropped
                    cur = {"tab": tab["id"], "tab_title": tab["title"],
                           "title": title, "heading_id": b.get("heading_id"),
                           "blocks": [b]}
                    raw.append(cur)
                else:
                    if cur is None:  # preamble before the first heading
                        cur = {"tab": tab["id"], "tab_title": tab["title"],
                               "title": tab["title"], "heading_id": None,
                               "blocks": []}
                        raw.append(cur)
                    cur["blocks"].append(b)

        # 2) split oversized sections into multiple pages at sub-headings
        used = set()
        for sec in raw:
            pages = chunk_blocks(sec["blocks"])
            sec["slug"] = unique_slug(slugify(sec["title"]), used)
            if sec["tab"] not in self.tab_first:
                self.tab_first[sec["tab"]] = sec["slug"]
            for i, (blocks, depth) in enumerate(pages):
                if i == 0:
                    s = sec
                    s["blocks"] = blocks  # may have been trimmed by chunking
                    s["subs"] = build_subs(blocks)
                    s["parent"] = None
                    s["depth"] = 0
                else:
                    title = text_of_runs(blocks[0]["runs"])
                    s = {
                        "tab": sec["tab"], "tab_title": sec["tab_title"],
                        "title": title,
                        "heading_id": blocks[0].get("heading_id"),
                        "blocks": blocks, "subs": build_subs(blocks),
                        "parent": sec["slug"], "depth": depth,
                        "slug": unique_slug(sec["slug"] + "-" + slugify(title), used),
                    }
                self.sections.append(s)

        # 3) heading id -> page map, for internal links
        for s in self.sections:
            hid = s.get("heading_id")
            if hid:
                self.heading_map.setdefault(hid, s)
                self.heading_tab_map.setdefault((s["tab"], hid), s)
            for sub in s.get("subs", []):
                if sub.get("id"):
                    self.heading_map.setdefault(sub["id"], s)
                    self.heading_tab_map.setdefault((s["tab"], sub["id"]), s)

    # -- link resolution ----------------------------------------------------

    def heading_slug(self, hid, tab=None):
        if tab:
            s = self.heading_tab_map.get((tab, hid))
            if s:
                return s["slug"]
        s = self.heading_map.get(hid)
        return s["slug"] if s else None

    def tab_url(self, tab):
        if tab in self.tab_first:
            return self.tab_first[tab] + ".html"
        return "index.html"

    def resolve_url(self, url):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https"):
            return url
        if u.netloc in ("docs.google.com", "docs.googleusercontent.com") \
                and u.path.startswith("/document/d/"):
            parts = u.path.split("/")
            if len(parts) >= 4 and parts[3] == self.doc_id:
                q = urllib.parse.parse_qs(u.query)
                tab = (q.get("tab") or [None])[0]
                frag = u.fragment
                if frag.startswith("heading="):
                    hid = frag.split("=", 1)[1]
                    slug = self.heading_slug(hid, tab)
                    return f"{slug}.html#{hid}" if slug else "index.html"
                if tab:
                    return self.tab_url(tab)
                return "index.html"
        return url

    def resolve_run(self, raw):
        if raw is None:
            return None
        kind = raw[0]
        if kind == "heading":
            hid, tab = raw[1], raw[2]
            slug = self.heading_slug(hid, tab)
            return f"{slug}.html#{hid}" if slug else "index.html"
        if kind == "bookmark":
            return self.tab_url(raw[1]) if raw[1] else "index.html"
        if kind == "tab":
            return self.tab_url(raw[1])
        if kind == "url":
            return self.resolve_url(raw[1])
        return None

    def edit_url(self, section):
        tab = section["tab"]
        hid = section.get("heading_id")
        base = f"https://docs.google.com/document/d/{self.doc_id}/edit?tab={tab}"
        if hid:
            base += f"#heading={hid}"
        return base

    # -- rendering ----------------------------------------------------------

    def render_runs(self, runs):
        out = []
        for r in runs:
            if r.get("br"):
                out.append("<br>")
                continue
            if r.get("img"):
                out.append(f'<img src="{attr(r["img"])}" alt="" loading="lazy">')
                continue
            t = esc(r.get("text", ""))
            if not t:
                continue
            if r.get("code"):
                t = f"<code>{t}</code>"
            else:
                if r.get("bold"):
                    t = f"<strong>{t}</strong>"
                if r.get("italic"):
                    t = f"<em>{t}</em>"
                if r.get("underline"):
                    t = f"<u>{t}</u>"
                if r.get("strike"):
                    t = f"<s>{t}</s>"
                if r.get("sub"):
                    t = f"<sub>{t}</sub>"
                if r.get("sup"):
                    t = f"<sup>{t}</sup>"
            style = []
            if r.get("color") and not color_is_dark(r["color"]):
                style.append(f"color:{r['color']}")
            if r.get("bg"):
                style.append(f"background-color:{r['bg']}")
            if style:
                t = f'<span style="{"; ".join(style)}">{t}</span>'
            href = self.resolve_run(r.get("link"))
            if href:
                t = f'<a href="{attr(href)}">{t}</a>'
            out.append(t)
        return "".join(out)

    def render_blocks(self, blocks):
        out = []
        for b in blocks:
            t = b["type"]
            if t == "para":
                out.append(f"<p>{self.render_runs(b['runs'])}</p>")
            elif t == "heading":
                lvl = min(max(b["level"], 1), 6)
                hid = f' id="{attr(b["heading_id"])}"' if b.get("heading_id") else ""
                cls = ' class="subtitle"' if b.get("subtitle") else ""
                out.append(f"<h{lvl}{hid}{cls}>{self.render_runs(b['runs'])}</h{lvl}>")
            elif t == "list":
                out.append(self.render_list(b["items"]))
            elif t == "table":
                out.append(self.render_table(b))
            elif t == "toc":
                items = []
                for pb in b.get("blocks", []):
                    if pb["type"] == "para":
                        items.append(f"<li>{self.render_runs(pb['runs'])}</li>")
                out.append('<details class="toc"><summary>Table of contents</summary>'
                           f'<ul>{"".join(items)}</ul></details>')
            elif t == "html":
                out.append(b["html"])
        return "\n".join(out)

    def render_list(self, items):
        parts = []
        stack = []  # [level, lid, tag, css, [[li_inner_parts], ...]]
        def close_list():
            level, lid, tag, css, lis = stack.pop()
            st = f' style="list-style-type:{css}"' if css else ""
            inner = f"<{tag}{st}>" + "".join(
                "<li>" + "".join(p) + "</li>" for p in lis) + f"</{tag}>"
            if stack:
                stack[-1][4][-1].append(inner)
            else:
                parts.append(inner)
        for it in items:
            level, lid = it["level"], it["list_id"]
            tag, css = it["kind"] or ("ul", "")
            while stack and (stack[-1][0] > level
                             or (stack[-1][0] == level and stack[-1][1] != lid)):
                close_list()
            if not stack or stack[-1][0] != level:
                stack.append([level, lid, tag, css, []])
            stack[-1][4].append([self.render_runs(it["runs"])])
        while stack:
            close_list()
        return "".join(parts)

    def render_table(self, b):
        out = ["<table>"]
        for row in b["rows"]:
            out.append("<tr>")
            for cell in row:
                out.append(f"<td>{self.render_blocks(cell)}</td>")
            out.append("</tr>")
        out.append("</table>")
        return "".join(out)


# ---------------------------------------------------------------------------
# source: Google Docs API
# ---------------------------------------------------------------------------

H_LEVELS = {
    "TITLE": 1, "SUBTITLE": 1,
    "HEADING_1": 2, "HEADING_2": 3, "HEADING_3": 4,
    "HEADING_4": 5, "HEADING_5": 6, "HEADING_6": 6,
}

MONO_FONTS = ("courier", "consolas", "menlo", "monaco", "andale",
              "roboto mono", "monospace", "mono")


def is_mono(fam):
    fam = (fam or "").lower()
    return any(m in fam for m in MONO_FONTS)


def rgb_to_css(color_obj):
    try:
        c = color_obj["color"]["rgbColor"]
        return "#%02x%02x%02x" % tuple(
            round(min(1.0, max(0.0, v)) * 255)
            for v in (c["red"], c["green"], c["blue"]))
    except Exception:
        return None


def run_style(style, raw_link):
    r = {"text": ""}
    if raw_link:
        r["link"] = raw_link
    if style.get("bold"):
        r["bold"] = True
    if style.get("italic"):
        r["italic"] = True
    if style.get("underline"):
        r["underline"] = True
    if style.get("strikethrough"):
        r["strike"] = True
    bo = style.get("baselineOffset")
    if bo == "SUPERSCRIPT":
        r["sup"] = True
    elif bo == "SUBSCRIPT":
        r["sub"] = True
    wf = style.get("weightedFontFamily") or {}
    if is_mono(wf.get("fontFamily")):
        r["code"] = True
    fg = rgb_to_css(style.get("foregroundColor")) if style.get("foregroundColor") else None
    if fg:
        r["color"] = fg
    bg = rgb_to_css(style.get("backgroundColor")) if style.get("backgroundColor") else None
    if bg:
        r["bg"] = bg
    return r


def runs_from_text_run(tr):
    style = tr.get("textStyle") or {}
    content = (tr.get("content") or "").replace("\uE907", "")
    if not content:
        return []
    raw_link = parse_api_link(style["link"]) if style.get("link") else None
    base = run_style(style, raw_link)
    out = []
    for i, chunk in enumerate(content.split("\n")):
        if i:
            out.append({"br": True})
        if chunk:
            out.append({**base, "text": chunk})
    return out


def list_kind(ctx, list_id, nesting):
    lst = (ctx.get("lists") or {}).get(list_id) or {}
    levels = (lst.get("listProperties") or {}).get("nestingLevels") or []
    glyph = ""
    if nesting < len(levels):
        glyph = (levels[nesting] or {}).get("glyphType", "")
    if glyph in ("DECIMAL", "DECIMAL_ZERO", "DECIMAL_ZERO_PADDED",
                 "UPPER_ALPHA", "LOWER_ALPHA", "UPPER_ROMAN", "LOWER_ROMAN"):
        css = {"UPPER_ALPHA": "upper-alpha", "LOWER_ALPHA": "lower-alpha",
               "UPPER_ROMAN": "upper-roman", "LOWER_ROMAN": "lower-roman"}.get(glyph, "decimal")
        return ("ol", css)
    return ("ul", "")


def parse_paragraph(p, ctx):
    pstyle = p.get("paragraphStyle") or {}
    named = pstyle.get("namedStyleType", "NORMAL_TEXT")
    level = H_LEVELS.get(named, 0)
    runs = []
    hr = False
    for e in p.get("elements", []):
        if "textRun" in e:
            runs.extend(runs_from_text_run(e["textRun"]))
        elif "inlineObjectElement" in e:
            oid = e["inlineObjectElement"].get("inlineObjectId")
            img = get_image_uri(ctx, oid)
            if img:
                runs.append({"img": img})
        elif "horizontalRule" in e:
            hr = True
        elif "footnoteReference" in e:
            runs.append({"text": "[" + str(e["footnoteReference"].get("footnoteNumber", "")) + "]"})
        elif "person" in e:
            pp = e["person"].get("personProperties") or {}
            email = pp.get("email", "")
            r = {"text": pp.get("name") or email}
            if email:
                r["link"] = ("url", "mailto:" + email)
            runs.append(r)
        elif "richLink" in e:
            rp = e["richLink"].get("richLinkProperties") or {}
            r = {"text": rp.get("title") or "link"}
            if rp.get("uri"):
                r["link"] = ("url", rp["uri"])
            runs.append(r)
        elif "dateElement" in e:
            dp = e["dateElement"].get("dateElementProperties") or {}
            runs.append({"text": dp.get("displayText", "")})
        # pageBreak / columnBreak / autoText / equation: ignored
    if hr:
        return {"type": "html", "html": "<hr>"}
    # drop the paragraph-ending newline (it arrives either as a text run
    # ending in \n or as a trailing <br> run after splitting on newlines)
    while runs and (runs[-1].get("br") or (runs[-1].get("text", "") or "").endswith("\n")):
        if runs[-1].get("br"):
            runs.pop()
        else:
            last = runs[-1]["text"]
            if last == "\n":
                runs.pop()
            else:
                runs[-1] = {**runs[-1], "text": last.rstrip("\n")}
                break
    if not runs:
        return None
    if p.get("bullet"):
        b = p["bullet"]
        return {
            "type": "listitem",
            "list_id": b.get("listId", ""),
            "level": b.get("nestingLevel", 0),
            "kind": list_kind(ctx, b.get("listId", ""), b.get("nestingLevel", 0)),
            "runs": runs,
        }
    if level:
        return {
            "type": "heading",
            "level": level,
            "heading_id": pstyle.get("headingId"),
            "subtitle": named == "SUBTITLE",
            "runs": runs,
        }
    return {"type": "para", "runs": runs}


def get_image_uri(ctx, oid):
    io = (ctx.get("inline_objects") or {}).get(oid) or {}
    emb = (io.get("inlineObjectProperties") or {}).get("embeddedObject") or {}
    return (emb.get("imageProperties") or {}).get("contentUri")


def walk_elements(content, ctx):
    blocks = []
    pending = []

    def flush_list():
        if pending:
            blocks.append({"type": "list", "items": list(pending)})
            pending.clear()

    for el in content or []:
        if "paragraph" in el:
            b = parse_paragraph(el["paragraph"], ctx)
            if b is None:
                continue
            if b["type"] == "listitem":
                pending.append(b)
            else:
                flush_list()
                blocks.append(b)
        elif "table" in el:
            flush_list()
            blocks.append({"type": "table", "rows": parse_table(el["table"], ctx)})
        elif "tableOfContents" in el:
            flush_list()
            toc = walk_elements(el["tableOfContents"].get("content", []), ctx)
            blocks.append({"type": "toc", "blocks": toc})
        # sectionBreak etc.: ignored
    flush_list()
    return blocks


def parse_table(t, ctx):
    rows = []
    for row in t.get("tableRows") or []:
        cells = []
        for cell in row.get("tableCells") or []:
            cells.append(walk_elements(cell.get("content", []), ctx))
        rows.append(cells)
    return rows


def flatten_tab(tab):
    props = tab.get("tabProperties") or {}
    dtab = tab.get("documentTab") or {}
    body = dtab.get("body") or {}
    ctx = {
        "inline_objects": dtab.get("inlineObjects") or {},
        "lists": dtab.get("lists") or {},
    }
    yield {
        "id": props.get("tabId", "t.0"),
        "title": props.get("title", "Tab"),
        "blocks": walk_elements(body.get("content", []), ctx),
    }
    for c in tab.get("childTabs") or []:
        yield from flatten_tab(c)


def parse_document(data):
    doc_id = data.get("documentId", "")
    title = data.get("title") or "Google Doc"
    tabs = []
    if data.get("tabs"):
        for tb in data["tabs"]:
            tabs.extend(flatten_tab(tb))
    else:
        body = data.get("body") or {}
        ctx = {"inline_objects": data.get("inlineObjects") or {},
               "lists": data.get("lists") or {}}
        tabs.append({"id": "t.0", "title": title,
                     "blocks": walk_elements(body.get("content", []), ctx)})
    return title, tabs


# ---------------------------------------------------------------------------
# source: public HTML export
# ---------------------------------------------------------------------------

def parse_css(css_text):
    """Parse a <style> block into {classname: {prop: value}}."""
    rules = {}
    i, n = 0, len(css_text)
    while i < n:
        start = i
        while i < n and css_text[i] != "{":
            i += 1
        if i >= n:
            break
        sel = css_text[start:i].strip()
        i += 1
        depth, dstart = 1, i
        while i < n and depth:
            if css_text[i] == "{":
                depth += 1
            elif css_text[i] == "}":
                depth -= 1
            i += 1
        decl = css_text[dstart:i - 1]
        m = re.fullmatch(r"\.([A-Za-z0-9_-]+)", sel)
        if not m:
            continue
        cls = m.group(1)
        props = rules.setdefault(cls, {})
        for pm in re.finditer(r"([a-zA-Z-]+)\s*:\s*([^;]+);", decl):
            props[pm.group(1).strip().lower()] = pm.group(2).strip()
    return rules


def normalize_color(c):
    c = c.strip()
    m = re.match(r"#([0-9a-fA-F]{3})$", c)
    if m:
        h = m.group(1)
        return "#" + "".join(ch * 2 for ch in h)
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", c)
    if m:
        return "#%02x%02x%02x" % tuple(int(g) for g in m.groups())
    return c


def color_is_dark(color):
    """True for dark, near-neutral colors (black / dark grays) that would be
    unreadable on the dark theme. Such colors are dropped so the text inherits
    the themed foreground. Vivid dark colors (blue links, red warnings) are
    kept — they stay distinguishable on dark and keep their meaning in light."""
    m = re.match(r"#([0-9a-fA-F]{6})$", color or "")
    if not m:
        return False
    h = m.group(1)
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    mx, mn = max(r, g, b), min(r, g, b)
    return mx < 0.5 and (mx - mn) < 0.35


def class_style(cls, css_map):
    props = css_map.get(cls, {})
    d = {}
    fw = props.get("font-weight")
    if fw and (fw == "bold" or (fw.isdigit() and int(fw) >= 700)):
        d["bold"] = True
    if props.get("font-style") == "italic":
        d["italic"] = True
    td = props.get("text-decoration", "")
    if "underline" in td:
        d["underline"] = True
    if "line-through" in td:
        d["strike"] = True
    col = props.get("color")
    if col:
        d["color"] = normalize_color(col)
    bg = props.get("background-color")
    if bg and bg != "transparent":
        d["bg"] = normalize_color(bg)
    if is_mono(props.get("font-family")):
        d["code"] = True
    return d


class ExportParser(HTMLParser):
    """Parse the export HTML into the same block model as the API source."""

    def __init__(self, css_map):
        super().__init__(convert_charrefs=True)
        self.css = css_map
        self.blocks = []
        self.pending = []
        self.inline = []
        self.style_stack = []
        self.style = {}
        self.link_stack = []
        self.heading = None
        self.para = False
        self.li = None            # (tag, level, lid)
        self.list_stack = []
        self.list_count = 0
        self.table = None
        self.table_row = None
        self.table_cell = None
        self.skip_depth = 0       # inside <style>/<script>

    # -- plumbing -----------------------------------------------------------

    def push_style(self, delta):
        self.style_stack.append((delta, self.style))
        self.style = dict(self.style)
        self.style.update(delta)

    def pop_style(self):
        if self.style_stack:
            _, prev = self.style_stack.pop()
            self.style = prev

    def flush_pending_list(self):
        if self.pending:
            self.blocks.append({"type": "list", "items": self.pending})
            self.pending = []

    def flush_inline(self):
        if not self.inline:
            return
        runs = self.inline
        self.inline = []
        if self.heading is not None:
            lvl, hid = self.heading
            self.blocks.append({"type": "heading", "level": lvl,
                                "heading_id": hid, "runs": runs})
        elif self.table is not None:
            self.table_cell.append({"type": "para", "runs": runs})
        elif self.li is not None:
            self.pending.append({"type": "listitem", "list_id": self.li[2],
                                 "level": self.li[1], "kind": (self.li[0], ""),
                                 "runs": runs})
        else:
            self.flush_pending_list()
            self.blocks.append({"type": "para", "runs": runs})

    # -- HTMLParser callbacks ----------------------------------------------

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "script"):
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush_inline()
            self.heading = (int(tag[1]), a.get("id"))
        elif tag == "p":
            self.flush_inline()
            self.para = True
        elif tag == "span":
            delta = {}
            for cls in (a.get("class") or "").split():
                delta.update(class_style(cls, self.css))
            self.push_style(delta)
        elif tag == "a":
            self.link_stack.append(raw_link_from_href(a.get("href")))
        elif tag == "b" or tag == "strong":
            self.push_style({"bold": True})
        elif tag == "i" or tag == "em":
            self.push_style({"italic": True})
        elif tag == "u":
            self.push_style({"underline": True})
        elif tag in ("s", "strike", "del"):
            self.push_style({"strike": True})
        elif tag in ("code", "tt", "kbd"):
            self.push_style({"code": True})
        elif tag == "sub":
            self.push_style({"sub": True})
        elif tag == "sup":
            self.push_style({"sup": True})
        elif tag == "br":
            self.inline.append({"br": True})
        elif tag == "img":
            self.inline.append({"img": a.get("src", "")})
        elif tag in ("ul", "ol"):
            self.flush_inline()
            self.list_stack.append(tag)
            self.list_count += 1
        elif tag == "li":
            self.flush_inline()
            level = max(len(self.list_stack) - 1, 0)
            self.li = (self.list_stack[-1] if self.list_stack else "ul",
                       level, f"l{self.list_count}")
        elif tag == "table":
            self.flush_inline()
            self.flush_pending_list()
            self.table = []
            self.table_row = None
            self.table_cell = None
        elif tag == "tr":
            self.flush_inline()
            self.table_row = []
        elif tag in ("td", "th"):
            self.flush_inline()
            self.table_cell = []
        elif tag == "hr":
            self.flush_inline()
            self.blocks.append({"type": "html", "html": "<hr>"})

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip_depth = max(self.skip_depth - 1, 0)
            return
        if self.skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush_inline()
            self.heading = None
        elif tag == "p":
            self.flush_inline()
            self.para = False
        elif tag == "span":
            self.pop_style()
        elif tag == "a":
            if self.link_stack:
                self.link_stack.pop()
        elif tag in ("b", "strong", "i", "em", "u", "s", "strike", "del",
                     "code", "tt", "kbd", "sub", "sup"):
            self.pop_style()
        elif tag == "li":
            self.flush_inline()
            self.li = None
        elif tag in ("ul", "ol"):
            self.flush_inline()
            if self.list_stack:
                self.list_stack.pop()
        elif tag == "td" or tag == "th":
            self.flush_inline()
            if self.table_row is not None and self.table_cell is not None:
                self.table_row.append(self.table_cell)
            self.table_cell = None
        elif tag == "tr":
            self.flush_inline()
            if self.table is not None and self.table_row is not None:
                self.table.append(self.table_row)
            self.table_row = None
        elif tag == "table":
            self.flush_inline()
            if self.table is not None:
                self.blocks.append({"type": "table", "rows": self.table})
            self.table = None

    def handle_data(self, data):
        if self.skip_depth or not data:
            return
        r = {"text": data}
        r.update(self.style)
        if self.link_stack:
            r["link"] = self.link_stack[-1]
        self.inline.append(r)


def is_toc_para(b):
    """A paragraph that is exactly one internal heading link (a TOC entry)."""
    runs = b.get("runs", [])
    if len(runs) != 1:
        return False
    r = runs[0]
    lnk = r.get("link")
    return bool(lnk and lnk[0] == "heading" and (r.get("text", "") or "").strip())


def collapse_toc(blocks):
    """Google's HTML export renders an in-document table of contents as plain
    single-link paragraphs. Group runs of them into one collapsible block so
    a 2,500-entry TOC doesn't blow up a page."""
    out = []
    i, n = 0, len(blocks)
    while i < n:
        if blocks[i]["type"] == "para" and is_toc_para(blocks[i]):
            j = i
            while j < n and blocks[j]["type"] == "para" and is_toc_para(blocks[j]):
                j += 1
            if j - i >= 5:
                out.append({"type": "toc", "blocks": blocks[i:j]})
                i = j
                continue
        out.append(blocks[i])
        i += 1
    return out


def parse_export(html_text, title="Google Doc", doc_id=""):
    css_map = {}
    for m in re.finditer(r"<style[^>]*>(.*?)</style>", html_text, re.S):
        css_map.update(parse_css(m.group(1)))
    parser = ExportParser(css_map)
    parser.feed(html_text)
    parser.close()
    blocks = collapse_toc(parser.blocks)
    return title, [{"id": "t.0", "title": title, "blocks": blocks}]


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def http_get(url, timeout=180, headers=None):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch_export(doc):
    return http_get(EXPORT_URL.format(doc=doc))


def token_post(params):
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_auth(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_auth(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def pkce_pair():
    import base64
    import hashlib
    import secrets
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def get_access_token(auth):
    tok = token_post({
        "client_id": auth["client_id"],
        "client_secret": auth["client_secret"],
        "refresh_token": auth["refresh_token"],
        "grant_type": "refresh_token",
    })
    if "access_token" not in tok:
        sys.stderr.write("Auth failed: " + json.dumps(tok)[:300] + "\n")
        sys.exit(1)
    return tok["access_token"]


def fetch_api(doc, auth):
    url = DOCS_API.format(doc=doc)
    token = get_access_token(auth)
    try:
        return json.loads(http_get(url, timeout=120,
                                   headers={"Authorization": "Bearer " + token}))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        if e.code in (401, 403):
            sys.stderr.write("Google rejected the saved authorization. Re-run:\n"
                             f"  python gdoc_site.py --auth --client-json <client_secret_*.json>\n")
            sys.exit(1)
        sys.stderr.write(f"Docs API error {e.code} (a very large document may "
                         f"be larger than the API can serve): {body[:200]}\n")
        return None
    except Exception as e:
        sys.stderr.write(f"Docs API fetch failed: {e}\n")
        return None


def run_auth_flow(client_json_path, port=8912):
    """One-time OAuth: open a browser, let the user click Allow, save tokens."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    with open(client_json_path, encoding="utf-8") as f:
        cfg = json.load(f)
    info = cfg.get("installed") or cfg.get("web") or {}
    client_id = info.get("client_id") or cfg.get("client_id")
    client_secret = info.get("client_secret") or cfg.get("client_secret")
    if not client_id or not client_secret:
        sys.exit(f"Could not find client_id/client_secret in {client_json_path}")

    redirect_uri = f"http://127.0.0.1:{port}"
    result = {"code": None, "error": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in q:
                result["code"] = q["code"][0]
                body = (b"<html><body style='font-family:sans-serif;margin:2em'>"
                        b"<h2>Authorized - you can close this tab.</h2></body></html>")
            elif "error" in q:
                result["error"] = q["error"][0]
                body = (f"<html><body><h2>Authorization failed: "
                        f"{q['error'][0]}</h2></body></html>").encode()
            else:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    verifier, challenge = pkce_pair()
    auth_url = AUTH_URL.format(
        client_id=urllib.parse.quote(client_id, safe=""),
        redirect_uri=urllib.parse.quote(redirect_uri, safe=""),
        scope=urllib.parse.quote(OAUTH_SCOPE, safe=""),
        challenge=challenge)
    server = HTTPServer(("127.0.0.1", port), Handler)
    print("Opening your browser…")
    print("If nothing opens, visit this URL and click Allow:\n  " + auth_url)
    webbrowser.open(auth_url)
    deadline = time.time() + 300
    while (result["code"] is None and result["error"] is None
           and time.time() < deadline):
        server.handle_request()
    server.server_close()
    if result["error"]:
        sys.exit(f"Authorization failed: {result['error']}")
    if not result["code"]:
        sys.exit("Timed out after 5 minutes waiting for authorization.")
    tok = token_post({
        "code": result["code"],
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": verifier,
    })
    if "refresh_token" not in tok:
        sys.exit("No refresh token returned: " + json.dumps(tok)[:300])
    return {"client_id": client_id, "client_secret": client_secret,
            "refresh_token": tok["refresh_token"]}


# ---------------------------------------------------------------------------
# page templates
# ---------------------------------------------------------------------------

STYLE_CSS = """\
:root {
  /* dark is the default theme */
  --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e;
  --sidebar-bg: #161b22; --border: #30363d;
  --accent: #4493f8; --accent-soft: #1f3a5f;
  --code-bg: #1c2128; --table-border: #30363d;
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #1f2328; --muted: #656d76;
  --sidebar-bg: #f6f8fa; --border: #d8dee4;
  --accent: #0969da; --accent-soft: #ddf4ff;
  --code-bg: #eff1f3; --table-border: #d0d7de;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
a { color: var(--accent); text-decoration: none; font-weight: 600; }
a:hover { text-decoration: underline; }

.topbar {
  position: sticky; top: 0; z-index: 30;
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; background: var(--sidebar-bg);
  border-bottom: 1px solid var(--border);
}
.topbar .brand { font-weight: 600; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; color: var(--fg); }
#nav-toggle { display: none; background: none; border: 1px solid var(--border);
  border-radius: 6px; color: var(--fg); font-size: 18px; padding: 2px 10px; cursor: pointer; }
#search { flex: 1; max-width: 420px; margin-left: auto;
  padding: 7px 12px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg); color: var(--fg); font-size: 14px; }
#theme-toggle { background: none; border: 1px solid var(--border);
  border-radius: 6px; color: var(--fg); font-size: 16px; line-height: 1;
  padding: 3px 9px; cursor: pointer; flex: 0 0 auto; }

.layout { display: flex; align-items: stretch; }
.sidebar {
  width: 300px; flex: 0 0 300px; border-right: 1px solid var(--border);
  background: var(--sidebar-bg); max-height: calc(100vh - 49px);
  position: sticky; top: 49px; overflow-y: auto; padding: 12px 10px 40px;
}
.sidebar-brand { display: block; font-weight: 700; margin: 4px 8px 10px; color: var(--fg); }
.tab-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 4px 12px; }
.chip { font-size: 13px; padding: 3px 10px; border: 1px solid var(--border);
  border-radius: 999px; background: var(--bg); color: var(--fg); }
.toc { list-style: none; margin: 0; padding: 0; }
.toc ul { list-style: none; margin: 0; padding-left: 14px; }
.toc-group { margin: 10px 0; }
.toc-tab { font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); display: block; margin: 0 6px 4px; }
.toc-section { display: block; padding: 3px 6px; border-radius: 6px;
  font-size: 14px; color: var(--fg); }
.toc-section:hover { background: var(--accent-soft); text-decoration: none; }
.toc-section.current { background: var(--accent-soft); font-weight: 600; }
.toc-subs a { display: block; padding: 1px 6px; font-size: 13px; color: var(--muted); }
.toc-subs a:hover { color: var(--fg); text-decoration: none; }
.toc-sub.current { color: var(--fg); font-weight: 600; }
.toc a, .toc .toc-tab { display: block; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; max-width: 100%; }
.toc-item { margin: 0; }
.toc-head { display: flex; align-items: center; gap: 2px; }
.toc-head .toc-section { flex: 1 1 auto; min-width: 0; }
.toc-caret { background: none; border: none; color: var(--muted); cursor: pointer;
  font-size: 10px; line-height: 1; padding: 5px 6px; transition: transform .12s ease; }
.toc-caret.open { transform: rotate(90deg); }
.toc-subs { display: none; }
.toc-subs.open { display: block; }
.toc-preview { position: fixed; z-index: 60; pointer-events: none;
  max-width: min(340px, calc(100vw - 32px)); padding: 8px 12px; border-radius: 8px;
  background: var(--bg); color: var(--fg); border: 1px solid var(--border);
  box-shadow: 0 4px 16px rgba(0, 0, 0, .25); font-size: 13px; line-height: 1.45;
  opacity: 0; transform: translateY(3px);
  transition: opacity .1s ease, transform .1s ease; }
.toc-preview.show { opacity: 1; transform: translateY(0); }
.sidebar-foot { margin: 18px 6px 0; font-size: 12px; color: var(--muted); }

.content { flex: 1; min-width: 0; padding: 28px 32px 80px; }
@media (min-width: 1000px) { .content { max-width: 820px; margin: 0 auto; } }

.crumbs { font-size: 14px; color: var(--muted); margin-bottom: 8px; }
h1 { font-size: 26px; line-height: 1.3; margin: 4px 0 10px; font-weight: 400; }
.meta { color: var(--muted); font-size: 14px; margin: 0 0 18px; }
.notice { background: var(--accent-soft); border-radius: 8px; padding: 10px 14px;
  font-size: 14px; margin: 0 0 26px; }
#search-results { margin: 12px 0; }
#search-results .sr { display: block; padding: 8px 12px; border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 8px; background: var(--bg); }
#search-results .sr .t { font-weight: 600; color: var(--fg); }
#search-results .sr .tab { font-size: 12px; color: var(--muted); margin-left: 8px; }
#search-results .sr .sn { font-size: 13px; color: var(--muted); display: block; margin-top: 2px; }

.doc { font-size: 15px; line-height: 1.6; overflow-wrap: break-word; }
.doc h2 { font-size: 20px; margin: 1.6em 0 .6em; padding-top: .3em;
  border-bottom: 1px solid var(--border); font-weight: 600; }
.doc h3 { font-size: 17px; margin: 1.4em 0 .5em; font-weight: 600; }
.doc h4, .doc h5, .doc h6 { font-size: 16px; margin: 1.2em 0 .3em; }
.doc h1.subtitle, .doc h2.subtitle { border-bottom: none; font-weight: 500;
  color: var(--muted); font-size: 18px; }
.doc p { margin: .8em 0; line-height: 1.6; }
.doc ul, .doc ol { padding-left: 1.6em; margin: .6em 0; }
.doc li { margin: .25em 0; }
.doc code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .88em; background: var(--code-bg); padding: .1em .35em; border-radius: 4px; }
.doc a { color: var(--accent) !important; }
.doc a span { color: inherit !important; }
.doc a code { color: inherit; }
.doc table { border-collapse: collapse; margin: 1em 0; max-width: 100%;
  display: block; overflow-x: auto; }
.doc td, .doc th { border: 1px solid var(--table-border); padding: 6px 10px;
  vertical-align: top; }
.doc img { max-width: 100%; height: auto; }
.doc hr { border: none; border-top: 1px solid var(--border); margin: 1.4em 0; }
.doc details.toc { border: 1px solid var(--border); border-radius: 8px;
  margin: 1em 0; }
.doc details.toc summary { cursor: pointer; padding: 8px 14px; font-weight: 600;
  user-select: none; }
.doc details.toc ul { list-style: none; margin: 0; padding: 0 18px 12px;
  columns: 2; column-gap: 2em; }
.doc details.toc li { margin: .2em 0; font-size: 14px; }
@media (max-width: 700px) { .doc details.toc ul { columns: 1; } }

.section-list { display: grid; gap: 10px; margin-top: 6px; }
.section-card { display: flex; align-items: baseline; gap: 8px;
  border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; color: var(--fg); }
.section-card:hover { border-color: var(--accent); text-decoration: none; }
.section-card .t { flex: 1 1 auto; min-width: 0; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.section-card .n { flex: 0 0 auto; font-size: 12px; color: var(--muted); white-space: nowrap; }
.card-children { padding: 2px 14px 10px; font-size: 13px; color: var(--muted); }
.card-children a { display: block; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; padding: 1px 0; }

.index-link { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }
.index-link a { color: var(--accent); font-size: 14px; }
.pager { display: flex; justify-content: space-between; gap: 10px;
  margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); }
.pager a { border: 1px solid var(--border); border-radius: 8px; padding: 8px 14px;
  color: var(--fg); }
.pager a:hover { border-color: var(--accent); text-decoration: none; }
.pager .next { margin-left: auto; }

footer { margin-top: 48px; font-size: 13px; color: var(--muted); }

@media (max-width: 900px) {
  #nav-toggle { display: block; }
  .sidebar { position: fixed; top: 49px; bottom: 0; left: 0; z-index: 20;
    transform: translateX(-100%); transition: transform .18s ease; max-height: none; }
  .sidebar.open { transform: translateX(0); }
  .content { padding: 20px 18px 60px; max-width: none; margin: 0; }
}
@media print {
  .topbar, .sidebar, .pager, .crumbs { display: none !important; }
  .content { max-width: none; padding: 0; margin: 0; }
}
"""

APP_JS = """\
(function () {
  "use strict";
  var data = null;
  var results = [];

  function qs(sel) { return document.querySelector(sel); }

  // --- theme toggle (persisted in localStorage; default dark) -------------
  var themeBtn = qs("#theme-toggle");
  if (themeBtn) {
    function applyTheme(t) {
      document.documentElement.setAttribute("data-theme", t);
      themeBtn.textContent = t === "light" ? "☀" : "☾";
      try { localStorage.setItem("doc-theme", t); } catch (e) {}
    }
    themeBtn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      applyTheme(cur === "light" ? "dark" : "light");
    });
    applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
  }

  // --- collapsible TOC groups --------------------------------------------
  document.querySelectorAll(".toc-head").forEach(function (head) {
    var caret = head.querySelector(".toc-caret");
    if (!caret) return;
    caret.addEventListener("click", function (e) {
      var li = head.parentNode;
      var subs = li && li.querySelector(".toc-subs");
      if (subs) {
        var open = subs.classList.toggle("open");
        caret.classList.toggle("open", open);
      }
      e.preventDefault();
    });
  });

  // --- truncated-title preview -------------------------------------------
  var pv = document.createElement("div");
  pv.className = "toc-preview";
  document.body.appendChild(pv);
  var isTrunc = function (el) { return el.scrollWidth > el.clientWidth + 1; };
  function showPreview(a) {
    pv.textContent = a.textContent.replace(/\s+/g, " ").trim();
    pv.classList.add("show");
    var r = a.getBoundingClientRect();
    var pw = pv.offsetWidth, ph = pv.offsetHeight;
    var left = Math.max(8, r.right + 10);
    if (left + pw > window.innerWidth - 8) left = Math.max(8, r.left - pw - 10);
    pv.style.left = left + "px";
    pv.style.top = Math.max(8, Math.min(r.top, window.innerHeight - ph - 8)) + "px";
  }
  function hidePreview() { pv.classList.remove("show"); }
  document.querySelectorAll(".toc a").forEach(function (a) {
    a.addEventListener("mouseenter", function () { if (isTrunc(a)) showPreview(a); });
    a.addEventListener("mouseleave", hidePreview);
    a.addEventListener("focus", function () { if (isTrunc(a)) showPreview(a); });
    a.addEventListener("blur", hidePreview);
  });
  var sbEl = qs(".sidebar");
  if (sbEl) sbEl.addEventListener("scroll", hidePreview);
  window.addEventListener("resize", hidePreview);

  // sidebar toggle (mobile)
  var toggle = qs("#nav-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var sb = qs(".sidebar");
      if (sb) sb.classList.toggle("open");
    });
  }

  // --- search -------------------------------------------------------------
  var input = qs("#search");
  if (input) {
    var box = document.createElement("div");
    box.id = "search-results";
    var host = qs("#search-host") || document.body;
    host.insertBefore(box, host.firstChild);

    fetch("data.json").then(function (r) { return r.json(); }).then(function (d) {
      data = d;
      d.sections.forEach(function (s) {
        var hay = (s.tabTitle + " " + s.title + " " +
          (s.subs || []).map(function (x) { return x.title; }).join(" ") + " " + s.text).toLowerCase();
        s._hay = hay;
      });
    }).catch(function () {});

    function render(list) {
      box.innerHTML = "";
      list.forEach(function (s) {
        var a = document.createElement("a");
        a.className = "sr";
        a.href = s.slug + ".html";
        var t = document.createElement("span");
        t.className = "t";
        t.textContent = s.title;
        var tab = document.createElement("span");
        tab.className = "tab";
        tab.textContent = s.tabTitle;
        var sn = document.createElement("span");
        sn.className = "sn";
        sn.textContent = (s.text || "").slice(0, 160);
        a.appendChild(t); a.appendChild(tab); a.appendChild(sn);
        box.appendChild(a);
      });
    }

    function search() {
      var val = input.value.trim().toLowerCase();
      if (!val || !data) { box.innerHTML = ""; return; }
      var tokens = val.split(/\\s+/);
      var out = [];
      data.sections.forEach(function (s) {
        if (tokens.every(function (tk) { return s._hay.indexOf(tk) !== -1; })) {
          var title = s.title.toLowerCase();
          var score = 0;
          if (title.indexOf(val) === 0) score -= 200;
          else if (title.indexOf(val) !== -1) score -= 100;
          score += s._hay.indexOf(tokens[0]);
          out.push({ s: s, score: score });
        }
      });
      out.sort(function (a, b) { return a.score - b.score; });
      render(out.slice(0, 25).map(function (o) { return o.s; }));
    }

    input.addEventListener("input", search);
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== input) {
        e.preventDefault(); input.focus();
      }
      if (e.key === "Escape") { input.value = ""; search(); input.blur(); }
    });
  }
})();
"""


def page_template(site, title, sidebar, body, active_slug):
    meta = f"{len(site.sections)} sections · generated {site.generated}"
    feed_link = ('<link rel="alternate" type="application/rss+xml" title="RSS feed" '
                 'href="feed.xml">\n') if getattr(site, "base_url", "") else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{feed_link}<link rel="stylesheet" href="assets/style.css">
<script>(function(){{var t;try{{t=localStorage.getItem("doc-theme");}}catch(e){{}}document.documentElement.setAttribute("data-theme",t||"dark");}})();</script>
</head>
<body>
<header class="topbar">
<button id="nav-toggle" aria-label="Toggle navigation">☰</button>
<a class="brand" href="index.html">{esc(site.title)}</a>
<input id="search" type="search" placeholder="Search this document…" autocomplete="off">
<button id="theme-toggle" aria-label="Toggle color theme" title="Toggle light/dark theme">☾</button>
</header>
<div class="layout">
<nav class="sidebar">{sidebar}</nav>
<main class="content">{body}
<footer>Mirror of <a href="https://docs.google.com/document/d/{attr(site.doc_id)}/edit">the Google Doc</a> — {esc(meta)}</footer>
</main>
</div>
<script src="assets/app.js"></script>
</body>
</html>
"""


def sidebar_html(site, active_slug=None):
    children = {}
    for s in site.sections:
        if s.get("parent"):
            children.setdefault(s["parent"], []).append(s)
    p = ['<div class="sidebar-inner">']
    p.append(f'<a class="sidebar-brand" href="index.html">{esc(site.title)}</a>')
    if len(site.tabs) > 1:
        p.append('<div class="tab-chips">')
        for t in site.tabs:
            p.append(f'<a class="chip" href="{site.tab_url(t["id"])}">{esc(t["title"])}</a>')
        p.append('</div>')
    p.append('<ul class="toc">')
    for t in site.tabs:
        roots = [s for s in site.sections
                 if s["tab"] == t["id"] and not s.get("parent")]
        if not roots:
            continue
        p.append(f'<li class="toc-group"><span class="toc-tab">{esc(t["title"])}</span><ul>')
        for s in roots:
            cls = ' current' if s["slug"] == active_slug else ""
            link = (f'<a class="toc-section{cls}" href="{s["slug"]}.html"'
                     f'>{esc(s["title"])}</a>')
            kids = children.get(s["slug"], [])
            subs = [x for x in s.get("subs", []) if x["level"] == 3]
            if kids or subs:
                # expanded by default only for the section you're currently in
                open_ = (s["slug"] == active_slug
                         or any(k["slug"] == active_slug for k in kids))
                subs_cls = " open" if open_ else ""
                caret_cls = " open" if open_ else ""
                p.append('<li class="toc-item"><div class="toc-head">'
                         f'<button class="toc-caret{caret_cls}" type="button" '
                         f'aria-label="Toggle sub-headings">▸</button>{link}</div>')
                p.append(f'<ul class="toc-subs{subs_cls}">')
                for k in kids:
                    ccls = ' current' if k["slug"] == active_slug else ""
                    p.append(f'<li><a class="toc-sub{ccls}" href="{k["slug"]}.html"'
                             f'>{esc(k["title"])}</a></li>')
                for sub in subs:
                    href = f'{s["slug"]}.html#{sub["id"]}' if sub.get("id") else f'{s["slug"]}.html'
                    p.append(f'<li><a href="{href}">{esc(sub["title"])}</a></li>')
                p.append('</ul></li>')
            else:
                p.append(f'<li class="toc-item">{link}</li>')
        p.append('</ul></li>')
    p.append('</ul>')
    p.append(f'<div class="sidebar-foot"><a href="index.html">Index</a> · '
             f'<a href="https://docs.google.com/document/d/{attr(site.doc_id)}/edit">Google Doc ↗</a></div>')
    p.append('</div>')
    return "".join(p)


def snippet(section, limit=400):
    parts = []
    for b in section["blocks"]:
        if b["type"] in ("para", "heading"):
            t = text_of_runs(b["runs"])
            if t:
                parts.append(t)
        elif b["type"] == "list":
            for it in b["items"]:
                t = text_of_runs(it["runs"])
                if t:
                    parts.append(t)
        elif b["type"] == "html":
            parts.append(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", b["html"])).strip())
    return " ".join(parts)[:limit]


# ---------------------------------------------------------------------------
# site writing
# ---------------------------------------------------------------------------

def render_index(site):
    if not site.sections:
        return page_template(site, site.title, sidebar_html(site), '<p>No content.</p>', None)
    first = site.sections[0]
    body = []
    blocks = first["blocks"]
    if (blocks and blocks[0]["type"] == "heading"
            and blocks[0].get("heading_id") == first.get("heading_id")):
        blocks = blocks[1:]
    body.append('<div class="doc">')
    body.append(site.render_blocks(blocks))
    body.append('</div>')
    body.append('<div class="index-link"><a href="contents.html">Full table of contents →</a></div>')
    return page_template(site, site.title, sidebar_html(site), chr(10).join(body), None)


def render_contents(site):
    body = ['<h1>Contents</h1>']
    children = {}
    for s in site.sections:
        if s.get("parent"):
            children.setdefault(s["parent"], []).append(s)
    for t in site.tabs:
        roots = [s for s in site.sections
                 if s["tab"] == t["id"] and not s.get("parent")]
        if not roots:
            continue
        body.append('<div class="section-list">')
        for s in roots:
            body.append(f'<a class="section-card" href="{s["slug"]}.html" title="{attr(s["title"])}"><span class="t">{esc(s["title"])}</span></a>')
            kids = children.get(s["slug"], [])
            if kids:
                body.append('<div class="card-children">')
                for k in kids:
                    body.append(f'<a href="{k["slug"]}.html">{esc(k["title"])}</a>')
                body.append('</div>')
        body.append('</div>')
    return page_template(site, site.title, sidebar_html(site), chr(10).join(body), None)


def render_section_page(site, i, sec):
    total = len(site.sections)
    body = [f'<nav class="crumbs"><a href="index.html">{esc(site.title)}</a> › '
            f'<span>{esc(sec["tab_title"])}</span>'
            + (f' › <a href="{sec["parent"]}.html">'
               + esc(next((s["title"] for s in site.sections
                           if s["slug"] == sec["parent"]), sec["parent"]))
               + '</a>' if sec.get("parent") else '')
            + '</nav>']
    body.append(f'<h1>{esc(sec["title"])}</h1>')
    body.append(f'<p class="meta">Page {i + 1} of {total} · '
                f'<a href="{attr(site.edit_url(sec))}" target="_blank" rel="noopener">'
                f'Edit this page in Google Docs ↗</a></p>')
    # the page title is already shown as <h1>; skip the section's own
    # title heading block so it isn't repeated inside the body
    blocks = sec["blocks"]
    if (blocks and blocks[0]["type"] == "heading"
            and blocks[0].get("heading_id") == sec.get("heading_id")):
        blocks = blocks[1:]
    body.append('<div class="doc">')
    body.append(site.render_blocks(blocks))
    body.append('</div>')
    prev = site.sections[i - 1] if i > 0 else None
    nxt = site.sections[i + 1] if i + 1 < total else None
    pager = ['<nav class="pager">']
    if prev:
        pager.append(f'<a class="prev" href="{prev["slug"]}.html">← {esc(prev["title"])}</a>')
    if nxt:
        pager.append(f'<a class="next" href="{nxt["slug"]}.html">{esc(nxt["title"])} →</a>')
    pager.append('</nav>')
    body.append("".join(pager))
    return page_template(site, f"{sec['title']} — {site.title}",
                         sidebar_html(site, sec["slug"]), "\n".join(body), sec["slug"])


def data_json(site):
    return json.dumps({
        "title": site.title,
        "doc": site.doc_id,
        "generated": site.generated,
        "source": site.source,
        "tabs": [{"id": t["id"], "title": t["title"]} for t in site.tabs],
        "sections": [
            {
                "slug": s["slug"],
                "title": s["title"],
                "tab": s["tab"],
                "tabTitle": s["tab_title"],
                "headingId": s.get("heading_id"),
                "parent": s.get("parent"),
                "depth": s.get("depth", 0),
                "subs": s.get("subs", []),
                "text": s.get("text", ""),
            }
            for s in site.sections
        ],
    }, ensure_ascii=False)


def _rfc2822():
    import email.utils
    return email.utils.formatdate(time.time(), usegmt=True)


def _esc_xml(s):
    from xml.sax.saxutils import escape
    return escape(s)


def write_seo_files(site, out, base_url):
    """Emit sitemap.xml and an RSS 2.0 feed (feed.xml). Requires base_url."""
    if not base_url:
        return False
    base_url = base_url.rstrip("/")
    stamp = _rfc2822()

    urls = ["index.html"] + [s["slug"] + ".html" for s in site.sections]
    locs = "\n".join(
        "  <url><loc>%s</loc></url>" % _esc_xml(base_url + "/" + u)
        for u in urls)
    sitemap = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
               + locs + "\n</urlset>\n")
    with open(os.path.join(out, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap)

    items = []
    for s in site.sections:
        loc = base_url + "/" + s["slug"] + ".html"
        desc = _esc_xml((s.get("text") or "")[:300])
        items.append(
            "  <item>\n"
            "    <title>%s</title>\n" % _esc_xml(s["title"])
            + "    <link>%s</link>\n" % _esc_xml(loc)
            + '    <guid isPermaLink="true">%s</guid>\n' % _esc_xml(loc)
            + "    <description>%s</description>\n" % desc
            + "    <pubDate>%s</pubDate>\n" % stamp
            + "  </item>")
    feed = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
            "<channel>\n"
            "  <title>%s</title>\n" % _esc_xml(site.title)
            + "  <link>%s/</link>\n" % _esc_xml(base_url)
            + "  <description>Fast, readable mirror of a Google Doc — %s. </description>\n"
                % _esc_xml(site.title)
            + "  <language>en</language>\n"
            + "  <generator>doc-site (gdoc_site.py)</generator>\n"
            + "  <lastBuildDate>%s</lastBuildDate>\n" % stamp
            + "\n".join(items) + "\n"
            + "</channel>\n"
            + "</rss>\n")
    with open(os.path.join(out, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(feed)
    return True


def write_site(site, out):
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)
    with open(os.path.join(out, "assets", "style.css"), "w", encoding="utf-8") as f:
        f.write(STYLE_CSS)
    with open(os.path.join(out, "assets", "app.js"), "w", encoding="utf-8") as f:
        f.write(APP_JS)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_index(site))
    with open(os.path.join(out, "contents.html"), "w", encoding="utf-8") as f:
        f.write(render_contents(site))
    for i, sec in enumerate(site.sections):
        with open(os.path.join(out, f"{sec['slug']}.html"), "w", encoding="utf-8") as f:
            f.write(render_section_page(site, i, sec))
    with open(os.path.join(out, "data.json"), "w", encoding="utf-8") as f:
        f.write(data_json(site))
    if write_seo_files(site, out, getattr(site, "base_url", "")):
        print("wrote:      sitemap.xml, feed.xml")
    else:
        # remove any stale sitemap/feed from an earlier build with --base-url
        for stale in ("sitemap.xml", "feed.xml"):
            p = os.path.join(out, stale)
            if os.path.exists(p):
                os.remove(p)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", help="Google Docs document ID (from the URL)")
    ap.add_argument("--source", choices=["api", "export", "file"],
                    help="default: api if auth.json exists, else export")
    ap.add_argument("--file", help="path to a saved documents.get JSON (for --source file)")
    ap.add_argument("--title", help="override the document title (export mode needs this)")
    ap.add_argument("--out", default="site", help="output directory (default: site)")
    ap.add_argument("--auth", action="store_true",
                    help="one-time OAuth authorization (saves --auth-file)")
    ap.add_argument("--client-json",
                    help="path to the client_secret_*.json downloaded from Google Cloud (for --auth)")
    ap.add_argument("--auth-file", default="auth.json",
                    help="where to store the OAuth token (default: auth.json)")
    ap.add_argument("--port", type=int, default=8912,
                    help="loopback port used by --auth (default: 8912)")
    ap.add_argument("--base-url",
                    help="public URL of the deployed site (e.g. https://example.com/) — "
                         "required to emit sitemap.xml and feed.xml")
    args = ap.parse_args(argv)

    if args.auth:
        if not args.client_json:
            ap.error("--auth needs --client-json PATH "
                     "(the client_secret_*.json from Google Cloud — see README.md)")
        if not os.path.exists(args.client_json):
            ap.error(f"--client-json file not found: {args.client_json}")
        auth = run_auth_flow(args.client_json, args.port)
        save_auth(args.auth_file, auth)
        print(f"Saved credentials to {args.auth_file}. You can now build with:")
        print(f"  python gdoc_site.py --doc <DOC_ID> --source api")
        return 0

    if not args.doc and args.source != "file":
        ap.error("--doc is required (except with --source file)")
    if args.source == "file" and not args.file:
        ap.error("--source file requires --file PATH")

    # Default to the fast, reliable export path. Tabs (via the API) are opt-in
    # with --source api; the API cannot serve very large documents and falls
    # back to export automatically when it times out.
    source = args.source or "export"

    if source == "api":
        auth = load_auth(args.auth_file)
        if not auth:
            ap.error("--source api needs authorization first:\n"
                     "  python gdoc_site.py --auth --client-json <client_secret_*.json>\n"
                     "(see README.md — Google no longer accepts API keys for the Docs API)")
        data = fetch_api(args.doc, auth)
        if data is not None:
            title, tabs = parse_document(data)
        else:
            sys.stderr.write("Falling back to the (flattened) HTML export.\n")
            html_text = fetch_export(args.doc)
            title, tabs = parse_export(html_text, title=args.title or "Google Doc",
                                       doc_id=args.doc)
            source = "export"
    elif source == "file":
        with open(args.file, encoding="utf-8") as f:
            data = json.load(f)
        title, tabs = parse_document(data)
        args.doc = args.doc or data.get("documentId", "")
    else:
        html_text = fetch_export(args.doc)
        title, tabs = parse_export(html_text, title=args.title or "Google Doc",
                                   doc_id=args.doc)

    if args.title:
        title = args.title

    site = Site(args.doc, title, tabs, source)
    site.base_url = args.base_url or ""
    for s in site.sections:
        s["text"] = snippet(s)

    t0 = time.time()
    write_site(site, args.out)
    if not site.base_url:
        print("(pass --base-url https://your.domain/ to also emit sitemap.xml + feed.xml)")

    sizes = []
    for root, _, files in os.walk(args.out):
        for fn in files:
            sizes.append(os.path.getsize(os.path.join(root, fn)))
    total = sum(sizes)
    biggest = max(sizes) if sizes else 0
    print(f"title:      {site.title}")
    print(f"source:     {source}")
    print(f"tabs:       {len(site.tabs)}")
    print(f"sections:   {len(site.sections)}  ->  {args.out}/")
    print(f"pages:      {len(site.sections) + 1}")
    print(f"size:       {total / 1024:.0f} KiB total, largest page {biggest / 1024:.0f} KiB")
    print(f"time:       {time.time() - t0:.1f}s")
    print(f"\nPreview:  python serve.py --dir {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
