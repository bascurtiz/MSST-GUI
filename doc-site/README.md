# Google Doc to Static Site Mirror

Converts a large Google Doc into a fast, lightweight static site with:
- One page per section (65+ pages for a 675-page doc)
- Sidebar table of contents (collapsible groups)
- Client-side search
- Light/dark theme toggle (persisted in localStorage)
- Working internal links
- "Edit this page" links back to Google Docs
- RSS feed and sitemap for SEO

## Quick Start

### Local development
```bash
# Install Python 3.10+ if not already installed

# Generate the site
python gdoc_site.py --doc 17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c --out site

# Preview locally
python serve.py --dir site
# Open http://localhost:8000
```

### Deploy to GitHub Pages (free hosting + auto-updates)

1. **Create a GitHub repo**
   ```bash
   cd D:\Downloads\msst_gui_neo\doc-site
   git init
   git add -A
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin master
   ```

2. **Enable GitHub Pages**
   - Go to your repo on GitHub
   - Settings → Pages
   - Source: GitHub Actions
   - Save

3. **That's it!** The site will:
   - Deploy automatically on first push
   - Update every hour via GitHub Actions
   - Be available at `https://YOUR_USERNAME.github.io/YOUR_REPO/`

### Manual deployment to other hosts

```bash
# Generate with your domain (for sitemap.xml + feed.xml)
python gdoc_site.py \
  --doc 17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c \
  --base-url https://your.domain/ \
  --out site

# Upload everything inside site/ to your web root
```

## Features

- **Fast loading** — 66 lightweight pages instead of one 13MB document
- **Dark mode** — default theme with light/dark toggle
- **Collapsible TOC** — sidebar groups expand/collapse
- **Search** — client-side full-text search
- **Responsive** — works on mobile and desktop
- **SEO ready** — sitemap.xml and RSS feed (with --base-url)

## CLI Options

```
python gdoc_site.py --doc DOC_ID [OPTIONS]

Options:
  --doc DOC_ID       Google Doc ID (required)
  --out DIR          Output directory (default: site)
  --source SOURCE    api | export | file (default: auto)
  --base-url URL     Your deployed URL (for sitemap/feed)
  --file PATH        Use saved JSON response (source=file)
  --auth             Run OAuth setup
  --client-json PATH Path to client_secret_*.json
  --auth-file PATH   Path to save/load tokens (default: auth.json)
  --title TITLE      Override document title
```

## How it works

1. Fetches the Google Doc (via API or HTML export)
2. Parses into sections based on headings
3. Generates one HTML page per section
4. Creates sidebar TOC, search index, and navigation
5. Deploys to GitHub Pages (or any static host)

## Credentials

- `auth.json` — OAuth tokens (auto-generated on first run)
- `client_secret_*.json` — Google API credentials (keep private!)

**Never commit these files** — they're in .gitignore.

## Troubleshooting

**"Google hasn't verified this app"**
- Normal for test apps. Click Advanced → Go to [app name] (unsafe)

**Auth expired**
- Re-run: `python gdoc_site.py --auth --client-json client_secret_*.json`

**Site not updating**
- Check GitHub Actions tab for errors
- Ensure Google Doc is public ("Anyone with the link")

## License

MIT
