"""
backend/update_checker.py
-------------------------
GitHub release checking (network + version logic only — no GUI imports).

Mirrors the STEM-organizer update checker: queries the latest GitHub
release for this repo and compares its tag against the running version.
Set MSST_FORCE_UPDATE_DIALOG=1 to show the update dialog with fake data
(no remote needed); MSST_FORCE_UPDATE_TAG overrides the version shown.
"""
import os

import requests
from packaging.version import parse as parse_version

from backend.version import APP_VERSION

GITHUB_REPO_OWNER = "bascurtiz"
GITHUB_REPO_NAME = "MSST-GUI"
GITHUB_API_URL = (
    f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
)
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases"

_FORCE_TRUTHY = ("1", "true", "True", "yes", "YES")


def app_version() -> str:
    return APP_VERSION


def force_update_dialog() -> bool:
    return os.environ.get("MSST_FORCE_UPDATE_DIALOG", "").strip() in _FORCE_TRUTHY


def force_update_tag() -> str:
    return os.environ.get("MSST_FORCE_UPDATE_TAG", "").strip() or "v99.0.0"


def get_latest_release_info():
    """Fetch the latest release information from the GitHub API."""
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(GITHUB_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        print(f"[Update Check] Network or API error: {exc}")
        return None
    except Exception as exc:
        print(f"[Update Check] Unexpected error fetching release info: {exc}")
        return None


def compare_versions(current_version_str, latest_version_str):
    """Return True when the latest release tag is newer than the current version."""
    try:
        current_version_str = current_version_str.lstrip("v").lstrip(".")
        latest_version_str = latest_version_str.lstrip("v").lstrip(".")
        return parse_version(latest_version_str) > parse_version(current_version_str)
    except Exception as exc:
        print(
            f"[Update Check] Error comparing versions "
            f"('{current_version_str}' vs '{latest_version_str}'): {exc}"
        )
        return False


def check_for_update():
    """Full check: returns (newer_available, latest_tag_or_None)."""
    if force_update_dialog():
        tag = force_update_tag()
        return compare_versions(APP_VERSION, tag), tag
    info = get_latest_release_info()
    if not info or "tag_name" not in info:
        return False, None
    tag = info["tag_name"]
    return compare_versions(APP_VERSION, tag), tag
