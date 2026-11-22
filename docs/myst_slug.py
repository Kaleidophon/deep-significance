"""
Heading anchors for the documentation (used by MyST via ``myst_heading_slug_func`` in conf.py).
"""

import re

import emoji


def github_slug(title: str) -> str:
    """
    Create GitHub-style heading anchors. Emojis are turned back into their shortcode names first, so that e.g.
    "⁉️ Why?" becomes "interrobang-why", which is what the links in the README point to.
    """
    title = emoji.demojize(title, language="alias").strip().lower()
    title = re.sub(r"[^\w\- ]", "", title)

    return title.replace(" ", "-")
