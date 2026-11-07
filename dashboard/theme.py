"""Load the TeqoAI visual design system."""

import os

import streamlit as st

STYLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles")
STYLE_FILES = ("theme.css", "layout.css", "components.css", "responsive.css")


def _read_styles():
    blocks = []
    for filename in STYLE_FILES:
        path = os.path.join(STYLES_DIR, filename)
        with open(path, encoding="utf-8") as style_file:
            blocks.append(f"/* {filename} */\n{style_file.read()}")
    return "\n".join(blocks)


def load_styles():
    """Inject the centralized design tokens and component styles."""
    st.markdown(f"<style>{_read_styles()}</style>", unsafe_allow_html=True)
