"""Reusable HTML presentation components for the TeqoAI dashboard.

The components return HTML strings so page code can combine them in a single
``st.markdown`` call. Every dynamic value must pass through ``esc`` before it
is interpolated.
"""

import html

ICONS = {
    "detect": (
        '<circle cx="12" cy="12" r="7"/>'
        '<path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>'
        '<circle cx="12" cy="12" r="2.2"/>'
    ),
    "investigate": (
        '<circle cx="11" cy="11" r="6.5"/>'
        '<path d="m15.8 15.8 4.2 4.2"/>'
    ),
    "connect": (
        '<circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/>'
        '<circle cx="12" cy="18" r="2.4"/>'
        '<path d="M8.2 7 11 15.8M15.8 7 13 15.8M8.4 6h7.2"/>'
    ),
    "brain": (
        '<path d="M9.5 3.5a3 3 0 0 0-3 3 2.7 2.7 0 0 0-1.5 4.9'
        'A3 3 0 0 0 6.6 17a3 3 0 0 0 3 3.4c.9 0 1.7-.4 2.2-1V4.5'
        'c-.5-.6-1.3-1-2.3-1Z"/>'
        '<path d="M14.5 3.5a3 3 0 0 1 3 3 2.7 2.7 0 0 1 1.5 4.9'
        'A3 3 0 0 1 17.4 17a3 3 0 0 1-3 3.4c-.9 0-1.7-.4-2.2-1V4.5'
        'c.5-.6 1.3-1 2.3-1Z"/>'
    ),
    "chart": (
        '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'
    ),
    "shield": (
        '<path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.3 7.5 9.5 4.3-1.2 7.5-4.9 7.5-9.5V6Z"/>'
        '<path d="m9 12 2.2 2.2L15.5 10"/>'
    ),
    "layers": (
        '<path d="m12 3 9 5-9 5-9-5Z"/>'
        '<path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5.5" rx="7.5" ry="3"/>'
        '<path d="M4.5 5.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>'
        '<path d="M4.5 11.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>'
    ),
    "lock": (
        '<rect x="4.5" y="10" width="15" height="10.5" rx="2"/>'
        '<path d="M8 10V7.5a4 4 0 0 1 8 0V10M12 14.5v2.5"/>'
    ),
    "user": (
        '<circle cx="12" cy="8" r="3.6"/>'
        '<path d="M4.5 20a7.5 7.5 0 0 1 15 0"/>'
    ),
    "device": (
        '<rect x="3" y="4" width="18" height="12" rx="2"/>'
        '<path d="M8 20h8M12 16v4"/>'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/>'
    ),
    "alert": (
        '<path d="M12 3 2.8 20h18.4Z"/>'
        '<path d="M12 9v4.5M12 17h.01"/>'
    ),
    "check": (
        '<path d="m4.5 12.5 5 5L19.5 7"/>'
    ),
    "arrow": (
        '<path d="M4 12h15M13 6l6 6-6 6"/>'
    ),
    "mail": (
        '<rect x="3" y="5" width="18" height="14" rx="2"/>'
        '<path d="m3 7 9 6 9-6"/>'
    ),
    "telegram": (
        '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>'
    ),
    "github": (
        '<path d="m8 9-3 3 3 3M16 9l3 3-3 3M14 5l-4 14"/>'
    ),
    "phone": (
        '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 '
        '19.4 19.4 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 '
        '0 1 2 1.7l.5 2.8a2 2 0 0 1-.6 1.9L7.7 9.7a16 16 0 0 0 6 6l1.3-1.3a2 '
        '2 0 0 1 1.9-.6l2.8.5a2 2 0 0 1 1.7 2.6Z"/>'
    ),
}


def esc(value):
    """Escape text for safe interpolation into component markup."""
    return html.escape(str(value), quote=True)


def icon(name, size=18):
    path = ICONS.get(name, ICONS["shield"])
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


def badge(label, tone="neutral", dot=False):
    dot_markup = '<span class="dot" aria-hidden="true"></span>' if dot else ""
    return (
        f'<span class="teqo-badge" data-tone="{esc(tone)}">'
        f"{dot_markup}{esc(label)}</span>"
    )


def status_badge(status):
    tones = {
        "New": "info",
        "Investigating": "warning",
        "Escalated": "critical",
        "Resolved": "success",
        "Benign": "neutral",
    }
    return badge(status, tones.get(status, "neutral"))


def environment_badge(dataset):
    if dataset == "synthetic":
        return badge("Synthetic demo", "warning", dot=True)
    return badge("LANL · historical", "info", dot=True)


def page_header(kicker, title, description, dataset, meta_note="Local analysis"):
    return (
        '<header class="teqo-page-head">'
        '<div class="teqo-page-head-copy">'
        f'<div class="teqo-eyebrow">{esc(kicker)}</div>'
        f"<h1>{esc(title)}</h1>"
        f"<p>{esc(description)}</p>"
        "</div>"
        '<div class="teqo-page-head-meta">'
        f"{environment_badge(dataset)}"
        f'<span class="teqo-page-meta-note">{esc(meta_note)}</span>'
        "</div></header>"
    )


def section_header(kicker, title=None, description=None):
    parts = [f'<div class="teqo-section-kicker">{esc(kicker)}</div>']
    if title:
        parts.append(f"<h2>{esc(title)}</h2>")
    if description:
        parts.append(f"<p>{esc(description)}</p>")
    return f'<div class="teqo-section-head">{"".join(parts)}</div>'


def process_diagram(kicker, title, description, steps):
    items = "".join(
        "<li><strong>{}</strong><span>{}</span></li>".format(esc(label), esc(detail))
        for label, detail in steps
    )
    return (
        section_header(kicker, title, description)
        + f'<ol class="teqo-process" aria-label="{esc(title)}">{items}</ol>'
    )


def metric_grid(items):
    cards = []
    for item in items:
        size = item.get("size", "default")
        cards.append(
            f'<div class="teqo-metric" data-tone="{esc(item.get("tone", "default"))}" '
            f'data-size="{esc(size)}">'
            f'<div class="teqo-metric-label">{esc(item["label"])}</div>'
            f'<div class="teqo-metric-value">{esc(item["value"])}</div>'
            f'<div class="teqo-metric-hint">{esc(item.get("hint", ""))}</div>'
            "</div>"
        )
    return f'<div class="teqo-metrics">{"".join(cards)}</div>'


def callout(title, body, tone="info", allow_html=False):
    body_markup = body if allow_html else esc(body)
    return (
        f'<div class="teqo-callout" data-tone="{esc(tone)}">'
        f'<div class="teqo-callout-title">{esc(title)}</div>'
        f"<p>{body_markup}</p></div>"
    )


def feature_grid(features):
    cards = []
    for feature in features:
        cards.append(
            '<article class="teqo-feature">'
            f'<div class="teqo-feature-icon">{icon(feature.get("icon", "shield"))}</div>'
            f'<h3>{esc(feature["title"])}</h3>'
            f'<p>{esc(feature["body"])}</p>'
            "</article>"
        )
    return f'<div class="teqo-feature-grid">{"".join(cards)}</div>'


def contribution_grid(items):
    cards = []
    for title, description, source in items:
        cards.append(
            '<article class="teqo-contribution">'
            f"<strong>{esc(title)}</strong>"
            f"<p>{esc(description)}</p>"
            f"<small>{esc(source)}</small></article>"
        )
    return f'<div class="teqo-contribution-grid">{"".join(cards)}</div>'


def kv_rows(items):
    rows = "".join(
        f'<div class="teqo-kv-row"><span>{esc(label)}</span>'
        f"<strong>{esc(value)}</strong></div>"
        for label, value in items
    )
    return f'<div class="teqo-kv-list">{rows}</div>'


def kv_card(items):
    rows = "".join(
        f'<div class="teqo-kv"><dt>{esc(label)}</dt><dd>{value}</dd></div>'
        for label, value in items
    )
    return f'<dl class="teqo-kv-grid">{rows}</dl>'


def factor_list(factors):
    rows = []
    for label, active, detail in factors:
        state = "Triggered" if active else "Not triggered"
        mark = "!" if active else "·"
        detail_markup = f" — {esc(detail)}" if detail else ""
        rows.append(
            f'<div class="teqo-factor" data-active="{"true" if active else "false"}">'
            f'<span class="teqo-factor-icon" aria-hidden="true">{mark}</span>'
            f"<span>{esc(label)}{detail_markup}</span>"
            f'<span class="teqo-factor-state">{state}</span></div>'
        )
    return f'<div class="teqo-factor-grid">{"".join(rows)}</div>'


def risk_distribution(counts, total, note=None):
    levels = (
        ("High", "Priority review", "critical"),
        ("Medium", "Review", "warning"),
        ("Low", "Routine", "success"),
    )
    rows = []
    for level, label, tone in levels:
        count = int(counts.get(level, 0))
        percent = count / max(int(total), 1) * 100
        rows.append(
            '<div class="teqo-risk-row">'
            f'<span class="teqo-risk-label"><i class="teqo-dot" '
            f'data-tone="{tone}" aria-hidden="true"></i>{esc(label)}</span>'
            f'<div class="teqo-risk-track" role="img" '
            f'aria-label="{esc(label)}: {count} alerts, {percent:.1f} percent">'
            f'<div class="teqo-risk-fill" style="width:{percent:.1f}%;'
            f'background:var(--teqo-{tone})"></div></div>'
            f"<strong>{count:,}</strong></div>"
        )
    note_markup = f'<p class="teqo-panel-note">{esc(note)}</p>' if note else ""
    return (
        '<div class="teqo-panel"><div class="teqo-panel-head">'
        '<h3 class="teqo-panel-title">Detection severity</h3>'
        f'<span class="teqo-panel-meta">{int(total):,} analyzed</span></div>'
        f'{"".join(rows)}{note_markup}</div>'
    )


def legend(items):
    parts = []
    for label, tone, line in items:
        swatch = '<i class="line" aria-hidden="true"></i>' if line else (
            f'<i data-tone="{esc(tone)}" aria-hidden="true"></i>'
        )
        parts.append(f"<span>{swatch}{esc(label)}</span>")
    return f'<div class="teqo-legend">{"".join(parts)}</div>'


def empty_state(title, body):
    return (
        f'<div class="teqo-empty"><strong>{esc(title)}</strong>'
        f"<p>{esc(body)}</p></div>"
    )


def provenance_block(items):
    cells = "".join(
        f"<div><span>{esc(label)}</span><code>{esc(value)}</code></div>"
        for label, value in items
    )
    return f'<div class="teqo-provenance">{cells}</div>'


def contact_list(contacts):
    links = []
    for contact in contacts:
        links.append(
            f'<a href="{esc(contact["href"])}"'
            + (' target="_blank" rel="noopener noreferrer"' if contact.get("external") else "")
            + f'>{icon(contact["icon"], 15)}{esc(contact["label"])}</a>'
        )
    return f'<div class="teqo-contact-list">{"".join(links)}</div>'


def limitations(items):
    rows = "".join(f"<li>{esc(item)}</li>" for item in items)
    return f'<ul class="teqo-limitations">{rows}</ul>'
