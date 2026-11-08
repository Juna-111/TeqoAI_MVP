"""Shared formatting helpers for model and research pages."""

MODEL_LABELS = {
    "first_seen_destination_baseline": "First-seen destination baseline",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Optional autoencoder",
}


def format_metric(value, *, percent=False):
    """Format a metric only when an actual numeric value is available."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1%}" if percent else f"{number:.3f}"