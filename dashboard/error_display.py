"""Consistent absolute and relative error labels for forecast results."""
import math


def format_error(error: float, reference: float, unit: str) -> str:
    """Show percentage first; zero or missing reference has no defined percentage."""
    percentage = (
        f"{100 * abs(error) / abs(reference):.1f}%"
        if math.isfinite(error) and math.isfinite(reference) and reference != 0
        else "N/A%"
    )
    return f"{percentage} ({abs(error):.1f} {unit})"
