"""Utility functions common to many plotting scripts."""
import numpy as np


def format_description(arr: np.ndarray, dp: float = 0, compact: bool = False) -> str:
    """Format a statistical description of an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    dp : float, optional
        Decimal places to round to, by default 0.
    compact : bool, optional
        Whether to return a compact string (i.e. on one line), by default False.

    Returns
    -------
    str
        Formatted string containing min, mean, max, and standard deviation.
    """
    min = arr.min()
    mean = np.mean(arr)
    max = arr.max()
    std = np.std(arr)
    min_label = f'min = {min:.{dp}f}'
    mean_label = f'μ = {mean:.{dp}f}'
    max_label = f'max = {max:.{dp}f}'
    std_label = f'σ = {std:.{dp}f}'
    if compact:
        return f"{min_label} / {mean_label} / {std_label} / {max_label}"
    return f"{min_label}\n{mean_label} ({std_label})\n{max_label}"
