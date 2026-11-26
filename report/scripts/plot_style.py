"""Unified publication-grade plotting theme for CPE342 Technical Report (Team ML003)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Colorblind-safe engineering palette matching LaTeX preamble
NAVY = "#1B365D"
SLATE = "#334E68"
TEAL = "#0D9488"
CORAL = "#D95F02"
AMBER = "#D97706"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
GRAY = "#94A3B8"
LIGHT_GRAY = "#E2E8F0"
RED = "#DC2626"

PALETTE = [NAVY, TEAL, CORAL, BLUE, PURPLE, AMBER, GRAY]


def apply_style():
    """Apply consistent high-DPI Matplotlib styling for report figures."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CBD5E1",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": NAVY,
        "axes.labelsize": 9.5,
        "axes.labelcolor": "#1E293B",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CBD5E1",
        "legend.framealpha": 0.92,
    })
