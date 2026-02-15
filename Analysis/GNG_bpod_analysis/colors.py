# ─── Global Color Palette ─────────────────────────────────────────────
COLOR_GO       = "#2c6e49"   # Dark Green
COLOR_NOGO     = "#A50026"   # Dark Red
COLOR_GRAY     = "#D3D3D3"   # Light Gray
COLOR_VERY_SUBTLE = "#cbcdd4"   # Light Gray
COLOR_BLUE     = "#104E8B "   # Blue
COLOR_BLUE_TRANSPARENT = "rgba(16, 78, 139, 0.2)"
COLOR_ACCENT   = '#1E90FA'   # Dodger Blue
COLOR_ACCENT_TRANSPARENT = "rgba(30, 144, 250, 0.2)"
COLOR_SUBTLE   = "#9699A7"   # Muted Gray-Blue

COLOR_LOW_BD   = '#ff4800'
COLOR_LOW_BD_TRANSPARENT = "rgba(255, 72, 0, 0.2)"
COLOR_HIGH_BD  = '#ffb600'
COLOR_HIGH_BD_TRANSPARENT = "rgba(255, 182, 0, 0.2)"

COLOR_D_PRIME = "#FF7F0E"  # Orange
COLOR_ORANGE = "#FF7F0E"   # Orange (general purpose)
# ─── Outcome Colors ─────────────────────────────────────────────
COLOR_HIT = "#008000"      # Green
COLOR_CR = "#1E90FA"       # Dodger Blue
COLOR_FA = "#FF9100"       # Orange
COLOR_MISS = "#B22222"     # Firebrick Red

OUTCOME_COLOR_MAP = {
    "Hit": COLOR_HIT,
    "CR": COLOR_CR,
    "FA": COLOR_FA,
    "False Alarm": COLOR_FA,
    "Miss": COLOR_MISS,
}

# ─── Setup Colors (Rig, Educage, etc.) ─────────────────────────
# Distinct from GROUP_COLORS to avoid confusion when comparing by Setup vs Group
COLOR_RIG = "#00838F"       # Dark Cyan/Teal - distinct from standard blues
COLOR_EDUCAGE = "#AD1457"   # Deep Magenta/Rose - distinct from standard orange/red

SETUP_COLOR_MAP = {
    "Rig": COLOR_RIG,
    "Bpod": COLOR_RIG,      # Legacy mapping
    "Educage": COLOR_EDUCAGE,
}

# Fallback colors for additional setups
SETUP_FALLBACK_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

def get_setup_color(setup_name: str, index: int = 0) -> str:
    """Get color for a setup, with fallback for unknown setups."""
    if setup_name in SETUP_COLOR_MAP:
        return SETUP_COLOR_MAP[setup_name]
    return SETUP_FALLBACK_COLORS[index % len(SETUP_FALLBACK_COLORS)]


# Fallback colors for groups (more distinct colors for many groups)
GROUP_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Teal
    "#393b79",  # Dark Indigo
    "#637939",  # Olive
]

def get_group_color(group_name: str, index: int = 0) -> str:
    """Get color for a group by index."""
    return GROUP_COLORS[index % len(GROUP_COLORS)]

# ─── Go/NoGo Stimulus Color Palettes ─────────────────────────────
# Green shades for Go stimuli
GO_COLORS = [
    "#2E8B57",  # Sea Green
    "#3CB371",  # Medium Sea Green
    "#20B2AA",  # Light Sea Green
    "#48D1CC",  # Medium Turquoise
    "#40E0D0",  # Turquoise
    "#7FFFD4",  # Aquamarine
    "#66CDAA",  # Medium Aquamarine
    "#98FB98",  # Pale Green
    "#90EE90",  # Light Green
    "#ADFF2F",  # Green Yellow
]

# Red shades for NoGo stimuli
NOGO_COLORS = [
    "#DC143C",  # Crimson
    "#B22222",  # Fire Brick
    "#CD5C5C",  # Indian Red
    "#F08080",  # Light Coral
    "#FA8072",  # Salmon
    "#E9967A",  # Dark Salmon
    "#FFA07A",  # Light Salmon
    "#FF6347",  # Tomato
    "#FF4500",  # Orange Red
    "#FF0000",  # Red
]

# ─── Subject Color Palette (Qualitative, Many Distinct Hues) ────────
# Use to map subjects deterministically by index: color = SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]
SUBJECT_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Teal
    "#393b79",  # Dark Indigo
    "#637939",  # Olive
    "#8c6d31",  # Ochre
    "#843c39",  # Maroon
    "#7b4173",  # Plum
    "#3182bd",  # Steel Blue
    "#e6550d",  # Burnt Orange
    "#31a354",  # Emerald
    "#756bb1",  # Lavender Purple
    "#636363",  # Mid Gray
    "#bdbdbd",  # Light Gray
    "#9ecae1",  # Light Blue
    "#fd8d3c",  # Soft Orange
    "#74c476",  # Soft Green
    "#c994c7",  # Soft Purple
    "#a1d99b",  # Pale Green
    "#6baed6",  # Sky Blue
    "#d6616b",  # Rose
    "#ce6dbd",  # Magenta
    "#e7ba52",  # Mustard
    "#ad494a",  # Brick
    "#a55194",  # Violet
]

# Map unique subject identifiers to distinct colors from SUBJECT_COLORS
def get_subject_color_map(subject_names):
    """
    Deterministically map each unique subject name to a distinct color.

    Args:
        subject_names (Iterable[str]): A sequence of subject identifiers (e.g., MouseName).

    Returns:
        dict[str, str]: Mapping of subject name -> color hex.
    """
    try:
        # Preserve first-seen order while ensuring uniqueness
        seen = set()
        unique_subjects = []
        for name in subject_names:
            key = str(name)
            if key not in seen:
                seen.add(key)
                unique_subjects.append(key)

        color_map = {
            subject: SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]
            for idx, subject in enumerate(unique_subjects)
        }
        return color_map
    except Exception:
        # Fallback: return empty mapping on any unexpected error
        return {}

# ─── Line Width ─────────────────────────────────────────────────
LINE_WIDTH_THIN     = 0.5
LINE_WIDTH_MEDIUM  = 1.0
LINE_WIDTH_THICK    = 5
LINE_WIDTH_VERY_THICK = 7

# ─── Global Font Sizes  ────────────────────────────────────
# Use these in all figures for consistent typography.
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
TICK_FONT_SIZE = 16


def apply_standard_font_sizes(fig, transparent_bg=True):
    """
    Apply global font sizes and styling to a Plotly figure:
    - Title
    - Axis labels
    - Legend
    - Tick labels
    - Transparent background (for SVG export)
    """
    try:
        # Base font for general text (axis labels, annotations without explicit font, etc.)
        layout_opts = dict(
            font=dict(size=LABEL_FONT_SIZE),
            title=dict(font=dict(size=TITLE_FONT_SIZE)),
            legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
        )
        # Apply transparent background for clean SVG exports
        if transparent_bg:
            layout_opts['paper_bgcolor'] = 'rgba(0,0,0,0)'
            layout_opts['plot_bgcolor'] = 'rgba(0,0,0,0)'
        
        fig.update_layout(**layout_opts)
        # Axes: labels and ticks
        fig.update_xaxes(
            title_font=dict(size=LABEL_FONT_SIZE),
            tickfont=dict(size=TICK_FONT_SIZE),
        )
        fig.update_yaxes(
            title_font=dict(size=LABEL_FONT_SIZE),
            tickfont=dict(size=TICK_FONT_SIZE),
        )
    except Exception:
        # Silently ignore if figure doesn't support some properties
        pass
    return fig


def marker_symbols_from_boundaries(boundaries):
    """
    Return a per-point marker symbol list based on number of boundaries.
    circle for 1 boundary, square for 2; defaults to circle on error.
    """
    try:
        return ["circle" if int(nb) == 1 else "square" for nb in list(boundaries)]
    except Exception:
        try:
            # Single scalar
            nb = int(boundaries)
            return ["circle" if nb == 1 else "square"]
        except Exception:
            return ["circle"]


def marker_sizes_from_tones(tones_per_class, scale=5.0, default_size=6.0):
    """
    Return a per-point marker size list scaled by tones_per_class.
    """
    import numpy as np
    try:
        tones = np.asarray(list(tones_per_class), dtype=float)
        sizes = tones * float(scale)
        sizes[~np.isfinite(sizes)] = float(default_size)
        return sizes.tolist()
    except Exception:
        return [float(default_size)]


def add_marker_legends(fig, boundaries, tones_per_class, scale=5.0):
    """
    Add legend entries to a Plotly figure for marker shapes (boundaries)
    and representative sizes (tones per class).
    """
    import plotly.graph_objects as go
    import pandas as pd
    # Shapes by boundaries
    try:
        unique_bounds = sorted(set(int(nb) for nb in boundaries if pd.notna(nb)))
    except Exception:
        unique_bounds = [1, 2]
    for nb in unique_bounds:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=('circle' if nb == 1 else 'square'), color='gray', size=10),
            name=("1 Boundary" if nb == 1 else "2 Boundaries"),
            showlegend=True, hoverinfo='skip'
        ))
    # Sizes by tones
    try:
        size_levels = sorted(set(int(t) for t in tones_per_class if pd.notna(t)))
    except Exception:
        size_levels = []
    for size in size_levels:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='circle', color='white', size=float(size)*float(scale), line=dict(width=3, color='gray')),
            name=f"{size} Tones",
            showlegend=True, hoverinfo='skip'
        ))