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

