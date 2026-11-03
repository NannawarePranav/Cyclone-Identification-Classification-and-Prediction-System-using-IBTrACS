"""
Cyclone Atlas — Configuration, Theme Tokens, and Styling Definitions
=====================================================================
Encapsulates global visual styling, color tokens, Plotly templates, Lucide SVG
icons, and domain mappings according to design-system/cyclone-atlas/MASTER.md.
"""

import streamlit as st

# Application Branding
APP_NAME = "CYCLONE ATLAS"
APP_SUBTITLE = "Global Tropical Cyclone Intelligence"
APP_DISCLAIMER = (
    "Research prototype using historical NOAA IBTrACS data. "
    "Predictions are experimental and are not official weather warnings."
)

# Dark Ocean Scientific Color Tokens
COLORS = {
    'bg_base': '#080E1E',          # Main application background
    'bg_surface': '#0F1A30',       # Sidebar & primary containers
    'bg_card': '#162444',          # Elevated data cards and tables
    'bg_hover': '#1D2F57',         # Hover/active states
    'border': '#23355D',           # Subtle structural borders
    'border_dark': '#23355D',      # Subtle structural borders alias
    'border_active': '#38BDF8',    # Focus rings & active borders
    'text_primary': '#F1F5F9',     # Headers and high-contrast values
    'text_bright': '#F1F5F9',      # Alias for headers and bright text
    'text_muted': '#94A3B8',       # Subtitles, units, axis titles
    'text_secondary': '#94A3B8',   # Alias for secondary body text
    'text_dim': '#64748B',         # Footnotes, timestamps, inactive
    'cyan': '#00D2D3',             # Primary accent (+6h forecast, observed track)
    'accent_cyan': '#00D2D3',      # Primary accent alias
    'teal': '#10B981',             # Secondary accent (+12h forecast, weakening)
    'accent_teal': '#10B981',      # Secondary accent alias
    'amber': '#F59E0B',            # Warning accent (+24h forecast, strengthening)
    'accent_amber': '#F59E0B',     # Warning accent alias
    'red': '#EF4444',              # High alert / severe intensity
    'accent_rose': '#EF4444',      # Rose / red alert alias
    'purple': '#8B5CF6',           # Comparative reference lines
    'accent_purple': '#8B5CF6',    # Purple accent alias
    'grid_line': 'rgba(35, 53, 93, 0.45)' # Chart grid lines
}

# Ocean Basin Mappings (NOAA IBTrACS Standard)
BASIN_NAMES = {
    'EP': 'Eastern North Pacific',
    'NA': 'North Atlantic',
    'NI': 'North Indian',
    'SI': 'South Indian',
    'SP': 'South Pacific',
    'WP': 'Western North Pacific',
    'SA': 'South Atlantic'
}

# Storm Nature Mappings
NATURE_NAMES = {
    'TS': 'Tropical Storm / Cyclone',
    'SS': 'Subtropical Cyclone',
    'ET': 'Extratropical Cyclone',
    'MX': 'Mixed / Transitioning System',
    'NR': 'Not Reported / Disturbance',
    'DS': 'Disturbance / Wave'
}

class IconRegistry(dict):
    """Dictionary that returns a default fallback SVG instead of raising KeyError."""
    def __missing__(self, key: str) -> str:
        return self.get('info', '')


# Lucide-Style SVG Icons (18px stroke, dark ocean theme)
ICONS = IconRegistry({
    'globe': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>''',
    'overview': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>''',
    'wind': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>''',
    'intensity': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>''',
    'navigation': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>''',
    'movement': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>''',
    'compass': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>''',
    'activity': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>''',
    'development': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>''',
    'play': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>''',
    'replay': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>''',
    'bar_chart': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>''',
    'performance': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>''',
    'calendar': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>''',
    'clock': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>''',
    'cpu': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>''',
    'shield_alert': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>''',
    'alert_circle': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>''',
    'alert': '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>''',
    'info': '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'''
})


def apply_custom_css():
    """Injects comprehensive dark ocean scientific CSS into the Streamlit app."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Base Reset */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg_base']};
        color: {COLORS['text_primary']};
        font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_surface']} !important;
        border-right: 1px solid {COLORS['border']};
    }}

    /* Compact Top Header */
    [data-testid="stHeader"] {{
        background-color: {COLORS['bg_base']};
        border-bottom: 1px solid {COLORS['border']};
    }}

    /* Main Container Spacing */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        font-family: 'Fira Code', monospace;
        color: {COLORS['text_primary']};
        font-weight: 600;
        letter-spacing: -0.02em;
    }}

    /* Scientific Metric Cards */
    .metric-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {COLORS['border_active']};
    }}
    .metric-title {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS['text_muted']};
        margin-bottom: 4px;
        font-weight: 500;
    }}
    .metric-value {{
        font-family: 'Fira Code', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: {COLORS['text_primary']};
        line-height: 1.2;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: {COLORS['text_dim']};
        margin-top: 4px;
    }}
    .metric-cyan {{ color: {COLORS['cyan']}; }}
    .metric-teal {{ color: {COLORS['teal']}; }}
    .metric-amber {{ color: {COLORS['amber']}; }}
    .metric-red {{ color: {COLORS['red']}; }}

    /* Scientific Panel / Container */
    .scientific-panel {{
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }}

    /* Persistent Disclaimer Banner */
    .disclaimer-banner {{
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.75rem;
        color: {COLORS['amber']};
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 16px;
    }}

    /* Streamlit Selectboxes and Inputs */
    div[data-baseweb="select"] > div {{
        background-color: {COLORS['bg_surface']} !important;
        border-color: {COLORS['border']} !important;
        color: {COLORS['text_primary']} !important;
        border-radius: 4px !important;
    }}
    div[data-baseweb="select"] > div:hover {{
        border-color: {COLORS['border_active']} !important;
    }}

    /* Button Styling */
    .stButton > button {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['cyan']} !important;
        border: 1px solid {COLORS['cyan']} !important;
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['cyan']};
        color: {COLORS['bg_base']} !important;
        border-color: {COLORS['cyan']} !important;
    }}

    /* Dataframe Table Customization */
    div[data-testid="stDataFrame"] {{
        background-color: {COLORS['bg_card']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px;
    }}

    /* Custom Radio Navigation in Sidebar */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 4px;
        border: 1px solid transparent;
        transition: all 0.15s ease;
    }}
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background-color: {COLORS['bg_card']};
        border-color: {COLORS['border']};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
