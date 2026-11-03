# Cyclone Atlas Design System Master Specification

> **Project:** Cyclone Atlas — Global Tropical Cyclone Intelligence  
> **Classification:** Scientific Meteorological Analytics Dashboard  
> **Design Dials:** Density 8/10 (Dense / Dashboard) | Variance 6/10 (Balanced / Modern) | Motion 3/10 (Subtle)  
> **Visual Identity:** Dark Ocean Scientific Palette — Deep Navy, Graphite, Ocean Cyan, Deep Teal, and Focused Amber.  
> **Source of Truth:** All dashboard components, Plotly themes, and Streamlit containers must adhere to the tokens defined in this document.

---

## 1. Global Visual Direction & Brand Tokens

The interface embodies a rigorous, academic-grade meteorological research platform. It avoids decorative clutter, emojis, and ungrounded visual metaphors. High information density is balanced with generous whitespace, structured data grids, and strict typographic hierarchy.

### 1.1 Color Tokens

| Role | Token Name | Hex Code | Purpose & Usage |
| :--- | :--- | :--- | :--- |
| **Deep Ocean Background** | `--color-bg-base` | `#080E1E` | Root dashboard backdrop; infinite canvas |
| **Navy Surface Base** | `--color-bg-surface` | `#0F1A30` | Navigation sidebar, primary panels, headers |
| **Card Surface Elevated** | `--color-bg-card` | `#162444` | Metric cards, interactive control panels, tables |
| **Subtle Hover Surface** | `--color-bg-hover` | `#1D2F57` | Hover states, selected tabs, table row focus |
| **Structural Border** | `--color-border-subtle` | `#23355D` | Container dividers, card perimeters, input borders |
| **Accent Border Focused** | `--color-border-active` | `#38BDF8` | Active input focus rings, active tab borders |
| **Text Primary (Foreground)** | `--color-text-primary` | `#F1F5F9` | Major headings, critical KPIs, table cell values |
| **Text Secondary (Muted)** | `--color-text-muted` | `#94A3B8` | Subtitles, axis titles, unit indicators, footnotes |
| **Text Dim** | `--color-text-dim` | `#64748B` | Timestamps, metadata keys, inactive tabs |
| **Primary Accent (Cyan)** | `--color-accent-cyan` | `#00D2D3` | Primary action states, observed track line, +6h forecast |
| **Secondary Accent (Teal)**| `--color-accent-teal` | `#10B981` | Safe/weakening states, historical baselines, +12h forecast |
| **Warning / Attention (Amber)**| `--color-accent-amber` | `#F59E0B` | Critical alerts, rapid intensification, +24h forecast |
| **Destructive / High Alert** | `--color-accent-red` | `#EF4444` | Category 5 intensity, extreme recurvature anomalies |

---

## 2. Typography & Hierarchical Scale

- **Heading Font:** `Fira Code`, `JetBrains Mono`, `Consolas`, monospace (technical, scientific, precise).
- **Body Font:** `Fira Sans`, `Inter`, `-apple-system`, sans-serif (legible, high contrast, compact).
- **Google Fonts Import:**
  ```css
  @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
  ```

### Typographic Scale

| Level | Size (px / rem) | Weight | Line Height | Tracking | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display Header** | `26px / 1.625rem` | `700` | `1.2` | `-0.02em` | App Header Title |
| **Section Header (H2)** | `20px / 1.25rem` | `600` | `1.3` | `-0.01em` | Page Titles, Major Views |
| **Card Header (H3)** | `15px / 0.9375rem`| `600` | `1.4` | `0` | Widget Titles, Metric Cards |
| **Body Standard** | `13.5px / 0.84rem` | `400` | `1.5` | `0` | Explanatory text, table cells |
| **Body Small / Meta** | `11.5px / 0.72rem` | `400` | `1.4` | `+0.02em` | Axis labels, timestamps, badges |
| **Monospace Data / KPI** | `18px / 1.125rem`| `600` | `1.1` | `0` | Wind speeds, coordinates, errors |

---

## 3. Spacing, Borders & Shadows

High-density dashboard specification (8/10):

| Token | Dimension | Application |
| :--- | :--- | :--- |
| `--space-2xs` | `2px` | Border widths, badge inner padding |
| `--space-xs` | `4px` | Button icon gaps, indicator margins |
| `--space-sm` | `8px` | Card internal element spacing, form field gap |
| `--space-md` | `12px` | Metric card padding, compact table row height |
| `--space-lg` | `16px` | Grid column gutters, container margins |
| `--space-xl` | `24px` | Major section vertical separation |

- **Border Radius:**
  - Cards, Containers: `6px` (sharp, architectural precision; avoids bubbly rounded aesthetics)
  - Buttons, Select Boxes: `4px`
  - Badges, Pills: `3px`
- **Box Shadows:**
  - Standard Card: `0 2px 8px rgba(0, 0, 0, 0.4)`
  - Active Dropdown / Modal: `0 8px 24px rgba(0, 0, 0, 0.6)`

---

## 4. Plotly Data Visualization Standards

All interactive Plotly charts and geospatial maps must inherit the centralized dark scientific theme:

- **Chart Background:** `#0F1A30` (transparent to container)
- **Plot Area Background:** `#080E1E`
- **Grid Lines:** `rgba(255, 255, 255, 0.07)` (faint, non-distracting)
- **Zero Lines:** `rgba(255, 255, 255, 0.15)`
- **Axis Text:** `#94A3B8` (Fira Sans, 11px)
- **Map Projections:** `natural earth` or `equirectangular` with custom dark ocean styling:
  - Ocean Base: `#080E1E`
  - Landmass Fill: `#1A2846`
  - Coastline & Borders: `#2D3E66`
- **Horizon Color Sequence:**
  - Observed / Past: `#94A3B8` (solid slate line with hollow circles)
  - Current Position ($t=0$): `#FFFFFF` (solid glowing point)
  - +6h Forecast: `#00D2D3` (cyan)
  - +12h Forecast: `#10B981` (emerald teal)
  - +24h Forecast: `#F59E0B` (amber)

---

## 5. Lucide-Style SVG Icon System

In accordance with strict UI/UX guidelines:
- **No emoji icons.**
- Use clean, inline SVG icons with a standard 16px/18px stroke for sidebar navigation and indicators:
  - Overview: `<svg ...><circle cx="12" cy="12" r="10"/><path d="..."></svg>`
  - Intensity: Barometer / Gauge icon
  - Movement: Navigation / Compass icon
  - Development: TrendingUp / Growth icon
  - Replay: PlayCircle / History icon
  - Performance: Award / CheckCircle icon

---

## 6. Pre-Delivery Quality Checklist

- [x] Dark ocean palette with high text contrast (WCAG AA ratio $\ge 4.5:1$).
- [x] Zero emoji icons in headings, cards, and buttons.
- [x] Persistent academic research disclaimer.
- [x] Monospace typography for scientific coordinates and metrics.
- [x] Plotly maps and charts formatted with dark ocean aesthetic.
- [x] Accessible tables accompanying spatial visualizations.
