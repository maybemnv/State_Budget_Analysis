# DataLens AI — Visual Design Specification

## Color Palette

```
┌─────────────────────────────────────────────────────────────────┐
│  BACKGROUNDS — Warm, not cold                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │          │  │          │  │          │                       │
│  │  #0A0A0F │  │  #14141A │  │  #1E1E28 │                       │
│  │  Warm    │  │  Surface │  │  Elevated│                       │
│  │  Black   │  │          │  │          │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ACCENTS — Energy, not corporate                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │          │  │          │  │          │                       │
│  │  #FF6B35 │  │  #00DCB4 │  │  #9D4EDD │                       │
│  │  Burnt   │  │  Teal    │  │  Deep    │                       │
│  │  Orange  │  │  (success)│  │  Purple  │                       │
│  │  (primary)│ │          │  │  (agent) │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TEXT — Warm off-white, easy on eyes                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │          │  │          │  │          │                       │
│  │  #E8E6E3 │  │  #C8C4BC │  │  #8B8878 │                       │
│  │  Primary │  │  Secondary│  │  Muted   │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Typography

```
HEADINGS — Satoshi (geometric, confident)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DataLens AI
├─ 2xl: 1.5rem / 24px — Page titles
├─ xl:  1.25rem / 20px — Section titles
└─ lg:  1.125rem / 18px — Emphasis

BODY — Geist Mono (terminal aesthetic, data density)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The agent is thinking about your query...
├─ base: 1rem / 16px — Default body
├─ sm:  0.875rem / 14px — Stats, labels
└─ xs:  0.75rem / 12px — Badges, captions

CODE — JetBrains Mono
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{ "session_id": "abc-123", "tool": "run_pca" }
└─ Used for specs, tool calls, data values
```

---

## Workspace Layout — Annotated

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  DataLens AI                                              [#] [─] [×]                   │
├──────────────┬──────────────────────────────────────────┬───────────────────────────────┤
│              │                                          │                               │
│  📊 budget   │  ◉ Agent avatar                         │  VISUALIZATIONS               │
│  2020-2024   │  (pulsing purple orb)                   │                               │
│              │                                          │  ┌─────────────────────────┐  │
│  8 columns   │  User: What's driving revenue variance? │  │                         │  │
│  ├─ 4 num    │                                          │  │     ●                   │  │
│  └─ 4 cat    │  ┌────────────────────────────────────┐  │  │    ● ●                  │  │
│              │  │ Thought                            │  │  │   ●   ●                 │  │
│  Columns     │  │ I need to check which factors      │  │  │  ●  ●  ●   PCA 3D      │  │
│  ├─ Revenue  │  │ correlate with revenue variance... │  │  │   ● ●                   │  │
│  ├─ Year     │  └────────────────────────────────────┘  │  │    ● ●                  │  │
│  ├─ Category │                                          │  │     ●                   │  │
│  ├─ Region   │  ┌────────────────────────────────────┐  │  │                         │  │
│  └─ ...      │  │ 🛠️ run_correlation                │  │  │  Depth-of-field blur    │  │
│              │  │ { columns: ["Revenue", "Category"] │  │  │  on distant points      │  │
│  Quick Stats │  └────────────────────────────────────┘  │  │  Glow on hover          │  │
│  Mean: 1.2M  │                                          │  └─────────────────────────┘  │
│  Std: 340K   │  ┌────────────────────────────────────┐  │                               │
│  Skew: 1.8   │  │ Result                             │  │  ┌─────────────────────────┐  │
│              │  │ Category C has 3.2x higher         │  │  │  ● Cluster 1 (n=234)    │  │
│  📈 Trend    │  │ variance than Category A.          │  │  │  ● Cluster 2 (n=189)    │  │
│  ↗ +12% YoY  │  │ R² = 0.67                          │  │  │  ● Cluster 3 (n=156)    │  │
│              │  └────────────────────────────────────┘  │  │                         │  │
│  ─────────── │                                          │  │  Cluster Orbs           │  │
│              │  ╔════════════════════════════════════╗  │  │  Refractive spheres     │  │
│  SESSION     │  ║ Final Answer                       ║  │  │  Density visible        │  │
│  HISTORY     │  ║ Category C is driving variance.    ║  │  │  through layers         │  │
│  ├─ Q4 2024  │  ║ 3.2x higher volatility, 60% of     ║  │  └─────────────────────────┘  │
│  ├─ Q3 2024  │  ║ costs. Recommend deep dive into    ║  │                               │
│  └─ Q2 2024  │  ║ cost structure.                    ║  │  [Fullscreen] [Export PNG]  │
│              │  ╚════════════════════════════════════╝  │  [SVG] [Chart History ▼]    │
│  [Settings]  │                                          │                               │
│  [LLM: ▼]    │  ─────────────────────────────────────── │                               │
│              │  [ Type your question...          ] [→]  │                               │
│              │  [ Cmd+K for suggestions            ]    │                               │
│              │                                          │                               │
├──────────────┴──────────────────────────────────────────┴───────────────────────────────┤
│  AGENT TIMELINE — Click to jump to any reasoning step                                   │
│  ●──────────○──────────●────────────○────────●                                          │
│  describe   stats      correlation   chart   answer                                     │
│  10:23:41   10:23:43   10:23:45      10:23:47  10:23:49                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Avatar States

```
THINKING          EXECUTING         DONE              ERROR
◉                 ◉                 ◉                 ◉
Purple glow       Orange glow       Teal steady       Red flicker
Slow pulse        Fast pulse        No pulse          Erratic pulse
~1Hz              ~3Hz              —                 —



┌─────────────────────────────────────────────────────────────────┐
│  Animation: Pulse                                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                 │
│  Thinking:  opacity 0.6 → 1.0 → 0.6  (1s ease-in-out, loop)    │
│  Executing: opacity 0.4 → 1.0 → 0.4  (0.3s ease-in-out, loop)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Micro-Interaction: Typewriter Effect

```
User sees agent thoughts appear character-by-character:

Frame 1:  I need to check...
Frame 2:  I need to check for...
Frame 3:  I need to check for time...
Frame 4:  I need to check for time series...
Frame 5:  I need to check for time series patterns...

Speed: ~60 WPM (words per minute)
       ~10 characters per second
       Cursor blinks at 1Hz

Sound (optional): Mechanical keyboard click per character
                  (muted by default)
```

---

## Micro-Interaction: Tool Card Build

```
Tool calls don't just appear — they construct themselves:

Frame 1:  ┌────────────────────────┐
          │ 🛠️ run_...             │
          └────────────────────────┘

Frame 2:  ┌────────────────────────┐
          │ 🛠️ run_correlation     │
          │ { columns: [...]       │
          └────────────────────────┘

Frame 3:  ┌────────────────────────┐
          │ 🛠️ run_correlation     │
          │ { columns: ["Revenue", │
          │   "Category"] }        │
          └────────────────────────┘

Frame 4:  ┌────────────────────────┐
          │ 🛠️ run_correlation     │
          │ { columns: ["Revenue", │
          │   "Category"] }        │
          │                        │
          │ Result: R² = 0.67      │
          └────────────────────────┘
          ↑ "snaps" in with scale(0.95) → scale(1.0)
```

---

## 3D Visualization: PCA Scatter

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                        ●                                        │
│                   ●         ●                                   │
│              ●                 ●                                │
│         ●    [●]  ●    ●         ●                              │
│              ●                 ●                                │     ← Depth-of-field:
│                   ●    ●                                         │        distant points blur
│                        ●                                        │
│                                                                 │
│  Hover effect:                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Point #234                                              │   │
│  │ PC1: 2.34  PC2: -1.23  PC3: 0.89                        │   │
│  │ Revenue: $1.2M  Category: C                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│  + Bloom glow around hovered point                             │
│                                                                 │
│  Controls: Left-click drag to rotate, right-click to pan,      │
│            scroll to zoom                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3D Visualization: Cluster Orbs

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           ╭───────╮                                             │
│         ╱         ╲                                             │
│        │    ●●●    │   ← Cluster 1 (semi-transparent)           │
│        │   ●●●●●   │     Can see Cluster 2 through it          │
│         ╲  ●●●●  ╱                                              │
│           ╰───────╯                                             │
│                                                                 │
│              ╭───────╮                                          │
│            ╱  ●●●●●  ╲                                          │
│           │   ●●●●●   │  ← Cluster 2 (refractive)               │
│            ╲  ●●●●●  ╱     Light bends through sphere           │
│              ╰───────╯                                          │
│                                                                 │
│  Particle trails emit from cluster centers:                     │
│  ● ─ ─ ─ ─ ○ ─ ─ ─ ─ ●  ← Shows assignment confidence          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Command Palette (Cmd+K)

```
┌─────────────────────────────────────────────────────────────────┐
│  [Q] Search or type command...                              [×] │
├─────────────────────────────────────────────────────────────────┤
│  SUGGESTED FOR YOUR DATA                                        │
│  ─────────────────────────────────────────────────────────────  │
│  🔍 Find anomalies in revenue                               ⏎  │
│  📊 Show correlation matrix                                 ⏎  │
│  📈 Forecast next quarter                                   ⏎  │
│  🔵 Cluster by category                                     ⏎  │
├─────────────────────────────────────────────────────────────────┤
│  TOOLS                                                          │
│  ─────────────────────────────────────────────────────────────  │
│  🛠️ run_pca            Reduce dimensions                      │
│  🛠️ run_kmeans         Cluster data                           │
│  🛠️ run_regression     Predict numeric value                  │
│  🛠️ run_forecast       Time series forecast                   │
├─────────────────────────────────────────────────────────────────┤
│  SESSION                                                        │
│  ─────────────────────────────────────────────────────────────  │
│  📁 Upload new file                                             │
│  🗑️  Clear session                                              │
│  ⚙️  Settings                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Auto-Insight Mode — The "Holy Shit" Moment

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║                                                           ║ │
│  ║   ⚡ Auto-Insight Mode                                    ║ │
│  ║                                                           ║ │
│  ║   I found something interesting...                        ║ │
│  ║                                                           ║ │
│  ║   Revenue peaks every Q4 (expected), but the variance    ║ │
│  ║   is increasing — 2024 had 3x the volatility of 2022.    ║ │
│  ║                                                           ║ │
│  ║   Also, Category C is dragging down margins — it's 40%   ║ │
│  ║   of revenue but 60% of costs.                           ║ │
│  ║                                                           ║ │
│  ║   ┌─────────────────────────────────────────────────┐    ║ │
│  ║   │           [Variance Over Time Chart]            │    ║ │
│  ║   │     ╱╲      ╱╲                                  │    ║ │
│  ║   │    ╱  ╲    ╱  ╲     ╱╲                          │    ║ │
│  ║   │   ╱    ╲  ╱    ╲   ╱  ╲    ╱╲                  │    ║ │
│  ║   │  ╱      ╲╱      ╲ ╱    ╲  ╱  ╲                 │    ║ │
│  ║   │ ╱                ╱      ╲╱    ╲╲╱╲              │    ║ │
│  ║   │ 2020    2021    2022    2023    2024            │    ║ │
│  ║   └─────────────────────────────────────────────────┘    ║ │
│  ║                                                           ║ │
│  ║   [Dig Deeper]  [Show Me More]  [Dismiss]                ║ │
│  ║                                                           ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Upload — "Unfold" Animation

```
Frame 1:  ┌─────────────────────────────┐
          │                             │
          │      Drop your file here    │
          │      or click to browse     │
          │                             │
          └─────────────────────────────┘

Frame 2:  ┌─────────────────────────────┐
          │  📄 budget_2024.csv         │
          │  ────────────────────────   │
          │  ████████████░░░░  45%      │
          │                             │
          │  Parsing rows...            │
          └─────────────────────────────┘

Frame 3:  ┌─────────────────────────────┐
          │  📄 budget_2024.csv         │
          │  ────────────────────────   │
          │  ████████████████ 100%      │
          │                             │
          │  ┌───────────────────────┐  │
          │  │ date     │ revenue │  │  │ ← Header slides in
          │  ├──────────┼─────────┤  │  │
          │  │ 2020-01  │ 1.2M    │  │  │ ← Row 1 fades in
          │  │ 2020-02  │ 1.1M    │  │  │ ← Row 2 fades in
          │  │ ...      │ ...     │  │  │ ← More rows cascade
          │  └───────────────────────┘  │
          └─────────────────────────────┘
```

---

## Sound Design (Optional)

```
┌─────────────────────────────────────────────────────────────────┐
│  Sound Event              Trigger              Volume  Sound    │
├─────────────────────────────────────────────────────────────────┤
│  Keyboard click           Each thought char    20%     Mechanical│
│  Pop                      Tool completes       30%     Soft pop  │
│  Chime (rising)           Big insight found    40%     Rising    │
│  Thud (low)               Error state          25%     Low thud  │
│  Paper rustle             File upload starts   15%     Subtle    │
│  Hum (ambient)            Agent thinking       10%     Very low  │
├─────────────────────────────────────────────────────────────────┤
│  Default: Muted                                                  │
│  Toggle: [🔇] / [🔊] in settings                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design References

| Product | What to Study |
|---------|---------------|
| Linear | Animation curves, micro-interactions |
| Raycast | Command palette, keyboard-first UX |
| Obsidian | Warm dark theme, information density |
| The Pudding | Data storytelling |
| Bloomberg Terminal | Monospace data legibility |

---

*DataLens AI — Visual Design Spec v2.0 · February 2026*
