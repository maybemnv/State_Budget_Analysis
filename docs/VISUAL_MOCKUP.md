# DataLens AI — Visual Mockup

## Full Workspace — Dark Terminal Aesthetic

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  DataLens AI                                                         ─ □ ×    ║
╠══════════════╦══════════════════════════════════════════════════════╦═════════╣
║              ║                                                       ║         ║
║  📊 state_   ║     ◉                          🔌 WebSocket Connected ║  VIZ    ║
║  budget.csv  ║   (purple glow)                                       ║         ║
║              ║                                                       ║ ─────── ║
║  12.4K rows  ║  You: Show me what's driving spending anomalies      ║         ║
║  8 columns   ║                                                       ║  ┌───┐  ║
║              ║  ┌──────────────────────────────────────────────────┐ ║  │ ● │  ║
║  COLUMNS     ║  │ Thought                                          │ ║  │   │  ║
║  ─────────   ║  │ I need to identify which columns have unusual    │ ║  │ ● │  ║
║  📊 Amount   │  │ patterns. Let me run anomaly detection on the    │ ║  │   │  ║
║  🔢 Year     │  │ spending data...                                 │ ║  └───┘  ║
║  🏷️ Category │  └──────────────────────────────────────────────────┘ ║  PCA    ║
║  📍 Agency   │                                                       ║  Depth  ║
║  🔢 Program  │  ┌──────────────────────────────────────────────────┐ ║  blur   ║
║              │  │ 🛠️ detect_anomalies                              │ ║  on     ║
║  QUICK STATS │  │ { columns: ["Amount"], method: "isolation_forest"}│ ║  distant║
║  ──────────  │  └──────────────────────────────────────────────────┘ ║  points ║
║  Mean: $2.3M │                                                       ║         ║
║  Std: $890K  │  ┌──────────────────────────────────────────────────┐ ║  ┌───┐  ║
║  Skew: 2.1   │  │ Result                                           │ ║  │ ● │  ║
║  Missing: 0  │  │ Found 47 anomalous transactions (3.8%)           │ ║  │ ● │  ║
║              │  │ Total anomaly value: $12.4M                      │ ║  │ ● │  ║
║  DATA TYPES  │  │ Top anomaly: Agency X, Program 42, $890K         │ ║  └───┘  ║
║  ──────────  │  └──────────────────────────────────────────────────┘ ║  Cluster║
║  ▸ Numeric:4 │                                                       ║  Orbs   ║
║  ▸ Text: 3   │  ╔══════════════════════════════════════════════════╗ ║  Refrac-║
║  ▸ Date: 1   │  ║ Final Answer                                     ║ ║  tive   ║
║              │  ║ Agency X's Program 42 is a major outlier.        ║ ║  spheres║
║  [⚙️ Settings]║  ║ At $890K, it's 4.2x the average spend.           ║ ║  with   ║
║  [🤓 LLM ▼]  ║  ║ This single program accounts for 7% of total     ║ ║  trails ║
║              │  ║ expenditure but shows no corresponding output    ║ ║         ║
║  SESSION     │  ║ metrics. Recommend audit.                        ║ ║  [⛶]   ║
║  ─────────   │  ╚══════════════════════════════════════════════════╝ ║  [📷]   ║
║  ● Q4 2024   │                                                       ║  [📥]   ║
║  ○ Q3 2024   │  ──────────────────────────────────────────────────── ║         ║
║  ○ Q2 2024   │  [ Type your question...                       ] [→]  ║         ║
║  ○ Q1 2024   │  [ ⌘K for suggestions                            ]    ║         ║
║              │                                                       ║         ║
╠══════════════╩══════════════════════════════════════════════════════╩═════════╣
║  AGENT TIMELINE — Click to jump                                               ║
║  ●────────────○────────────●────────────○────────────●                        ║
║  describe     stats        anomaly      chart        answer                   ║
║  10:42:01     10:42:03     10:42:05     10:42:07     10:42:09                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Command Palette (⌘K)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  [Q] Search or type command...                                              [×] │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SUGGESTED FOR YOUR DATA                                                        │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  🔍 Find anomalies in spending                                              ⏎  │
│  📊 Show correlation matrix                                                 ⏎  │
│  📈 Forecast next fiscal year                                               ⏎  │
│  🔵 Cluster agencies by spending pattern                                    ⏎  │
│  📉 Detect trends over time                                                 ⏎  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TOOLS                                                                          │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  🛠️  detect_anomalies      Find outliers using Isolation Forest               │
│  🛠️  run_correlation       Compute Pearson correlation matrix                 │
│  🛠️  run_kmeans            Cluster data into K groups                         │
│  🛠️  run_pca               Reduce dimensions, find principal components       │
│  🛠️  run_forecast          ARIMA/Prophet time series forecast                 │
│  🛠️  run_regression        Predict numeric target variable                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SESSION                                                                        │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  📁 Upload new dataset                                                          │
│  🗑️  Clear current session                                                      │
│  📊 Export analysis results (PDF/CSV)                                           │
│  ⚙️  Settings                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Auto-Insight Mode — The "Holy Shit" Moment

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ⚡  Auto-Insight Mode                                                       ║
║   ═══════════════════                                                         ║
║                                                                               ║
║   I analyzed your dataset and found 3 non-obvious patterns:                  ║
║                                                                               ║
║   ┌─────────────────────────────────────────────────────────────────────────┐ ║
║   │  1️⃣  Spending variance is increasing over time                          │ ║
║   │                                                                       │ ║
║   │   While total budget grew 12% annually, the variance between         │ ║
║   │   agencies increased 3x faster — suggesting diverging priorities.     │ ║
║   │                                                                       │ ║
║   │   ┌─────────────────────────────────────────────────────────────┐    │ ║
║   │   │  Variance (σ²) by Year                                      │    │ ║
║   │   │                                                             │    │ ║
║   │   │      ╱╲                    ╱╲                               │    │ ║
║   │   │     ╱  ╲                  ╱  ╲         ╱╲                   │    │ ║
║   │   │    ╱    ╲                ╱    ╲       ╱  ╲      ╱╲          │    │ ║
║   │   │   ╱      ╲              ╱      ╲     ╱    ╲    ╱  ╲         │    │ ║
║   │   │  ╱        ╲            ╱        ╲   ╱      ╲  ╱    ╲╱╲       │    │ ║
║   │   │ 2020      2021        2022      2023      2024               │    │ ║
║   │   └─────────────────────────────────────────────────────────────┘    │ ║
║   │                                                                       │ ║
║   └───────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║   ┌─────────────────────────────────────────────────────────────────────────┐ ║
║   │  2️⃣  Category C drains 60% of budget but is only 40% of programs       │ ║
║   │                                                                       │ ║
║   │   This category has 2.1x higher cost-per-program than the average.   │ ║
║   │   Agencies X, Y, Z account for 78% of Category C spend.              │ ║
║   │                                                                       │ ║
║   └───────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║   ┌─────────────────────────────────────────────────────────────────────────┐ ║
║   │  3️⃣  Q4 spending spikes correlate with lower performance scores        │ ║
║   │                                                                       │ ║
║   │   Agencies that spend >30% of budget in Q4 show 23% lower            │ ║
║   │   outcome scores — suggesting rushed end-of-year spending.           │ ║
║   │                                                                       │ ║
║   └───────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║              [ 🔍 Dig Deeper ]    [ 📊 Show Visualizations ]    [ ✕ Dismiss ]║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3D PCA Scatter — Annotated

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                           ● PC3 (18% variance)                                  │
│                          ╱                                                      │
│                         ╱                                                       │
│                        ●                                                        │
│                   ╱●         ●                                                  │
│              ╱●                 ●                                               │
│         ╱●    [●]  ●    ●         ●                                             │  ← Points glow
│              ●    ╲●                 ●                                          │    on hover
│                   ╲    ●                                                        │
│                        ●    ●                                                   │
│                                                                                 │
│              PC1 (42% variance) ───────────────────►                              │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  Point #847 [●]                                                           │  │
│  │  ─────────────────────────────────                                        │  │
│  │  PC1:  2.34   (contributes: Amount, Year)                                 │  │
│  │  PC2: -1.23   (contributes: Category C)                                   │  │
│  │  PC3:  0.89   (contributes: Agency X)                                     │  │
│  │                                                                             │  │
│  │  Raw values:                                                                │  │
│  │  Amount: $890,234  |  Year: 2024  |  Category: C  |  Agency: X            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  [Left-click drag: Rotate]  [Right-click: Pan]  [Scroll: Zoom]                  │
│  [Double-click point: Show details]                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Cluster Orbs — Refractive 3D Spheres

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│           ╭─────────────────╮                                                   │
│         ╱    ●  ●  ●  ●    ╲                         ╭───────────╮             │
│        │   ●  ●  ●  ●  ●  ●  │                      ╱  ●  ●  ●  ╲              │
│        │  ●   [●]  ●   ●  ● │ ← Cluster 1          │   ●  ●  ●   │             │
│        │   ●  ●  ●  ●  ●  ●  │   (n=234, 42%)      │  ●  ●  ●  ● │ ← Cluster 2 │
│         ╲    ●  ●  ●  ●    ╱    Semi-transparent   │   ●  ●  ●   │   (n=189,   │
│           ╰─────────────────╯     Refractive       │  ●  ●  ●  ● │     34%)    │
│                                  Density visible   │   ●  ●  ●   │             │
│                                  through layers    ╲  ●  ●  ●  ╱              │
│                                                     ╰───────────╯             │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  Cluster 1 Profile                                                        │  │
│  │  ─────────────────────────────────                                        │  │
│  │  Size: 234 agencies (42% of total)                                        │  │
│  │  Avg spend: $2.1M  |  Std: $340K                                          │  │
│  │  Dominant category: A (67%)                                               │  │
│  │  Top agencies: X, Y, Z                                                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Particle trails show assignment confidence:                                    │
│  ● ─ ─ ─ ○ ─ ─ ─ ●  ← Dotted line = low confidence                            │
│  ● ──────●────── ●  ← Solid line = high confidence                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Upload — "Unfold" Animation Sequence

```
Frame 1: Initial State
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                                                         │
│           Drop your file here                           │
│           or click to browse                            │
│                                                         │
│           Supported: CSV, XLSX, Parquet (max 100MB)    │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘

Frame 2: Upload Progress (45%)
┌─────────────────────────────────────────────────────────┐
│  📄 state_budget_2024.csv                               │
│  ─────────────────────────────────────────────────────  │
│  ████████████████░░░░░░░░░░░░░░░░  45%                 │
│                                                         │
│  Uploading... 12.4MB / 27.8MB                          │
│                                                         │
└─────────────────────────────────────────────────────────┘

Frame 3: Parsing — Rows Cascade In
┌─────────────────────────────────────────────────────────┐
│  📄 state_budget_2024.csv                               │
│  ─────────────────────────────────────────────────────  │
│  ████████████████████████████████  100%                │
│                                                         │
│  Parsing rows...                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ date     │ agency  │ category │ amount    │ year │  │ ← Header slides in
│  ├──────────┼─────────┼──────────┼───────────┼──────┤  │
│  │ 2024-01  │ AG-001  │ A        │ $123,456  │ 2024 │  │ ← Row 1 fades in
│  │ 2024-02  │ AG-002  │ B        │ $234,567  │ 2024 │  │ ← Row 2 fades in
│  │ 2024-03  │ AG-001  │ C        │ $345,678  │ 2024 │  │ ← Row 3 fades in
│  │ ...      │ ...     │ ...      │ ...       │ ...  │  │ ← More cascade
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  12,438 rows parsed in 2.3s                            │
│                                                         │
│  [ Start Analyzing → ]                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Avatar States

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  THINKING                    EXECUTING                   DONE                   │
│                                                                                 │
│       ◉                           ◉                         ◉                   │
│     ╱   ╲                       ╱   ╲                                           │
│    │  ●  │  Purple glow         │  ●  │  Orange glow       │  ●  │  Teal       │
│     ╲   ╱   Slow pulse          ╲   ╱   Fast pulse        ╲   ╱   Steady       │
│       ◉     (~1Hz)                ◉     (~3Hz)                ◉     (no pulse)  │
│                                                                                 │
│  When: Agent reasoning         When: Tool executing      When: Answer ready    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Color Reference — Side by Side

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  BACKGROUNDS                                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │          │ │          │ │          │ │          │ │          │              │
│  │ #0A0A0F  │ │ #14141A  │ │ #1E1E28  │ │ #2A2A35  │ │ #3D3D4D  │              │
│  │ Warm     │ │ Surface  │ │ Elevated │ │ Border   │ │ Border   │              │
│  │ Black    │ │          │ │          │ │          │ │ Hover    │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ACCENTS                                                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │          │ │          │ │          │ │          │ │          │              │
│  │ #FF6B35  │ │ #00DCB4  │ │ #9D4EDD  │ │ #F59E0B  │ │ #EF4444  │              │
│  │ Burnt    │ │ Teal     │ │ Deep     │ │ Amber    │ │ Red      │              │
│  │ Orange   │ │ (success)│ │ Purple   │ │ (warn)   │ │ (error)  │              │
│  │ (primary)│ │          │ │ (agent)  │ │          │ │          │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TEXT                                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │          │ │          │ │          │ │          │ │          │              │
│  │ #E8E6E3  │ │ #C8C4BC  │ │ #8B8878  │ │ #5A5850  │ │ #FFFFFF  │              │
│  │ Primary  │ │ Secondary│ │ Muted    │ │ Disabled │ │ Pure     │              │
│  │          │ │          │ │          │ │          │ │ (avoid)  │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*DataLens AI — Visual Mockup v2.0 · February 2026*
