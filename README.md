# DC Crash Network Recovery Prototype

A Streamlit dashboard and validation pipeline for identifying road-network matching issues in Washington, DC crash data — and recovering missing intersection identifiers with measurable accuracy.

**Live demo:** [Link](https://dc-crash-recovery-7s8gwg5ywxzzg4ahnt2ern.streamlit.app/)

---

## Why this exists

The initial hypothesis was that DC crash data would have a large number of unmappable records, similar to the Madison case study published by Citian. Exploration showed otherwise: nearly all DC crash records have valid coordinates. The real problem is different.

About **20.6%** of recent DC crash records (~9,100 of ~44,000 in 2024–2026) are mapped but carry road-network matching warnings — they have coordinates but unreliable intersection, route, or block assignments. These records contain real safety events, including injury and fatal crashes, so excluding them biases hotspot analysis. Including them as-is distorts corridor-level screening.

This prototype takes those records seriously: identifying them, validating a recovery method against held-out ground truth, and building a CRASH-style screening interface a planner could open on a Tuesday morning.

---

## What the app shows

- **Top-level KPIs** — total crashes, records with quality issues, recovery pipeline accuracy
- **Interactive DC map** — recent crashes colored by severity; top-50 hotspot intersections sized by crash volume
- **Network-level hotspot table** — ranked intersections filtered by year, ward, mode, and severity
- **Intersection details panel** — click a hotspot to see crash burden, fatalities, and data-quality breakdown
- **Method progression visualization** — V1 → V2 → V3 accuracy with the engineering reasoning for each jump
- **Deep dive expanders** — data sources, cleaning decisions, validation design, what I learned, what comes next

---

## Recovery pipeline

The recovery target is `NEARESTINTKEY`, DDOT's nearest-intersection network identifier. `STREETSEGID` and `ROADWAYSEGID` are missing for ~99.86% of recent records — likely an upstream schema change at DDOT — so they aren't viable targets in this dataset.

Method progression:

| Method | Description | Accuracy |
|---|---|---|
| V1 | Nearest official intersection by distance only | 73.0% |
| V2 | Top-5 nearest intersections + address text matching | 79.8% |
| V3 | Top-5 + both-street name scoring (address + nearest cross-street) | **90.4%** |

Key insight: the correct intersection is in the top-5 nearest 96.4% of the time. The main challenge isn't finding the right area — it's ranking the candidates correctly. Re-ranking the top-5 using street-name context adds +17.4 percentage points over a pure coordinate snap.

---

## Validation design

The 500-record validation set is drawn from HIGH-quality crash records where DDOT's `NEARESTINTKEY` assignment is already trusted. The target value is artificially hidden; the pipeline reconstructs it; accuracy is measured against what was held aside.

Records are severity-stratified so rare but high-consequence crashes are always measurable:

- 30 fatal crashes
- 80 major-injury crashes
- 200 minor / unknown-injury crashes
- 190 property-damage-only crashes

Reproducibility: `random_state=42` throughout. Validation truth and inputs are stored as separate files; pipeline code never touches the truth file.

**What this validation does and doesn't measure.** The 90.4% accuracy is measured on synthetically-degraded HIGH records, not on real MEDIUM records. Real MEDIUM records may be systematically harder — they're cases where DDOT's own pipeline gave up, often because of mid-block coordinates, complex junctions, or unstructured addresses. The 90.4% is the **upper bound** of what to expect on production MEDIUM data. Hand-validating a stratified sample of real MEDIUM records is the most important next step.

---

## Data sources

Both datasets are from Open Data DC and used in compliance with their terms.

- **`Crashes_in_DC.csv`** — 348,000 crash records, 66 columns; filtered to the 44,348 records from 2024–2026 for the working dataset.
- **`Intersection_Points.csv`** — DDOT's authoritative Master Address Repository intersection layer; 8,424 active intersections after cleaning, used as the reference layer for the recovery pipeline.

---

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app loads `data/processed/crashes_clean_recent.csv` directly. To regenerate it from the raw data, run `notebooks/01_exploration.ipynb` end-to-end after placing `Crashes_in_DC.csv` and `Intersection_Points.csv` in `data/raw/`.

---

## Repository structure
dc-crash-recovery/
├── app/
│   └── streamlit_app.py          # The Streamlit interface
├── data/
│   ├── raw/                      # Source CSVs (gitignored, downloaded from Open Data DC)
│   ├── processed/                # Cleaned working dataset
│   └── app/                      # Pre-computed files for the app
├── notebooks/
│   └── 01_exploration.ipynb      # End-to-end analysis: cleaning, location quality, validation
├── src/                          # Supporting modules
├── requirements.txt
└── README.md

---

## What I'd build next

1. **Validate on real MEDIUM records.** Hand-verify a stratified sample of ~50 MEDIUM crashes against Google Maps to convert the synthetic upper-bound into a measured real-world number. Expect the accuracy to drop, possibly to 70–80%.
2. **Confidence scoring.** Expose match-score and nearest-intersection distance as per-record confidence flags so low-confidence predictions can be routed to manual review.
3. **Segment-level recovery.** Apply the same architecture to `STREETSEGID` if DDOT republishes it, or recover segment assignments using OpenStreetMap road geometry.
4. **Mid-block and ramp handling.** Crashes clearly not at intersections — highway ramps, mid-block driveways — should use a separate segment-snap pipeline rather than being forced to intersection-level matching.

---

*Built by [Krishav Gulati](https://github.com/gulatikrishav)*
