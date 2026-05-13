# DC Crash Data Quality + Recovery Prototype

This project is a Streamlit prototype for exploring recent Washington, DC crash data, identifying road-network matching issues, and validating an official-intersection recovery method.

## Project Summary

The initial hypothesis was that DC crash data would have a large number of unmappable records. After exploring the data, I found another issue: most records have coordinates, but a meaningful share are mapped with road-network matching warnings.

In the 2024–2026 dataset, about 20.6% of records are classified as 'medium' location quality. Many of these records still contain real safety events, including injury and fatal crashes, but they may be unreliable for precise intersection or corridor-level analysis.

This prototype demonstrates a workflow for identifying, validating, and partially recovering those records.

## What the App Shows

- Recent DC crash records from 2024–2026
- Location-quality classifications: HIGH vs. MEDIUM
- Crash hotspot screening by intersection
- Filters by year, ward, mode, and severity
- A validation-tested intersection recovery pipeline
- Proposed recovered intersections for medium-quality records with confidence labels

## Recovery Pipeline

The recovery target is `NEARESTINTKEY`, the nearest-intersection network identifier.

I validated the method using high-quality records where the true `NEARESTINTKEY` was already known. I hid the target value, snapped each crash to official DC intersection points, and evaluated whether the method recovered the original key.
The next step from here would be applying this to medium-quality records. 

Method progression:

| Method | Description | Accuracy |
|---|---|---:|
| V1 | Nearest official intersection by distance only | 73.0% |
| V2 | Top-5 nearest intersections + address matching | 79.8% |
| V3 | Top-5 nearest intersections + both-street scoring | 90.4% |

The main finding is that the correct intersection is usually nearby, but it is not always the single closest point. Re-ranking the top 5 candidates using street-name context substantially improves performance.

## Validation Design

The validation sample contains 500 high-quality records, stratified by crash severity:

- 30 fatal crashes
- 80 major-injury crashes
- 200 minor/unknown-injury crashes
- 190 property-damage-only crashes

This ensures the pipeline is tested on both common and high-consequence crash types.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
