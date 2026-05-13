from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DC Crash Recovery",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styling ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #07111F; }
    section[data-testid="stSidebar"] { background-color: #0E1B2A; }
    h1, h2, h3 { letter-spacing: -0.03em; }
    div[data-testid="stMetric"] {
        background-color: #0E1B2A;
        border: 1px solid rgba(148,163,184,0.18);
        padding: 1rem 1.25rem;
        border-radius: 14px;
    }
    .stAlert { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants (from validated pipeline runs) ───────────────────────────────────
V1_ACC = 73.0   # nearest official intersection, coordinate snap only
V2_ACC = 79.8   # top-5 candidates + address text
V3_ACC = 90.4   # top-5 + both-street name scoring

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_crashes():
    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(
        base / "data" / "processed" / "crashes_clean_recent.csv",
        low_memory=False,
    )
    df["REPORTDATE"] = pd.to_datetime(df["REPORTDATE"], errors="coerce")
    if "crash_year" not in df.columns:
        df["crash_year"] = df["REPORTDATE"].dt.year
    df["crash_year"] = pd.to_numeric(df["crash_year"], errors="coerce").astype("Int64")

    for col in ["has_pedestrian", "has_bicyclist", "has_fatality", "has_major_injury", "has_speeding"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    for col in ["total_minor_injuries", "total_unknown_injuries",
                "total_fatalities", "total_major_injuries", "severity_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "NEARESTINTSTREETNAME" in df.columns:
        _bad = {"ROUTE NOT FOUND", "UNKNOWN", "NOT FOUND", ""}
        df["NEARESTINTSTREETNAME"] = df["NEARESTINTSTREETNAME"].where(
            ~df["NEARESTINTSTREETNAME"].str.strip().str.upper().isin(_bad), other=pd.NA
        )

    return df


crashes = load_crashes()

# Compute medium share from full dataset (used in pipeline story text)
medium_share_pct = (
    (crashes["location_quality"] == "MEDIUM").mean() * 100
    if "location_quality" in crashes.columns else 0.0
)


# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")

    all_years = sorted(crashes["crash_year"].dropna().astype(int).unique().tolist())
    sel_years = st.multiselect("Year", all_years, default=all_years)

    all_wards = sorted(crashes["WARD"].dropna().unique().tolist())
    sel_wards = st.multiselect("Ward", all_wards, default=all_wards)

    MODE_OPTS = ["Motor vehicle", "Pedestrian", "Bicyclist"]
    sel_modes = st.multiselect("Mode", MODE_OPTS, default=MODE_OPTS)

    SEV_OPTS = ["Fatal", "Major injury", "Minor / Unknown", "Property damage only"]
    sel_sev = st.multiselect("Severity", SEV_OPTS, default=SEV_OPTS)

    st.divider()
    st.caption("Filters apply to the map, hotspot table, and KPIs.")


# ── Filter logic ───────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if sel_years:
        df = df[df["crash_year"].isin(sel_years)]
    if sel_wards:
        df = df[df["WARD"].isin(sel_wards)]

    if len(sel_modes) < len(MODE_OPTS):
        m = pd.Series(False, index=df.index)
        if "Motor vehicle" in sel_modes:
            m |= ~df["has_pedestrian"] & ~df["has_bicyclist"]
        if "Pedestrian" in sel_modes:
            m |= df["has_pedestrian"]
        if "Bicyclist" in sel_modes:
            m |= df["has_bicyclist"]
        df = df[m]

    if len(sel_sev) < len(SEV_OPTS):
        s = pd.Series(False, index=df.index)
        if "Fatal" in sel_sev:
            s |= df["has_fatality"]
        if "Major injury" in sel_sev:
            s |= df["has_major_injury"] & ~df["has_fatality"]
        if "Minor / Unknown" in sel_sev:
            s |= (
                (df["total_minor_injuries"] + df["total_unknown_injuries"] > 0)
                & ~df["has_major_injury"]
                & ~df["has_fatality"]
            )
        if "Property damage only" in sel_sev:
            s |= df["severity_score"] == 0
        df = df[s]

    return df.copy()


filtered = apply_filters(crashes)


# ── Hotspot computation ────────────────────────────────────────────────────────
_SENTINEL_KEYS = {"route not found", "not found", "unknown", "none", ""}

def compute_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["NEARESTINTKEY", "LATITUDE", "LONGITUDE"])
    # Remove DDOT sentinel keys that are not real intersections
    df = df[~df["NEARESTINTKEY"].str.strip().str.lower().isin(_SENTINEL_KEYS)]
    def first_valid(s):
        clean = s.dropna()
        return clean.iloc[0] if len(clean) > 0 else None

    grp = (
        df.groupby("NEARESTINTKEY", as_index=False)
        .agg(
            intersection=("NEARESTINTSTREETNAME", first_valid),
            address_fallback=("ADDRESS", "first"),
            crashes=("CRIMEID", "count"),
            fatalities=("total_fatalities", "sum"),
            major_injuries=("total_major_injuries", "sum"),
            lat=("LATITUDE", "mean"),
            lon=("LONGITUDE", "mean"),
            n_medium=("location_quality", lambda x: (x == "MEDIUM").sum()),
            n_total=("location_quality", "count"),
        )
        .sort_values("crashes", ascending=False)
        .reset_index(drop=True)
    )
    grp["rank"] = grp.index + 1
    grp["quality_issues_pct"] = (grp["n_medium"] / grp["n_total"] * 100).round(1)
    # Fall back to crash address when no clean street name exists for the group
    grp["address_fallback"] = grp["address_fallback"].str.replace(r"\s*\n.*", "", regex=True).str.strip()
    grp["intersection"] = grp["intersection"].fillna(grp["address_fallback"]).fillna("Unknown")
    grp.drop(columns=["address_fallback"], inplace=True)
    return grp


hotspots = compute_hotspots(filtered)
top20 = hotspots.head(20).copy()
top50 = hotspots.head(50).copy()


# ── Session state ──────────────────────────────────────────────────────────────
if "selected_int" not in st.session_state:
    st.session_state["selected_int"] = None


# ── Title + KPIs ───────────────────────────────────────────────────────────────
st.title("DC Crash Network Recovery Prototype")
st.caption(
    "Identifies road-network matching problems in recent DC crash records and validates "
    "a pipeline that recovers missing intersection identifiers — built on 2024–2026 Open Data DC."
)

n_filtered = len(filtered)
n_medium = int((filtered["location_quality"] == "MEDIUM").sum()) if "location_quality" in filtered.columns else 0

k1, k2, k3 = st.columns(3)
k1.metric("Crashes", f"{n_filtered:,}")
k2.metric("Records with quality issues", f"{n_medium:,}", help="MEDIUM location quality — coordinates valid but network matching flagged")
k3.metric(
    "Recovery accuracy (V3)",
    f"{V3_ACC:.1f}%",
    delta=f"+{V3_ACC - V1_ACC:.1f}pp vs coordinate-snap baseline",
)

st.divider()

# ── Key Finding & Small Cards ──────────────────────────────────────────────────────────
st.markdown("### Key finding")

st.markdown(
    f"""
    DC's crash data is not primarily missing coordinates. The more important problem is
    network reliability: about **{medium_share_pct:.1f}%** of recent crash records are mapped
    but still carry location-quality warnings. These records can still involve real injuries
    and serious crashes, so simply dropping them would weaken safety analysis.

    This prototype treats those records as recoverable or reviewable, rather than unusable.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Original hypothesis**")
    st.write("Find unmappable DC crashes and recover their locations.")

with c2:
    st.markdown("**What the data showed**")
    st.write("Most records are mapped, but many are not reliable enough for network-level analysis.")

with c3:
    st.markdown("**Prototype response**")
    st.write("Validate an official-intersection recovery method and build a crash-screening interface.")

# ── Map + right panel ──────────────────────────────────────────────────────────
map_col, panel_col = st.columns([0.65, 0.35])

with map_col:
    # Build crash point colors (severity-coded)
    map_df = filtered.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    if len(map_df) > 8_000:
        map_df = map_df.sample(8_000, random_state=42)

    n = len(map_df)
    colors = np.tile([56, 189, 248, 130], (n, 1))   # default: PDO blue
    minor = (map_df["total_minor_injuries"].values + map_df["total_unknown_injuries"].values) > 0
    major = map_df["has_major_injury"].values.astype(bool)
    fatal = map_df["has_fatality"].values.astype(bool)
    colors[minor] = [250, 204, 21, 170]    # yellow
    colors[major] = [251, 146, 60, 200]    # orange
    colors[fatal] = [220, 38, 38, 230]     # red

    crash_pts = pd.DataFrame({
        "lat": map_df["LATITUDE"].values,
        "lon": map_df["LONGITUDE"].values,
        "color": colors.tolist(),
    })

    # Scale hotspot circles by crash volume
    hs = top50.copy()
    if len(hs) > 0:
        max_c = hs["crashes"].max()
        hs["radius"] = (50 + (hs["crashes"] / max_c) * 60).astype(int)
    else:
        hs["radius"] = 80

    crash_layer = pdk.Layer(
        "ScatterplotLayer",
        data=crash_pts,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=12,
        radius_units="pixels",
        pickable=False,
        opacity=0.8,
    )

    hotspot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=hs[["lat", "lon", "radius", "intersection", "crashes", "fatalities", "quality_issues_pct"]],
        get_position=["lon", "lat"],
        get_fill_color=[251, 146, 60, 70],
        get_line_color=[255, 200, 100, 220],
        get_radius="radius",
        radius_units="meters",
        stroked=True,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[crash_layer, hotspot_layer],
            initial_view_state=pdk.ViewState(
                latitude=38.905, longitude=-77.016, zoom=11, pitch=0
            ),
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={
                "html": (
                    "<b>{intersection}</b><br/>"
                    "{crashes} crashes &nbsp;·&nbsp; {fatalities} fatalities<br/>"
                    "{quality_issues_pct}% have quality issues"
                ),
                "style": {
                    "background": "#0E1B2A",
                    "color": "#F8FAFC",
                    "font-size": "0.85rem",
                    "padding": "10px 14px",
                    "border-radius": "8px",
                    "border": "1px solid rgba(148,163,184,0.25)",
                },
            },
        ),
        use_container_width=True,
    )
    st.caption(
        "🔵 PDO · 🟡 Minor/Unknown · 🟠 Major injury · 🔴 Fatal  "
        " | Amber rings = top 50 hotspot intersections, size ∝ crash volume. Hover for details. "
        f"Showing {len(crash_pts):,} of {len(filtered):,} crashes — large filter sets are subsampled to 8,000 for map performance."
    )

with panel_col:
    st.markdown("#### Intersection details")
    sel_key = st.session_state.get("selected_int")

    if sel_key is None:
        st.info("Click a row in the **Hotspot Table** below to inspect an intersection here.")

        # Default preview: show #1 hotspot
        if len(top20) > 0:
            t = top20.iloc[0]
            st.markdown(f"**Top hotspot preview**")
            st.markdown(f"_{t['intersection']}_")
            pa, pb = st.columns(2)
            pa.metric("Crashes", int(t["crashes"]))
            pb.metric("Fatalities", int(t["fatalities"]))
            pa.metric("Major injuries", int(t["major_injuries"]))
            pb.metric("Quality issues", f"{t['quality_issues_pct']:.1f}%")
    else:
        row = hotspots[hotspots["NEARESTINTKEY"] == sel_key]
        if len(row) == 0:
            st.info("Intersection not in current filter view. Adjust filters or select another.")
        else:
            row = row.iloc[0]
            st.markdown(f"**{row['intersection']}**")
            st.caption(f"Rank #{int(row['rank'])} · {int(row['crashes'])} crashes under current filters")

            pa, pb = st.columns(2)
            pa.metric("Fatalities", int(row["fatalities"]))
            pb.metric("Major injuries", int(row["major_injuries"]))
            pa.metric("Quality issues", f"{row['quality_issues_pct']:.1f}%")
            pb.metric("Rank", f"#{int(row['rank'])}")

            recent = (
                filtered[filtered["NEARESTINTKEY"] == sel_key]
                .sort_values("REPORTDATE", ascending=False)
                .head(8)
            )
            if len(recent) > 0:
                st.markdown("**Recent crashes**")
                disp = recent[["REPORTDATE", "ADDRESS", "severity_score", "location_quality"]].copy()
                disp["REPORTDATE"] = pd.to_datetime(disp["REPORTDATE"]).dt.strftime("%Y-%m-%d")
                disp.columns = ["Date", "Address", "Severity", "Quality"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

        if st.button("Clear selection", use_container_width=True):
            st.session_state["selected_int"] = None
            st.rerun()

st.divider()


# ── Hotspot ranked table ───────────────────────────────────────────────────────
st.header("Network-level hotspot screening")
st.caption("Ranked intersections under current filters. Select a row to inspect crash burden and data-quality issues.")

table_df = top20[["rank", "intersection", "crashes", "fatalities", "major_injuries", "quality_issues_pct"]].copy()
table_df.columns = ["Rank", "Intersection", "Crashes", "Fatalities", "Major injuries", "Quality issues %"]

event = st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

if event.selection.rows:
    new_key = top20.iloc[event.selection.rows[0]]["NEARESTINTKEY"]
    if st.session_state.get("selected_int") != new_key:
        st.session_state["selected_int"] = new_key
        st.rerun()

st.divider()


# ── Pipeline story ─────────────────────────────────────────────────────────────
st.header("Why this needs a pipeline")

st.markdown(
    f"""
    About **{medium_share_pct:.1f}%** of recent DC crash records have road-network matching warnings.
    These records still have coordinates, but their intersection, route, or block assignments may be
    less reliable. If left uncorrected, they can distort hotspot rankings or get excluded from
    corridor-level analysis.

    The recovery pipeline tests whether those records can be matched back to official intersection
    identifiers with measurable accuracy.
    """
)

# 4-step progression chart
steps_df = pd.DataFrame([
    {"Step": "V1 — coordinate snap",      "Accuracy": V1_ACC, "Note": "Nearest official intersection by distance only"},
    {"Step": "V2 — top-5 + address",      "Accuracy": V2_ACC, "Note": "Candidate set re-ranked with address text"},
    {"Step": "V3 — top-5 + street names", "Accuracy": V3_ACC, "Note": "Both-street name scoring on top-5"},
])

fig = go.Figure(
    go.Bar(
        x=steps_df["Step"],
        y=steps_df["Accuracy"],
        text=[f"{a:.1f}%" for a in steps_df["Accuracy"]],
        textposition="outside",
        customdata=steps_df["Note"],
        hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<br>%{customdata}<extra></extra>",
        marker_color=["#38BDF8", "#60A5FA", "#22D3EE"],
        marker_line_color="rgba(255,255,255,0.12)",
        marker_line_width=1,
    )
)
fig.update_layout(
    title="Recovery accuracy — method progression",
    yaxis=dict(
        title="Accuracy (%)",
        range=[0, 100],
        gridcolor="rgba(148,163,184,0.12)",
    ),
    xaxis_title="",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#CBD5E1"),
    height=320,
    margin=dict(t=50, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.success(
    "Top-5 candidate coverage was 96.4%, meaning the correct intersection was usually nearby; the main challenge was ranking the candidates correctly."
)

st.info(
    f"**Key result:** Re-ranking the top-5 nearest intersection candidates using street-name "
    f"context adds **+{V3_ACC - V1_ACC:.1f} percentage points** over a pure coordinate snap. "
    "The correct intersection is often not the single geometrically nearest point — "
    "especially for mid-block crashes and complex junctions."
)

st.divider()


# ── Deep dive ──────────────────────────────────────────────────────────────────
st.markdown("## Deep dive")

with st.expander("1. Data sources & scope"):
    st.markdown("""
    **Datasets**
    - `Crashes_in_DC.csv` (Open Data DC) — 348,000 records, 66 columns
    - `Intersection_Points.csv` (Open Data DC) — DDOT's authoritative intersection layer, 8,424 active intersections after cleaning

    **Scope**
    - Filtered to 2024–2026 (44,348 records) as the working dataset.
    - Older records (2009–2013) contain extreme injury outliers — one record reports 51 injuries — likely from legacy reporting patterns. Recent records max at 7–14 injuries and are more interpretable.
    - The full 348K historical dataset is preserved separately for any longitudinal analysis.
    """)

with st.expander("2. Data preparation"):
    st.markdown("""
    Three things had to happen before any recovery method could run: cleaning the raw crash data, defining what "data quality" means at the record level, and choosing the right field to recover.

    **Cleaning decisions.** Three load-bearing choices that shape every downstream number:
    - **Missing injury counts → 0.** Fields like `FATALBICYCLIST` are blank when no bicyclist was involved. Filled with 0 in the first pass.
    - **Severity score weights (100 / 10 / 3 / 1).** Each fatality counts 100, major injury 10, minor injury 3, unknown 1. The ratios mirror the AASHTO Highway Safety Manual's EPDO framework.
    - **2,349 → 6 location-error categories.** Raw `LOCATIONERROR` strings contain specific block IDs, making them effectively unique per record. Collapsed into six useful categories: `FAR_FROM_CENTERLINE`, `INTERSECTING_ROUTE_ERROR`, `BLOCKKEY_ERROR`, `BLOCKKEY_AND_CORRIDOR_ERROR`, `CORRIDOR_ERROR`, `OTHER_LOCATION_ERROR`.

    **Location quality framework.** HIGH / MEDIUM / LOW labels combine three independent signals. A record needs to pass *all three* to be HIGH — a conservative classifier.

    | Signal | Meaning |
    |---|---|
    | `has_valid_lat_lon` | Coordinates inside DC bounding box |
    | `has_location_error` is False | DDOT's LOCATIONERROR field is null |
    | `MAR_SCORE ≥ 80` | Address matched DDOT's Master Address Repository with strong confidence |

    Result on 2024–2026 records: **79.4% HIGH, 20.6% MEDIUM, <0.1% LOW.** The 20.6% MEDIUM bucket is what the recovery pipeline targets. These records have valid coordinates but DDOT's own systems flagged them as unreliable for network-level analysis. Their average severity matches HIGH records — dropping them would bias safety screening downstream.

    **Recovery target: NEARESTINTKEY.** Three candidate fields could serve as targets. Their completeness in recent records:

    | Field | Missing % | Verdict |
    |---|---|---|
    | `STREETSEGID` | 99.86% | Unusable in recent data |
    | `ROADWAYSEGID` | 99.86% | Unusable |
    | `NEARESTINTKEY` | 0.04% | The only viable first target |

    The near-total absence of `STREETSEGID` in recent records is itself a finding — likely an upstream schema or export change at DDOT, since these fields are populated in older data. Intersection-level recovery is also directly useful: hotspot rankings naturally aggregate at intersections.
    """)

with st.expander("3. Validation and method progression"):
    st.markdown("""
    **The synthetic-degradation idea.** MEDIUM records have no ground truth — by definition, they're the records where DDOT's match is unreliable. To measure pipeline accuracy honestly, the validation set is built from HIGH records (where the answer is trusted), with `NEARESTINTKEY` hidden. The pipeline tries to recover the hidden value; accuracy is measured against what was held aside.

    **The 500-record stratified sample.** Severity is highly imbalanced (~0.2% fatal). A random sample of 500 would contain ~1 fatal crash — useless for measuring pipeline behavior on the records that matter most. The sample is deliberately stratified:

    | Bucket | Target | Why |
    |---|---|---|
    | fatal | 30 | Oversample to measure performance on critical crashes |
    | major | 80 | Oversample |
    | minor | 200 | Representative |
    | pdo | 190 | Representative |

    Reproducibility: `random_state=42` throughout. Validation truth and inputs saved as separate files; pipeline code never touches the truth file.

    **Method progression.**

    - **V1 — Nearest official intersection (73.0%).** For each crash, snap to the closest intersection in DDOT's Intersection Points layer. Surprisingly low. Diagnostic: only 21 of 500 validation records were within 5m of any intersection. DC crash coordinates rarely land exactly on intersections — they sit mid-block. Pure distance can't tell which end of the block to assign.
    - **V2 — Top-5 candidates + address text (79.8%).** Find the 5 nearest intersections, score each by whether its street names appear in the crash address. Improvement, but plateaus: when the address only contains one street ("1200 PENNSYLVANIA AVENUE NW"), multiple candidates legitimately share that street.
    - **V3 — Top-5 + both street signals (90.4%).** Use both the crash's `ADDRESS` and `NEARESTINTSTREETNAME` (DDOT-recorded nearest cross-street). For each candidate, score how well its two streets match the two signals. Candidates that match both streets get a strong bonus; ties broken by distance.

    **What this validation does and doesn't measure.** The 90.4% accuracy is measured on HIGH-quality records with their NEARESTINTKEY artificially hidden — a controlled test where the pipeline tries to recover a known answer. This is a strong signal that the method is sound, but it is **not** the same as measuring performance on real MEDIUM records.

    Real MEDIUM records may be systematically harder than synthetically-degraded HIGH records. They are records where DDOT's own pipeline gave up — often because of mid-block coordinates, complex junctions, ramps, or unstructured addresses. A 90.4% recovery rate on the validation set is the **upper bound** of what to expect on production MEDIUM data; the real number is likely lower, possibly meaningfully so.

    The recovery pipeline has not yet been run on the 9,140 real MEDIUM records, and its predictions have not been compared against hand-verified ground truth. This is the most important next step — see *What comes next*.
    """)

with st.expander("4. What I learned"):
    st.markdown("""
    - **The data tells you the problem, not the other way around.** I started thinking DC had unmappable crashes like Madison. Exploration showed otherwise. The pivot — from "find lost crashes" to "fix mislabeled crashes" — was the most important decision in the project.

    - **The first metric is rarely the right metric.** V1's 73% looked bad until I realized the official-intersection snap and DDOT's NEARESTINTKEY assignment use different logic. The question wasn't "is my code broken" but "is my target what I think it is."

    - **Top-K + re-rank beats single-shot prediction.** The 96.4% top-5 coverage made the V2/V3 improvements possible. Recognizing this earlier would have saved time.

    - **Cross-field signals beat single-field signals.** `ADDRESS` alone plateaued at 79.8%. Adding `NEARESTINTSTREETNAME` jumped to 90.4%. The address tells you one street; the recorded nearest-street tells you the other. Together they uniquely identify the intersection.

    - **Validation design is harder than pipeline design.** Most of the project's hours went into deciding what "correct" means, what to stratify on, and how to make the metric defensible. The pipeline itself is a few hundred lines of code; the harness around it is what made the numbers trustworthy.
    """)

with st.expander("5. What comes next"):
    st.markdown("""
    - **Validate on real MEDIUM records (highest priority).** The current 90.4% accuracy is measured on synthetically-degraded HIGH records. The pipeline has not been tested on real MEDIUM data. The next step is to hand-verify a stratified sample (~50 records) by checking each crash's address against Google Maps to establish the correct intersection, then comparing against V3's prediction. This converts the inferred upper bound into a measured real-world number. Expect the number to drop — possibly to 70–80% — because MEDIUM records are systematically harder than randomly-hidden HIGH records.

    - **Confidence scoring.** Expose nearest-intersection distance and match-score as per-record confidence flags; route low-confidence predictions to manual review rather than treating all predictions as equally trustworthy.

    - **Segment-level recovery.** Apply the same architecture to `STREETSEGID` if DDOT republishes it, or recover segment assignments using road-network geometry from OpenStreetMap.

    - **Mid-block & ramp handling.** Crashes clearly not at intersections — highway ramps, mid-block driveways — should be handled by a separate segment-snap pipeline rather than forced to intersection-level matching.

    - **Hand-labeled validation on real MEDIUM records.** The synthetic-degradation test gives an upper bound; manually verifying ~50 MEDIUM crashes against ground truth would tell us how much the real number drops.
    """)