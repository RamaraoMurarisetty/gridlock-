"""
Event-Driven Congestion Forecaster -- Streamlit demo app.

Takes a new event's details from a form, runs the trained closure/duration
models plus the case-based recommender, and shows the impact forecast,
resource recommendation, and diversion plan.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy: push this folder to a public GitHub repo, then deploy on
Streamlit Community Cloud (share.streamlit.io) or Hugging Face Spaces
(Streamlit SDK) for a public shareable URL. See DEPLOY.md.
"""

import datetime as dt
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from event_congestion_model import CAT_COLS, FEATURE_COLS, NUM_COLS  # noqa: F401  (keeps unpickling happy)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="Event-Driven Congestion Forecaster",
    page_icon=":material/traffic:",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    closure_model = joblib.load(ARTIFACT_DIR / "closure_model.joblib")
    duration_model = joblib.load(ARTIFACT_DIR / "duration_model.joblib")
    try:
        recommender = joblib.load(ARTIFACT_DIR / "resource_recommender.joblib")
    except Exception as e:
        st.error(f"Recommender load failed: {e}")
        recommender = None
    return closure_model, duration_model, recommender


closure_model, duration_model, recommender = load_artifacts()
history = recommender.history

EVENT_TYPES = sorted(history["event_type"].unique())
EVENT_CAUSES = sorted(history["event_cause"].unique())
CORRIDORS = sorted(history["corridor"].unique())
ZONES = sorted(history["zone"].unique())
POLICE_STATIONS = sorted(history["police_station"].unique())
VEH_TYPES = sorted(history["veh_type"].unique())
AUTH_VALUES = sorted(history["authenticated"].unique())

st.title("Event-driven congestion: impact forecast & response plan")
st.caption(
    "Prototype for the Gridlock Hackathon. Predicts road-closure risk and "
    "clearance time for a new traffic-disrupting event, then recommends "
    "manpower, barricades, and a diversion plan."
)

with st.form("event_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        event_type = st.selectbox("Event type", EVENT_TYPES)
        event_cause = st.selectbox("Event cause", EVENT_CAUSES)
        corridor = st.selectbox("Corridor", CORRIDORS)
    with col2:
        zone = st.selectbox("Zone", ZONES)
        police_station = st.selectbox("Police station", POLICE_STATIONS)
        veh_type = st.selectbox("Vehicle type (if applicable)", VEH_TYPES)
    with col3:
        authenticated = st.selectbox("Authenticated", AUTH_VALUES)
        event_date = st.date_input("Event date", dt.date.today())
        event_time = st.time_input("Event time", dt.datetime.now().time())

    use_gps = st.checkbox("I have exact GPS coordinates for this event")
    lat = lon = None
    if use_gps:
        gc1, gc2 = st.columns(2)
        lat = gc1.number_input("Latitude", value=12.97, format="%.6f")
        lon = gc2.number_input("Longitude", value=77.59, format="%.6f")

    submitted = st.form_submit_button("Forecast impact & get recommendation")

if submitted:
    event_dt = dt.datetime.combine(event_date, event_time)
    hour = event_dt.hour
    day_of_week = event_dt.weekday()
    month = event_dt.month
    is_weekend = int(day_of_week >= 5)
    is_peak_hour = int(hour in [7, 8, 9, 10, 17, 18, 19, 20])

    event = {
        "event_type": event_type,
        "event_cause": event_cause,
        "corridor": corridor,
        "zone": zone,
        "police_station": police_station,
        "veh_type": veh_type,
        "authenticated": authenticated,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "age_of_truck": np.nan,
    }
    if lat is not None and lon is not None:
        event["latitude"] = lat
        event["longitude"] = lon

    rec = recommender.recommend(event, closure_model, duration_model)

    st.subheader("Impact forecast")
    m1, m2, m3 = st.columns(3)
    m1.metric("Road closure probability", f"{rec['predicted_road_closure_prob'] * 100:.1f}%")
    m2.metric("Expected clearance time", f"{rec['predicted_clearance_minutes']:.0f} min")
    m3.metric("Severity tier", rec["severity_tier"])

    st.subheader("Recommended response")
    rc1, rc2, rc3 = st.columns(3)
    rc1.write(f"**Personnel**\n\n{rec['recommended_personnel']}")
    rc2.write(f"**Barricades**\n\n{rec['recommended_barricades']}")
    rc3.write(f"**Action**\n\n{rec['recommended_action']}")

    st.subheader("Diversion plan")
    div = rec["diversion_plan"]
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.write("Alternate corridors:")
        alt = div.get("alternate_corridors")
        if isinstance(alt, list) and alt:
            st.table(pd.DataFrame(alt))
        else:
            st.write(alt)
    with dcol2:
        st.write("Candidate reroute junctions:")
        jn = div.get("candidate_reroute_junctions")
        if isinstance(jn, list) and jn:
            st.table(pd.DataFrame(jn))
        else:
            st.write("none found")
    st.caption(div.get("caveat", ""))

    st.subheader("Historical context")
    st.write(
        f"Based on {rec['similar_historical_events']} similar past events: "
        f"historical closure rate {rec['historical_closure_rate_for_similar_events'] * 100:.0f}%, "
        + (
            f"median duration {rec['historical_median_duration_minutes']:.0f} min, "
            if rec["historical_median_duration_minutes"] is not None
            else ""
        )
        + f"most commonly handled by {rec['most_common_handling_station']} station."
    )

    with st.expander("Raw model output"):
        st.json(rec)

st.divider()
st.caption(
    "Prototype notes: the duration model currently performs close to a naive "
    "median-by-cause baseline (see MODEL_REPORT.md). Resource tiers and the "
    "diversion plan are geometry/heuristic-driven, pending real deployment-"
    "outcome data and a road-network routing API (e.g. MapmyIndia) for "
    "production use."
)
