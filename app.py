import datetime as dt
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure model module path is in sys.path for joblib unpickling
sys.path.insert(0, str(Path(__file__).parent))
from event_congestion_model import CAT_COLS, FEATURE_COLS, NUM_COLS, ResourceRecommender  # noqa: F401

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Configure Page
st.set_page_config(
    page_title="Gridlock Forecaster",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom styles for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8 !important;
    }
    
    /* Customize form submit button */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        padding: 0.75rem 2.2rem !important;
        border-radius: 10px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        width: 100% !important;
        margin-top: 10px;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Form container styling */
    form[data-testid="stForm"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Alert style overrides */
    div[data-testid="stNotification"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Tab buttons hover */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        padding: 0.6rem 1.2rem !important;
    }
    
    /* Card Hover Glows */
    .custom-card {
        transition: all 0.3s ease;
    }
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    closure_model = joblib.load(ARTIFACT_DIR / "closure_model.joblib")
    duration_model = joblib.load(ARTIFACT_DIR / "duration_model.joblib")
    try:
        recommender = joblib.load(ARTIFACT_DIR / "resource_recommender.joblib")
    except Exception as e:
        st.error(f"Recommender load failed: {e}")
        recommender = None
    try:
        with open(ARTIFACT_DIR / "metrics.json", "r") as f:
            metrics_data = joblib.load(ARTIFACT_DIR / "metrics.json") if hasattr(joblib, "load") and False else json.load(f)
    except Exception:
        import json
        try:
            with open(ARTIFACT_DIR / "metrics.json", "r") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}
    return closure_model, duration_model, recommender, metrics_data


# Load models and recommender
closure_model, duration_model, recommender, metrics_data = load_artifacts()

if recommender is None:
    st.error("Failed to load historical recommendation model. Please verify artifacts folder contents.")
    st.stop()

history = recommender.history

# Safely extract unique categories for dropdowns
EVENT_TYPES = sorted([str(x) for x in history["event_type"].unique() if pd.notna(x)])
EVENT_CAUSES = sorted([str(x) for x in history["event_cause"].unique() if pd.notna(x)])
CORRIDORS = sorted([str(x) for x in history["corridor"].unique() if pd.notna(x)])
ZONES = sorted([str(x) for x in history["zone"].unique() if pd.notna(x)])
POLICE_STATIONS = sorted([str(x) for x in history["police_station"].unique() if pd.notna(x)])
VEH_TYPES = sorted([str(x) for x in history["veh_type"].unique() if pd.notna(x)])
AUTH_VALUES = sorted([str(x) for x in history["authenticated"].unique() if pd.notna(x)])

# ---------------------------------------------------------------------------
# Sidebar Settings and Presets
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <span style="font-size: 3rem;">🚦</span>
    <h2 style="margin: 0.5rem 0 0 0; color: #f8fafc;">Gridlock Control</h2>
    <p style="color: #64748b; font-size: 0.85rem;">Bengaluru Traffic Incident Center</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# Location Presets callback
def set_preset(lat, lon, name, corridor_preset):
    st.session_state.preset_lat = lat
    st.session_state.preset_lon = lon
    st.session_state.preset_use_gps = True
    st.session_state.preset_corridor = corridor_preset
    st.sidebar.success(f"Loaded: {name}")

st.sidebar.subheader("📍 Geospatial Presets")
st.sidebar.caption("Click a preset below to autofill the coordinate fields in the registry form.")

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("🚗 Silk Board", help="Load Silk Board coordinate values"):
        set_preset(12.9176, 77.6244, "Silk Board Junction", "Hosur Road")
    if st.button("✈️ Hebbal Flyover", help="Load Hebbal coordinate values"):
        set_preset(13.0358, 77.5978, "Hebbal Flyover", "Bellary Road 1")
with col_p2:
    if st.button("🏭 Tin Factory", help="Load Tin Factory coordinate values"):
        set_preset(12.9897, 77.6766, "Tin Factory (K.R. Pura)", "Hosur Road")
    if st.button("🔄 Reset Presets", type="secondary"):
        st.session_state.preset_use_gps = False
        st.session_state.pop("preset_lat", None)
        st.session_state.pop("preset_lon", None)
        st.session_state.pop("preset_corridor", None)
        st.sidebar.info("Presets cleared.")

st.sidebar.divider()

st.sidebar.subheader("⚙️ Severity Weights")
st.sidebar.caption("Adjust the importance of closure risk vs clearance duration in severity calculation.")
w_closure = st.sidebar.slider("Road Closure Probability Weight", 0.0, 1.0, 0.55, 0.05)
w_duration = 1.0 - w_closure
st.sidebar.info(f"⚖️ Current weights:\n- Closure Risk: {w_closure * 100:.0f}%\n- Clearance Time: {w_duration * 100:.0f}%")

# ---------------------------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e1b4b 0%, #2e1065 50%, #0f172a 100%); 
    padding: 2.2rem; 
    border-radius: 16px; 
    margin-bottom: 2rem; 
    border: 1px solid rgba(255,255,255,0.08); 
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
">
    <h1 style="color: #ffffff; margin: 0; font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px; display: flex; align-items: center; gap: 12px;">
        <span>🚦</span> Gridlock Forecaster & Dispatcher
    </h1>
    <p style="color: #cbd5e1; margin: 0.6rem 0 0 0; font-size: 1.05rem; font-weight: 400; opacity: 0.9; line-height: 1.5;">
        Real-time clearance-time modeling and deployment coordinator for the Bengaluru Traffic incident response network.
    </p>
</div>
""", unsafe_allow_html=True)

tab_forecaster, tab_historical, tab_diagnostics = st.tabs([
    "🎯 Live Forecaster & Response Plan", 
    "🕒 Historical Case Analogs", 
    "📈 Model Calibration & Diagnostics"
])

# Helpers for Cards & Color-coding
def get_severity_details(tier):
    if tier == "Low": 
        return "#10b981", "🟢 Low Severity - Normal clearance procedures apply."
    if tier == "Moderate": 
        return "#eab308", "🟡 Moderate Severity - Actively monitor surrounding intersections."
    if tier == "High": 
        return "#f97316", "🟠 High Severity - Multi-point coordination required immediately."
    return "#ef4444", "🔴 Critical Severity - Peak congestion risk. Emergency units deployed."

def make_card(title, icon, value, details, border_color="#6366f1"):
    return f"""
    <div class="custom-card" style="
        background-color: #111827; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid {border_color}; 
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 1rem;
    ">
        <div>
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.2rem;">{icon}</span> {title}
            </div>
            <div style="color: #ffffff; font-size: 1.25rem; font-weight: 700; line-height: 1.3; margin: 0.75rem 0 0.5rem 0;">
                {value}
            </div>
        </div>
        <div style="color: #64748b; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.5rem; margin-top: 0.5rem;">
            {details}
        </div>
    </div>
    """

# ---------------------------------------------------------------------------
# Tab 1: Live Forecaster & Response Plan
# ---------------------------------------------------------------------------
with tab_forecaster:
    col_form, col_results = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown("<h3 style='margin-top:0; color:#f8fafc;'>📝 Register Incident</h3>", unsafe_allow_html=True)
        
        with st.form("event_form"):
            st.subheader("Event Parameters")
            
            # Setup presets or default values
            default_corridor = st.session_state.get("preset_corridor", CORRIDORS[0] if CORRIDORS else "Unknown")
            default_corridor_index = CORRIDORS.index(default_corridor) if default_corridor in CORRIDORS else 0
            
            event_type = st.selectbox("Event Type", EVENT_TYPES)
            event_cause = st.selectbox("Event Cause", EVENT_CAUSES)
            corridor = st.selectbox("Corridor", CORRIDORS, index=default_corridor_index)
            zone = st.selectbox("Zone", ZONES)
            police_station = st.selectbox("Handling Police Station", POLICE_STATIONS)
            veh_type = st.selectbox("Vehicle Type (if applicable)", VEH_TYPES)
            authenticated = st.selectbox("Authenticated Source", AUTH_VALUES)
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.subheader("Schedule & Timing")
            
            c_date, c_time = st.columns(2)
            event_date = c_date.date_input("Event Date", dt.date.today())
            event_time = c_time.time_input("Event Time", dt.datetime.now().time())

            st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            preset_use_gps = st.session_state.get("preset_use_gps", False)
            use_gps = st.checkbox("Exact GPS coordinates are available", value=preset_use_gps)
            
            lat = lon = None
            if use_gps:
                default_lat = st.session_state.get("preset_lat", 12.9788)
                default_lon = st.session_state.get("preset_lon", 77.5913)
                gc1, gc2 = st.columns(2)
                lat = gc1.number_input("Latitude", value=default_lat, format="%.6f", min_value=12.0, max_value=14.0)
                lon = gc2.number_input("Longitude", value=default_lon, format="%.6f", min_value=77.0, max_value=79.0)

            submitted = st.form_submit_button("Forecast Impact & Plan Response")

    with col_results:
        st.markdown("<h3 style='margin-top:0; color:#f8fafc;'>📊 Prediction & Operations Center</h3>", unsafe_allow_html=True)
        
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
            
            # Geolocation setup
            actual_lat, actual_lon = lat, lon
            if (actual_lat is None or actual_lon is None) and corridor in recommender.corridor_centroids["corridor"].values:
                row = recommender.corridor_centroids.loc[recommender.corridor_centroids["corridor"] == corridor].iloc[0]
                actual_lat, actual_lon = row["latitude"], row["longitude"]
                
            if actual_lat is not None and actual_lon is not None:
                event["latitude"] = actual_lat
                event["longitude"] = actual_lon

            # Retrieve prediction recommendations
            with st.spinner("Generating congestion models..."):
                rec = recommender.recommend(event, closure_model, duration_model)

            # Recalculate severity score dynamically using sidebar weights
            p_closure = rec["predicted_road_closure_prob"]
            dur_pred = rec["predicted_clearance_minutes"]
            dur_norm = min(dur_pred / 180.0, 1.0)
            custom_severity_score = w_closure * p_closure + w_duration * dur_norm
            
            # Reclassify tier based on custom score
            custom_tier = next(t for t in recommender.SEVERITY_TIERS if custom_severity_score <= t[0])
            severity_label = custom_tier[1]
            rec_personnel = custom_tier[2]
            rec_barricades = custom_tier[3]
            rec_action = custom_tier[4]

            sev_color, sev_text = get_severity_details(severity_label)
            
            # Dynamic Banner Display
            st.markdown(f"""
            <div style="
                padding: 1.5rem; 
                background: linear-gradient(135deg, {sev_color}18 0%, rgba(17, 24, 39, 0.95) 100%); 
                border: 1px solid {sev_color}44; 
                border-left: 6px solid {sev_color}; 
                border-radius: 12px; 
                margin-bottom: 2rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            ">
                <div>
                    <span style="font-weight:700; color:{sev_color}; font-size:0.9rem; text-transform: uppercase; letter-spacing: 1px;">Classification Classification</span>
                    <h3 style="color:#ffffff; margin: 0.2rem 0; font-size:1.8rem; font-weight:700;">{severity_label.upper()} Severity Impact</h3>
                    <span style="color:#cbd5e1; font-size:0.95rem; line-height: 1.4;">{sev_text}</span>
                </div>
                <div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.08); padding-left: 1.5rem;">
                    <span style="color:#94a3b8; font-size:0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Custom Score</span>
                    <h2 style="color:#ffffff; margin: 0; font-size: 2.2rem; font-weight:700;">{custom_severity_score:.2f}</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Primary metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Closure Probability", f"{p_closure * 100:.1f}%")
            m2.metric("Clearance Expectation", f"{dur_pred:.0f} min")
            m3.metric("Composite Score (Orig)", f"{rec['severity_score']:.2f}")

            st.markdown("<h4 style='color:#f8fafc; margin-top: 1.5rem;'>📋 Resource Deployment Details</h4>", unsafe_allow_html=True)
            
            # Deployment response cards
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(make_card(
                    "Personnel Required", "👮", rec_personnel, 
                    "Deploy designated team immediately. Instructors oversee critical points.", 
                    sev_color
                ), unsafe_allow_html=True)
            with rc2:
                st.markdown(make_card(
                    "Barricades Required", "🚧", rec_barricades, 
                    "Secure perimeter. Use high-visibility barriers at diversion junctions.", 
                    sev_color
                ), unsafe_allow_html=True)
            with rc3:
                st.markdown(make_card(
                    "Operational Action", "⚡", rec_action, 
                    "Alert nearby control rooms. Divert non-essential traffic.", 
                    sev_color
                ), unsafe_allow_html=True)

            st.markdown("<h4 style='color:#f8fafc; margin-top: 1.5rem;'>🗺️ Incident Diversion Map & Plan</h4>", unsafe_allow_html=True)
            
            div = rec["diversion_plan"]
            
            # Map visualization if coordinates are present
            if actual_lat is not None and actual_lon is not None:
                map_data = []
                map_data.append({
                    "latitude": actual_lat,
                    "longitude": actual_lon,
                    "name": "Incident Location",
                    "color": "#ef4444" # Red
                })
                
                # Extract alternate corridors
                alt_corrs = div.get("alternate_corridors")
                if isinstance(alt_corrs, list):
                    for item in alt_corrs:
                        c_name = item["corridor"]
                        c_row = recommender.corridor_centroids[recommender.corridor_centroids["corridor"] == c_name]
                        if not c_row.empty:
                            map_data.append({
                                "latitude": c_row.iloc[0]["latitude"],
                                "longitude": c_row.iloc[0]["longitude"],
                                "name": f"Alternate Corridor: {c_name}",
                                "color": "#3b82f6" # Blue
                            })
                            
                # Extract junctions
                cand_jns = div.get("candidate_reroute_junctions")
                if isinstance(cand_jns, list):
                    for item in cand_jns:
                        j_name = item["junction"]
                        j_row = recommender.junction_centroids[recommender.junction_centroids["junction"] == j_name]
                        if not j_row.empty:
                            map_data.append({
                                "latitude": j_row.iloc[0]["latitude"],
                                "longitude": j_row.iloc[0]["longitude"],
                                "name": f"Reroute Junction: {j_name}",
                                "color": "#10b981" # Green
                            })
                
                df_map = pd.DataFrame(map_data)
                # Mapbox standard plot
                st.map(df_map, latitude="latitude", longitude="longitude", size=50)
                st.caption("🔴 Red = Incident | 🔵 Blue = Alternate Corridors | 🟢 Green = Reroute Junctions")
            else:
                st.info("No geospatial data available for map rendering.")

            # Tables for rerouting details
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown("**Alternate Corridors**")
                alt = div.get("alternate_corridors")
                if isinstance(alt, list) and alt:
                    st.dataframe(pd.DataFrame(alt), width="stretch", hide_index=True)
                else:
                    st.write(alt)
            with dcol2:
                st.markdown("**Candidate Reroute Junctions**")
                jn = div.get("candidate_reroute_junctions")
                if isinstance(jn, list) and jn:
                    st.dataframe(pd.DataFrame(jn), width="stretch", hide_index=True)
                else:
                    st.write("No candidate reroute junctions found")
            
            if div.get("caveat"):
                st.caption(f"ℹ️ {div.get('caveat')}")

            # Safe storage of session state for tab 2
            st.session_state.last_rec = rec
            st.session_state.last_neighbors = recommender.history.iloc[
                recommender.knn.kneighbors(
                    recommender._vectorize(
                        pd.DataFrame([{
                            "event_cause": event.get("event_cause", "Unknown"),
                            "corridor": event.get("corridor", "Unknown"),
                            "zone": event.get("zone", "Unknown"),
                            "hour": event.get("hour", 12),
                            "day_of_week": event.get("day_of_week", 0),
                            "is_weekend": event.get("is_weekend", 0),
                        }])
                    )
                )[1][0]
            ]
        else:
            if "last_rec" in st.session_state:
                st.info("Results of the last ran forecast are cached. Click submit on the form to run a new query.")
            else:
                # Default placeholder when no event has been submitted
                st.markdown("""
                <div style="
                    display: flex; 
                    flex-direction: column; 
                    align-items: center; 
                    justify-content: center; 
                    min-height: 400px; 
                    background-color: #111827; 
                    border: 1px dashed rgba(255, 255, 255, 0.1); 
                    border-radius: 16px;
                    padding: 2rem;
                    text-align: center;
                ">
                    <span style="font-size: 4rem; margin-bottom: 1rem; filter: grayscale(0.2);">📡</span>
                    <h4 style="color: #cbd5e1; margin-bottom: 0.5rem;">Waiting for Incident Parameters</h4>
                    <p style="color: #64748b; max-width: 400px; font-size: 0.95rem;">
                        Fill out the incident registry form on the left side and press "Forecast Impact" to run the HGBRT predictive models and view the dispatch plan.
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: Historical Case Analogs
# ---------------------------------------------------------------------------
with tab_historical:
    st.markdown("<h3 style='margin-top:0; color:#f8fafc;'>🕒 Case-Based Analog Finder</h3>", unsafe_allow_html=True)
    st.markdown("Retrieves the nearest historical matching incidents from the model history database based on cause, corridor, zone, and time of day.")

    if "last_neighbors" in st.session_state and "last_rec" in st.session_state:
        neighbors = st.session_state.last_neighbors
        rec = st.session_state.last_rec
        
        # Summary metrics
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Analogs Found", len(neighbors))
        sc2.metric("Hist. Closure Rate", f"{rec['historical_closure_rate_for_similar_events'] * 100:.0f}%")
        sc3.metric("Hist. Median Duration", f"{rec['historical_median_duration_minutes']:.0f} min" if rec['historical_median_duration_minutes'] else "N/A")
        
        st.subheader("Matching Cases Records")
        display_cols = [
            "event_type", "event_cause", "corridor", "zone", 
            "police_station", "veh_type", "authenticated", "duration_minutes", "closure_label"
        ]
        readable_df = neighbors[display_cols].copy()
        readable_df.columns = [
            "Type", "Cause", "Corridor", "Zone", "Station", "Vehicle", "Auth", "Duration (Min)", "Closure needed"
        ]
        st.dataframe(readable_df, width="stretch", hide_index=True)
        
        # Charts section
        st.subheader("Data Analytics on Matching Cases")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Incident Cause Distribution**")
            cause_counts = neighbors["event_cause"].value_counts()
            st.bar_chart(cause_counts, color="#818cf8")
        with c2:
            st.markdown("**Incident Clearance Time Breakdown (minutes)**")
            durations = neighbors["duration_minutes"].dropna()
            if not durations.empty:
                st.bar_chart(durations, color="#c084fc")
            else:
                st.write("No duration data available for matching cases.")
    else:
        st.markdown("""
        <div style="
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            min-height: 250px; 
            background-color: #111827; 
            border: 1px dashed rgba(255, 255, 255, 0.1); 
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        ">
            <span style="font-size: 2.5rem; margin-bottom: 0.5rem; filter: grayscale(0.5);">📋</span>
            <h5 style="color: #cbd5e1; margin-bottom: 0.5rem;">No active forecast loaded</h5>
            <p style="color: #64748b; max-width: 350px; font-size: 0.9rem;">
                Run a forecast query first under the "Live Forecaster" tab to inspect matching historical cases.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 3: Model Diagnostics
# ---------------------------------------------------------------------------
with tab_diagnostics:
    st.markdown("<h3 style='margin-top:0; color:#f8fafc;'>📈 Predictive Engine Performance</h3>", unsafe_allow_html=True)
    st.markdown("These statistics show validation and accuracy metrics computed from the time-split model training (80/20 train/test split).")

    if metrics_data:
        def make_metric_card(title, metrics, desc, color="#818cf8"):
            metrics_html = ""
            for k, v in metrics.items():
                metrics_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.9rem; color:#cbd5e1;'><strong>{k}</strong> <span>{v}</span></div>"
            
            return f"""
            <div class="custom-card" style="
                background-color: #111827; 
                padding: 1.5rem; 
                border-radius: 12px; 
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-top: 5px solid {color}; 
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
                min-height: 280px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                margin-bottom: 1rem;
            ">
                <div>
                    <h4 style="color:#ffffff; margin: 0 0 1rem 0; font-size:1.1rem; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem;">{title}</h4>
                    <div>{metrics_html}</div>
                </div>
                <div style="color: #64748b; font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.5rem; margin-top: 1rem;">
                    {desc}
                </div>
            </div>
            """

        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            p_metrics = metrics_data.get("priority_model", {})
            st.markdown(make_metric_card(
                "Priority Classifier", 
                {
                    "Accuracy": f"{p_metrics.get('accuracy', 0) * 100:.1f}%",
                    "F1 Score": f"{p_metrics.get('f1', 0):.3f}",
                    "ROC AUC": f"{p_metrics.get('roc_auc', 0):.3f}",
                    "Train Records": p_metrics.get('n_train', 0),
                    "Test Records": p_metrics.get('n_test', 0)
                },
                "Classifies incident as high or low severity. Heavily determined by administrative corridor assignments in this dataset.",
                "#10b981"
            ), unsafe_allow_html=True)
            
        with m_col2:
            c_metrics = metrics_data.get("closure_model", {})
            st.markdown(make_metric_card(
                "Road Closure Classifier", 
                {
                    "Accuracy": f"{c_metrics.get('accuracy', 0) * 100:.1f}%",
                    "F1 Score": f"{c_metrics.get('f1', 0):.3f}",
                    "ROC AUC": f"{c_metrics.get('roc_auc', 0):.3f}",
                    "Base Closure Rate": f"{c_metrics.get('base_rate', 0) * 100:.1f}%",
                    "Test Records": c_metrics.get('n_test', 0)
                },
                "Predicts whether the event will require a full road closure. Shows high accuracy on imbalanced classes.",
                "#6366f1"
            ), unsafe_allow_html=True)
            
        with m_col3:
            d_metrics = metrics_data.get("duration_model", {})
            st.markdown(make_metric_card(
                "Clearance Duration Regressor", 
                {
                    "MAE (Minutes)": f"{d_metrics.get('MAE_minutes', 0)} min",
                    "RMSE (Minutes)": f"{d_metrics.get('RMSE_minutes', 0)} min",
                    "R2 Log-Scale": f"{d_metrics.get('R2_log_scale', 0):.3f}",
                    "Median Actual": f"{d_metrics.get('median_actual_minutes', 0)} min",
                    "Test Records": d_metrics.get('n_test', 0)
                },
                "Predicts expected clearance time. High variance indicates a need for supplementary data streams (e.g. weather, tow-truck proximity).",
                "#c084fc"
            ), unsafe_allow_html=True)
            
        st.markdown("""
        > [!NOTE]
        > **Model Performance Insight**:
        > The Duration model's R2 is currently close to zero (or slightly negative), meaning it performs similarly to predicting the historical median duration.
        > To improve this performance in future iterations:
        > - Integrate real-time weather logs (rainfall is a major delay factor in Bengaluru).
        > - Extract towing and emergency crane service location history.
        > - Incorporate active vehicle counts or speed data at the start of the incident.
        """)
    else:
        st.info("Metrics data file `metrics.json` could not be found.")

st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
st.caption(
    "Prototype notes: The clearance-time predictive regressor operates near naive baseline performance "
    "(refer to MODEL_REPORT.md). Resource scaling and rerouting strategies are heuristic-driven geometry bounds "
    "subject to domain expert feedback and road network routing API integration (e.g., MapmyIndia) for production deploy."
)
