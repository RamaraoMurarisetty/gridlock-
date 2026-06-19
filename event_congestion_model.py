"""
Event-Driven Congestion: Impact Forecasting + Resource Recommendation
Prototype model for Bengaluru traffic-incident data.

Pipeline
--------
1. Feature engineering from raw incident log (no leakage between targets).
2. Three HistGradientBoosting models (impact forecasting):
     a. priority_model      -> predicts High/Low severity
     b. closure_model       -> predicts whether road closure will be needed
     c. duration_model      -> predicts expected clearance time (minutes)
3. Case-based recommendation engine:
     Given a new event, retrieves the k most similar historical events
     and turns the Stage-1 predictions + historical analogs into a
     manpower / barricade / diversion recommendation.

Run:
    python3 event_congestion_model.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "/mnt/project/Astram_event_data_anonymized__Astram_event_data_anonymizedb40ac87.csv"
OUT_DIR = Path("/home/claude/work/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["event_type", "event_cause", "corridor", "zone", "police_station",
            "veh_type", "authenticated"]
NUM_COLS = ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour", "age_of_truck"]
FEATURE_COLS = CAT_COLS + NUM_COLS


# ---------------------------------------------------------------------------
# 1. Load + feature engineering
# ---------------------------------------------------------------------------
def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in ["start_datetime", "closed_datetime", "end_datetime", "resolved_datetime"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    # normalize messy categorical text (e.g. "Debris" vs "debris")
    df["event_cause"] = df["event_cause"].str.strip().str.lower()
    df["event_cause"] = df["event_cause"].replace({"fog / low visibility": "low_visibility"})

    for c in CAT_COLS:
        df[c] = df[c].fillna("Unknown").astype(str)

    df["hour"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek
    df["month"] = df["start_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)

    # duration target: best available end-of-incident timestamp minus start
    dur_min = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
    # sanity bounds: drop negative timestamps and clearly bad entries (>3 days)
    dur_min = dur_min.where((dur_min >= 1) & (dur_min <= 3 * 24 * 60))
    df["duration_minutes"] = dur_min

    df["priority_label"] = df["priority"].map({"High": 1, "Low": 0})
    df["closure_label"] = df["requires_road_closure"].astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Stage 1: impact forecasting models
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, frac=0.8):
    df_sorted = df.sort_values("start_datetime")
    cut = int(len(df_sorted) * frac)
    return df_sorted.iloc[:cut], df_sorted.iloc[cut:]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def train_classifier(df: pd.DataFrame, label_col: str, name: str):
    sub = df.dropna(subset=[label_col])
    train, test = time_split(sub)
    X_train, X_test = encode_features(train), encode_features(test)
    y_train, y_test = train[label_col], test[label_col]

    model = HistGradientBoostingClassifier(
        categorical_features=[c in CAT_COLS for c in FEATURE_COLS],
        max_depth=6, learning_rate=0.08, max_iter=200, random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "base_rate": round(y_train.mean(), 3),
    }
    print(f"[{name}] {metrics}")
    return model, metrics


def train_duration_regressor(df: pd.DataFrame):
    sub = df.dropna(subset=["duration_minutes"])
    train, test = time_split(sub)
    X_train, X_test = encode_features(train), encode_features(test)
    y_train = np.log1p(train["duration_minutes"])
    y_test = np.log1p(test["duration_minutes"])

    model = HistGradientBoostingRegressor(
        categorical_features=[c in CAT_COLS for c in FEATURE_COLS],
        max_depth=6, learning_rate=0.08, max_iter=200, random_state=42,
    )
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    pred_min = np.expm1(pred_log)
    true_min = np.expm1(y_test)
    metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "MAE_minutes": round(mean_absolute_error(true_min, pred_min), 1),
        "RMSE_minutes": round(mean_squared_error(true_min, pred_min) ** 0.5, 1),
        "R2_log_scale": round(r2_score(y_test, pred_log), 3),
        "median_actual_minutes": round(true_min.median(), 1),
    }
    print(f"[duration_model] {metrics}")
    return model, metrics


# ---------------------------------------------------------------------------
# Geographic helper for diversion suggestions
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# 3. Stage 2: case-based resource recommendation
# ---------------------------------------------------------------------------
class ResourceRecommender:
    """
    Retrieves the k most similar historical events and turns them +
    Stage-1 predictions into a manpower / barricade / diversion suggestion.

    NOTE: the dataset has no ground-truth manpower/barricade counts, AND no
    ground-truth diversion routes either -- it records *where* an incident
    is (lat/lon, address, sometimes a polyline of the blocked stretch) but
    never what alternate route traffic was actually sent down. So both the
    resource tiers below and the diversion suggestions are geometry-driven
    heuristics calibrated against historical corridor/junction proximity,
    not learned from labeled outcomes. They are meant to be reviewed by
    traffic-ops domain experts and tightened with real data once available
    (see MODEL_REPORT.md, sections on data gaps).
    """

    SEVERITY_TIERS = [
        # (max_score, label, personnel, barricades, action)
        (0.25, "Low",      "2 traffic constables",            "2-4",  "Local signage + on-spot diversion if needed"),
        (0.50, "Moderate", "3-4 constables + 1 head constable","4-8",  "Single-lane diversion, alert nearby junctions"),
        (0.75, "High",     "1 inspector + 6-8 constables",     "8-15", "Multi-point diversion plan, notify control room"),
        (1.01, "Critical", "Sub-inspector + 10+ constables, QRT on standby", "15+", "Full corridor diversion, pre-position recovery vehicle"),
    ]

    def __init__(self, history_df: pd.DataFrame, encoder: OneHotEncoder,
                 scaler: StandardScaler, k: int = 8):
        self.history = history_df.reset_index(drop=True)
        self.encoder = encoder
        self.scaler = scaler
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k)
        self.knn.fit(self._vectorize(self.history))

        # geographic centroids for diversion suggestions
        named = self.history[~self.history["corridor"].isin(["Non-corridor", "Unknown"])]
        self.corridor_centroids = (
            named.groupby("corridor")[["latitude", "longitude"]].mean().reset_index()
        )
        self.junction_centroids = (
            self.history[self.history["junction"].notna()]
            .groupby("junction")[["latitude", "longitude"]].mean().reset_index()
        )

    def _vectorize(self, df: pd.DataFrame) -> np.ndarray:
        cat = self.encoder.transform(df[["event_cause", "corridor", "zone"]])
        num = self.scaler.transform(df[["hour", "day_of_week", "is_weekend"]])
        return np.hstack([np.asarray(cat), np.asarray(num)])

    def _suggest_diversion(self, lat: float, lon: float, own_corridor: str,
                            top_n: int = 2, radius_km: float = 6.0) -> dict:
        """
        Geometry-only diversion heuristic: nearest named corridors and
        junctions to the incident location, excluding the blocked corridor
        itself. This is NOT shortest-path routing -- there is no road-
        network graph in this dataset, only point locations -- so treat it
        as 'which alternates are physically nearby', to be confirmed against
        an actual road graph (e.g. MapmyIndia, a stated hackathon data
        partner) before real deployment.
        """
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return {"note": "no coordinates supplied; cannot suggest diversion"}

        corr = self.corridor_centroids.copy()
        corr = corr[corr["corridor"] != own_corridor]
        corr["distance_km"] = haversine_km(lat, lon, corr["latitude"], corr["longitude"])
        corr = corr[corr["distance_km"] <= radius_km].sort_values("distance_km").head(top_n)

        jn = self.junction_centroids.copy()
        jn["distance_km"] = haversine_km(lat, lon, jn["latitude"], jn["longitude"])
        jn = jn.sort_values("distance_km").head(top_n)

        return {
            "alternate_corridors": [
                {"corridor": r["corridor"], "distance_km": round(r["distance_km"], 2)}
                for _, r in corr.iterrows()
            ] or "no named corridor within radius; treat as local-roads diversion only",
            "candidate_reroute_junctions": [
                {"junction": r["junction"], "distance_km": round(r["distance_km"], 2)}
                for _, r in jn.iterrows()
            ],
            "caveat": "geographic proximity heuristic, not shortest-path road routing",
        }

    def recommend(self, event: dict, closure_model, duration_model) -> dict:
        ev = pd.DataFrame([event])
        for c in CAT_COLS:
            ev[c] = ev.get(c, "Unknown")
            ev[c] = ev[c].fillna("Unknown").astype("category")
        for c in NUM_COLS:
            if c not in ev.columns:
                ev[c] = np.nan
        X = ev[FEATURE_COLS]

        p_closure = closure_model.predict_proba(X)[0, 1]
        dur_pred = float(np.expm1(duration_model.predict(X)[0]))

        # composite severity score: priority field deliberately excluded —
        # it is ~perfectly determined by `corridor` in this dataset (see
        # diagnostic printed at startup), so it carries no independent
        # severity signal and would just double-count corridor.
        dur_norm = min(dur_pred / 180.0, 1.0)  # 3hr+ treated as max norm
        severity_score = 0.55 * p_closure + 0.45 * dur_norm

        tier = next(t for t in self.SEVERITY_TIERS if severity_score <= t[0])

        ev_for_knn = pd.DataFrame([{
            "event_cause": event.get("event_cause", "Unknown"),
            "corridor": event.get("corridor", "Unknown"),
            "zone": event.get("zone", "Unknown"),
            "hour": event.get("hour", 12),
            "day_of_week": event.get("day_of_week", 0),
            "is_weekend": event.get("is_weekend", 0),
        }])
        dist, idx = self.knn.kneighbors(self._vectorize(ev_for_knn))
        neighbors = self.history.iloc[idx[0]]

        own_corridor = event.get("corridor", "Unknown")
        lat, lon = event.get("latitude"), event.get("longitude")
        if (lat is None or lon is None) and own_corridor in self.corridor_centroids["corridor"].values:
            row = self.corridor_centroids.loc[self.corridor_centroids["corridor"] == own_corridor].iloc[0]
            lat, lon = row["latitude"], row["longitude"]
        diversion = self._suggest_diversion(lat, lon, own_corridor)

        return {
            "predicted_road_closure_prob": round(float(p_closure), 3),
            "predicted_clearance_minutes": round(dur_pred, 1),
            "severity_score": round(float(severity_score), 3),
            "severity_tier": tier[1],
            "recommended_personnel": tier[2],
            "recommended_barricades": tier[3],
            "recommended_action": tier[4],
            "similar_historical_events": int(len(neighbors)),
            "historical_closure_rate_for_similar_events": round(neighbors["closure_label"].mean(), 2),
            "historical_median_duration_minutes": round(
                neighbors["duration_minutes"].median(), 1
            ) if neighbors["duration_minutes"].notna().any() else None,
            "most_common_handling_station": neighbors["police_station"].mode().iloc[0]
            if not neighbors["police_station"].mode().empty else None,
            "diversion_plan": diversion,
        }


def build_recommender(df: pd.DataFrame) -> ResourceRecommender:
    clean = df.dropna(subset=["hour", "day_of_week"])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(clean[["event_cause", "corridor", "zone"]])
    scaler = StandardScaler()
    scaler.fit(clean[["hour", "day_of_week", "is_weekend"]])
    return ResourceRecommender(clean, encoder, scaler, k=8)


# ---------------------------------------------------------------------------
# 4. Run end to end
# ---------------------------------------------------------------------------
def main():
    df = load_and_engineer(DATA_PATH)
    print(f"Loaded {len(df)} events, date range "
          f"{df['start_datetime'].min().date()} -> {df['start_datetime'].max().date()}\n")

    priority_model, priority_metrics = train_classifier(df, "priority_label", "priority_model")
    closure_model, closure_metrics = train_classifier(df, "closure_label", "closure_model")
    duration_model, duration_metrics = train_duration_regressor(df)

    # --- diagnostic: is `priority` independent signal, or a corridor lookup? ---
    corr_priority = df.groupby("corridor")["priority_label"].mean()
    near_deterministic = ((corr_priority < 0.02) | (corr_priority > 0.98)).mean()
    print(f"\n[diagnostic] {near_deterministic:.0%} of corridors have priority "
          f"label >98% pure (all-High or all-Low) -> `priority` is treated as "
          f"administrative/corridor-derived, NOT used in severity scoring below.")

    # --- diagnostic: does the duration model beat a naive median-by-cause guess? ---
    dur_sub = df.dropna(subset=["duration_minutes"])
    dur_train, dur_test = time_split(dur_sub)
    cause_medians = dur_train.groupby("event_cause")["duration_minutes"].median()
    naive_pred = dur_test["event_cause"].map(cause_medians).fillna(dur_train["duration_minutes"].median())
    naive_mae = mean_absolute_error(dur_test["duration_minutes"], naive_pred)
    print(f"[diagnostic] duration_model MAE={duration_metrics['MAE_minutes']} min vs "
          f"naive median-by-cause baseline MAE={naive_mae:.1f} min -> current features "
          f"add {'real' if duration_metrics['MAE_minutes'] < naive_mae * 0.9 else 'little-to-no'} "
          f"lift over the naive baseline.\n")

    recommender = build_recommender(df)

    joblib.dump(priority_model, OUT_DIR / "priority_model.joblib")
    joblib.dump(closure_model, OUT_DIR / "closure_model.joblib")
    joblib.dump(duration_model, OUT_DIR / "duration_model.joblib")
    joblib.dump(recommender, OUT_DIR / "resource_recommender.joblib")

    metrics = {
        "priority_model": priority_metrics,
        "closure_model": closure_metrics,
        "duration_model": duration_metrics,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- demo: a few sample new-event scenarios ----
    demo_events = [
        {"event_type": "planned", "event_cause": "procession", "corridor": "Mysore Road",
         "zone": "West Zone 1", "police_station": "Yeshwanthpura", "veh_type": "Unknown",
         "authenticated": "yes", "hour": 18, "day_of_week": 5, "month": 3,
         "is_weekend": 1, "is_peak_hour": 1, "age_of_truck": np.nan},
        {"event_type": "unplanned", "event_cause": "accident", "corridor": "Bellary Road 1",
         "zone": "North Zone 1", "police_station": "Hennuru", "veh_type": "heavy_vehicle",
         "authenticated": "yes", "hour": 9, "day_of_week": 1, "month": 2,
         "is_weekend": 0, "is_peak_hour": 1, "age_of_truck": np.nan},
        {"event_type": "unplanned", "event_cause": "vehicle_breakdown", "corridor": "Hosur Road",
         "zone": "South Zone 1", "police_station": "K.R. Pura", "veh_type": "truck",
         "authenticated": "yes", "hour": 14, "day_of_week": 2, "month": 1,
         "is_weekend": 0, "is_peak_hour": 0, "age_of_truck": 8.0},
    ]

    print("\n--- Sample recommendations ---")
    demo_results = []
    for event in demo_events:
        rec = recommender.recommend(event, closure_model, duration_model)
        demo_results.append({"input_event": event, "recommendation": rec})
        print(json.dumps({"cause": event["event_cause"], "corridor": event["corridor"],
                           **rec}, indent=2))

    with open(OUT_DIR / "demo_recommendations.json", "w") as f:
        json.dump(demo_results, f, indent=2, default=str)

    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
