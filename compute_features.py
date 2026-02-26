import os
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN

APPLICANT_EXCEL = os.path.join(os.path.dirname(__file__), "utils", "Clustering Analysis - Base Table.xlsx")

SANITIZATION_RULES = {
    'VEHICLE_SPEED': {'min': 0.0, 'max': 60.0},
    'BATTERY_VOLTAGE': {'min': 40.0, 'max': 100.0},
    'BATTERY_TEMPERATURE': {'min': -10.0, 'max': 60.0},
    'MOTOR_TEMPERATURE': {'min': -20.0, 'max': 140.0},
    'CONTROLLER_TEMPERATURE': {'min': -40.0, 'max': 65.0},
    'BATTERY_CURRENT': {'min': -200.0, 'max': 200.0},
    'SOC': {'min': 0.0, 'max': 100.0},
    'SOH': {'min': 60.0, 'max': 100.0},
    'REMAINING_CAPACITY': {'min': 0.0, 'max': 1900.0},
    'DISTANCE_TO_EMPTY': {'min': 0.0, 'max': 120.0},
    'ACCELERATION': {'min': -3.0, 'max': 3.0},
    'ODOMETER': {'min': 0.0, 'max': 250000.0}
}


def log(msg, elapsed=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if elapsed is not None:
        print(f"[{ts}] {msg} (took {elapsed:.2f}s)")
    else:
        print(f"[{ts}] {msg}")


def log_df_size(name, df):
    if df is None or not isinstance(df, pd.DataFrame):
        return
    rows, cols = df.shape
    log(f"df size {name}: {rows:,} rows x {cols} cols")


def compute_stats(series, stats_list):
    s = series.dropna()
    res = {}

    percentiles = {
        "p1": 0.01, "p5": 0.05, "p10": 0.10, "p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99
    }

    if "latest" in stats_list:
        res["latest"] = s.iloc[-1] if len(s) else None

    if "avg" in stats_list: res["avg"] = s.mean()
    if "min" in stats_list: res["min"] = s.min()
    if "max" in stats_list: res["max"] = s.max()
    if "var" in stats_list: res["var"] = s.var()

    for k, p in percentiles.items():
        if k in stats_list:
            res[k] = s.quantile(p)

    return res


def count_threshold_crossings(series, threshold):
    s = series.dropna()
    above = s > threshold
    crossings = (above & ~above.shift(fill_value=False)).sum()
    return crossings


def count_threshold_crossings_below(series, threshold):
    s = series.dropna()
    below = s < threshold
    crossings = (below & ~below.shift(fill_value=False)).sum()
    return crossings


def duration_above_threshold(subdf, col, threshold):
    df = subdf.copy()
    df["delta_t"] = df["EVENT_AT"].diff().dt.total_seconds().fillna(0)
    df["above"] = df[col] > threshold
    return df.loc[df["above"], "delta_t"].sum()


def drop_per_km(df, col, discharging_mask=None):
    if len(df) < 2:
        return None

    if discharging_mask is None:
        discharging_mask = pd.Series(True, index=df.index)

    delta_km = df["ODOMETER"].diff()
    is_running = delta_km > 0
    mask = discharging_mask & is_running
    if mask.sum() == 0:
        return None

    delta_val = (-df[col].diff()).clip(lower=0)
    total_drop = delta_val[mask].sum()
    total_dist = delta_km[mask].sum()
    if total_dist == 0:
        return None

    return total_drop / total_dist


def drop_per_hour(subdf, col, mask):
    df = subdf.copy()
    drops = []
    df["diff"] = df[col].diff()
    df["delta_t"] = df["EVENT_AT"].diff().dt.total_seconds() / 3600

    is_dropping = False
    t1 = v1 = None

    for i in range(1, len(df)):
        if not mask.iloc[i]:
            is_dropping = False
            t1 = v1 = None
            continue
        d = df["diff"].iloc[i]
        if d < 0:
            if not is_dropping:
                is_dropping = True
                t1 = df["EVENT_AT"].iloc[i-1]
                v1 = df[col].iloc[i-1]
        elif d > 0 and is_dropping:
            t2 = df["EVENT_AT"].iloc[i-1]
            v2 = df[col].iloc[i-1]
            dt = (t2-t1).total_seconds()/3600 if t1 and t2 else 0
            dv = v1 - v2 if v1 and v2 else 0
            if dt > 1/60 and dv > 0:
                drops.append(dv/dt)
            is_dropping = False
            t1 = v1 = None
    if drops:
        return sum(drops)/len(drops)
    return None


def distance_per_month(subdf):
    df = subdf
    total_distance = df["ODOMETER"].iloc[-1] - df["ODOMETER"].iloc[0]
    days = (df["EVENT_AT"].dt.date.nunique())
    if days == 0:
        return None
    return total_distance * (30 / days)


def detect_charging_state(subdf):
    df = subdf.copy()
    soc = pd.to_numeric(df["SOC"], errors="coerce")
    df["soc_diff"] = soc.diff()
    charging = False
    charging_state = []

    for diff in df["soc_diff"]:
        if diff > 0:
            charging = True
        elif diff < 0:
            charging = False
        charging_state.append(charging if diff is not None else False)
    df["is_charging"] = charging_state
    return df["is_charging"]


def charging_start_soc(subdf):
    df = subdf.copy()
    soc = pd.to_numeric(df["SOC"], errors="coerce")
    df["soc_diff"] = soc.diff()
    charging = False
    charging_starts = []

    for idx, diff in enumerate(df["soc_diff"]):
        if diff > 0 and not charging:
            soc_start = df["SOC"].iloc[idx]
            charging_starts.append(soc_start)
            charging = True
        elif diff < 0 and charging:
            charging = False

    return pd.Series(charging_starts, name="SOC")


def charging_end_soc(subdf):
    df = subdf.copy()
    soc_values = pd.to_numeric(df["SOC"], errors="coerce")
    soc_difference = soc_values.diff()
    is_charging = False
    charging_end_soc_values = []
    for row_index in range(len(soc_difference)):
        current_soc_change = soc_difference.iloc[row_index]

        if current_soc_change >= 1:
            is_charging = True
        elif current_soc_change == 0:
            pass
        elif current_soc_change <= -1:
            if is_charging:
                last_charged_soc = soc_values.iloc[row_index - 1] if row_index > 0 else None
                if last_charged_soc is not None:
                    charging_end_soc_values.append(last_charged_soc)
            is_charging = False

    return pd.Series(charging_end_soc_values, name="SOC")


def charging_cycle_count(subdf):
    df = subdf.copy()
    soc = pd.to_numeric(df["SOC"], errors="coerce")
    df["soc_diff"] = soc.diff()
    charging = False
    count = 0

    for diff in df["soc_diff"]:
        if diff > 0 and not charging:
            count += 1
            charging = True
        elif diff < 0 and charging:
            charging = False
    return count


def avg_charging_duration(subdf):
    starts = charging_start_soc(subdf)
    ends = charging_end_soc(subdf)
    event_times = subdf["EVENT_AT"].reset_index(drop=True)
    n = min(len(starts), len(ends))
    if n == 0:
        return None
    start_times = event_times.iloc[starts.index[:n]].reset_index(drop=True)
    end_times = event_times.iloc[ends.index[:n]].reset_index(drop=True)
    durations = (end_times - start_times).dt.total_seconds()
    durations = durations[durations > 0]
    return durations.mean() if len(durations) > 0 else None


def avg_charging_duration_per_soc(subdf):
    df = subdf.copy()
    soc = pd.to_numeric(df["SOC"], errors="coerce")
    soc_diff = soc.diff().reset_index(drop=True)
    event_times = df["EVENT_AT"].reset_index(drop=True)
    soc_increased = False
    last_increase_idx = None
    durations = []

    for i in range(1, len(soc_diff)):
        diff = soc_diff[i]
        if diff > 0:
            if not soc_increased:
                soc_increased = True
                last_increase_idx = i
            else:
                duration = (event_times[i] - event_times[last_increase_idx]).total_seconds()
                durations.append(duration)
                last_increase_idx = i
        elif diff < 0:
            soc_increased = False
            last_increase_idx = None

    if len(durations) == 0:
        return None
    return sum(durations) / len(durations)


def soh_degradation_per_day(subdf, days=None):
    soh_values = subdf["SOH"].dropna()
    if len(soh_values) < 2 or not days:
        return None
    first_soh = soh_values.iloc[0]
    last_soh = soh_values.iloc[-1]
    return (first_soh - last_soh) / days


def days_vehicle_used(subdf):
    df = subdf.dropna(subset=["ODOMETER", "EVENT_AT"]).copy()
    if len(df) == 0:
        return None
    days_used = 0
    for _, group in df.groupby(df["EVENT_AT"].dt.date):
        if group["ODOMETER"].iloc[-1] - group["ODOMETER"].iloc[0] > 1:
            days_used += 1
    return days_used


def power_per_km(subdf):
    df = subdf.copy()
    if "POWER_DRAWN_KWH" not in df.columns:
        return None

    df["delta_power"] = df["POWER_DRAWN_KWH"].diff()
    df["delta_km"] = df["ODOMETER"].diff()
    mask = (df["delta_km"] > 0) & (df["delta_power"] > 0)

    if mask.sum() == 0:
        return None

    return (df.loc[mask, "delta_power"] / df.loc[mask, "delta_km"]).mean()


def power_per_hour(subdf):
    df = subdf.copy()
    if "POWER_DRAWN_KWH" not in df.columns:
        return None

    df["delta_power"] = df["POWER_DRAWN_KWH"].diff()
    df["delta_t"] = df["EVENT_AT"].diff().dt.total_seconds() / 3600
    mask = (df["delta_t"] > 0) & (df["delta_power"] > 0)

    if mask.sum() == 0:
        return None

    return (df.loc[mask, "delta_power"] / df.loc[mask, "delta_t"]).mean()


def compute_soc_threshold_features(subdf):
    soc = subdf["SOC"]
    n_events = int(soc.notna().sum())
    n_days = subdf["EVENT_AT"].dt.date.nunique() if "EVENT_AT" in subdf.columns else 0
    n_charges = charging_cycle_count(subdf)

    thresholds_above = [(95, "95"), (90, "90"), (80, "80")]
    thresholds_below = [(20, "20"), (10, "10"), (5, "5")]

    result = {}

    for thresh, label in thresholds_above:
        count = count_threshold_crossings(soc, thresh)
        result[f"SOC_ABOVE_{label}_COUNT"] = count
        result[f"SOC_ABOVE_{label}_AVG_PER_EVENT"] = count / n_events if n_events > 0 else None
        result[f"SOC_ABOVE_{label}_AVG_PER_DAY"] = count / n_days if n_days > 0 else None
        result[f"SOC_ABOVE_{label}_AVG_PER_CHARGE"] = count / n_charges if n_charges and n_charges > 0 else None

    for thresh, label in thresholds_below:
        count = count_threshold_crossings_below(soc, thresh)
        result[f"SOC_BELOW_{label}_COUNT"] = count
        result[f"SOC_BELOW_{label}_AVG_PER_EVENT"] = count / n_events if n_events > 0 else None
        result[f"SOC_BELOW_{label}_AVG_PER_DAY"] = count / n_days if n_days > 0 else None
        result[f"SOC_BELOW_{label}_AVG_PER_CHARGE"] = count / n_charges if n_charges and n_charges > 0 else None

    return result


def ignition_on_count(subdf):
    df = subdf.copy()
    if "IGNITION_STATUS" not in df.columns:
        return None
    df["ignition_on"] = df["IGNITION_STATUS"] == 1
    df["ignition_start"] = df["ignition_on"] & ~df["ignition_on"].shift(fill_value=False)
    return int(df["ignition_start"].sum())


def compute_cached_features(subdf, days):
    n_events = len(subdf)
    result = {}
    overtemp_count = count_threshold_crossings(subdf["BATTERY_TEMPERATURE"], 60)
    overvoltage_count = count_threshold_crossings(subdf["BATTERY_VOLTAGE"], 100)
    overcurrent_count = count_threshold_crossings(subdf["BATTERY_CURRENT"], 150)
    speed_50_count = count_threshold_crossings(subdf["VEHICLE_SPEED"], 50)
    discharging_mask = ~detect_charging_state(subdf)
    cc_count = charging_cycle_count(subdf)
    dvu = days_vehicle_used(subdf)
    ign_count = ignition_on_count(subdf)
    result["BATTERY_OVERTEMP_COUNT"] = overtemp_count
    result["AVG_BATTERY_OVERTEMP_COUNT_PER_DAY"] = overtemp_count / days if days else None
    result["AVG_BATTERY_OVERTEMP_COUNT_PER_EVENT"] = overtemp_count / n_events if n_events > 0 else None
    result["BATTERY_OVERVOLTAGE_COUNT"] = overvoltage_count
    result["BATTERY_OVERVOLTAGE_COUNT_PER_DAY"] = overvoltage_count / days if days else None
    result["BATTERY_OVERVOLTAGE_COUNT_PER_EVENT"] = overvoltage_count / n_events if n_events > 0 else None
    result["BATTERY_OVERCURRENT_COUNT"] = overcurrent_count
    result["AVG_BATTERY_OVERCURRENT_COUNT_PER_DAY"] = overcurrent_count / days if days else None
    result["SPEED_ABOVE_50_COUNT"] = speed_50_count
    result["SPEED_ABOVE_50_COUNT_PER_EVENT"] = speed_50_count / n_events if n_events > 0 else None
    result["SOC_DROP_PER_KM_RUNNING"] = drop_per_km(subdf, "SOC", discharging_mask)
    result["BATTERY_REMAINING_CAPACITY_DROP_PER_KM_RUNNING"] = drop_per_km(subdf, "REMAINING_CAPACITY", discharging_mask)
    result["CHARGING_CYCLE_COUNT"] = cc_count
    result["AVG_CHARGING_CYCLE_COUNT_PER_DAY"] = cc_count / days if days else None
    result["DAYS_VEHICLE_USED"] = dvu
    result["DAYS_VEHICLE_USED_PER_DAY"] = dvu / days if days and dvu is not None else None
    result["IGNITION_ON_COUNT_PER_DAY"] = (ign_count / days) if days and ign_count is not None else None
    return result


def haversine_meters(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two (lat, lon) points."""
    R = 6371000
    la1, lo1, la2, lo2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
    dlat, dlon = la2 - la1, lo2 - lo1
    a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def night_location(subdf):
    lat = pd.to_numeric(subdf["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(subdf["LONGITUDE"], errors="coerce")
    hour = subdf["EVENT_AT"].dt.hour

    mask = (
        lat.notna() & lon.notna() &
        lat.between(8.0, 37.0) & lon.between(68.0, 98.0) &
        hour.between(0, 6)
    )

    lat_valid, lon_valid = lat[mask].values, lon[mask].values
    if len(lat_valid) < 1:
        return None

    coords_rad = np.radians(np.column_stack([lat_valid, lon_valid]))
    labels = DBSCAN(
        eps=100 / 6371000, min_samples=1, metric='haversine', algorithm='ball_tree'
    ).fit_predict(coords_rad)

    dominant = max(set(labels) - {-1}, key=lambda c: (labels == c).sum())
    m = labels == dominant
    return f"{lat_valid[m].mean():.5f}, {lon_valid[m].mean():.5f}"


FEATURES = {
    "BATTERY_REMAINING_CAPACITY": {
        "col": "REMAINING_CAPACITY",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "SOC": {
        "col": "SOC",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "SOH": {
        "col": "SOH",
        "stats": ["latest"]
    },
    "BATTERY_CURRENT": {
        "col": "BATTERY_CURRENT",
        "stats": ["latest", "avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "BATTERY_TEMPERATURE": {
        "col": "BATTERY_TEMPERATURE",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "BATTERY_VOLTAGE": {
        "col": "BATTERY_VOLTAGE",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "VEHICLE_SPEED": {
        "col": "VEHICLE_SPEED",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "ACCELERATION": {
        "col": "ACCELERATION",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "CONTROLLER_TEMPERATURE": {
        "col": "CONTROLLER_TEMPERATURE",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "CHARGING_START_SOC": {
        "col": "CHARGING_START_SOC",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "CHARGING_END_SOC": {
        "col": "CHARGING_END_SOC",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "MOTOR_TEMPERATURE": {
        "col": "MOTOR_TEMPERATURE",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "BATTERY_TEMPERATURE_DURING_CHARGE": {
        "col": "BATTERY_TEMP_DURING_CHARGE",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "BATTERY_TEMPERATURE_DURING_RUNNING": {
        "col": "BATTERY_TEMP_DURING_RUNNING",
        "stats": ["avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    },
    "POWER_DRAWN_KWH": {
        "col": "POWER_DRAWN_KWH",
        "stats": ["latest", "avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    }
}

COMPLEX_FEATURES = {
    "SOC_DROP_PER_HR_RUNNING": lambda df, days: drop_per_hour(df, "SOC", df["VEHICLE_SPEED"] > 1),
    "SOC_DROP_PER_HR_IDLE": lambda df, days: drop_per_hour(df, "SOC", df["VEHICLE_SPEED"] == 0),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_HR_RUNNING": lambda df, days: drop_per_hour(df, "REMAINING_CAPACITY", df["VEHICLE_SPEED"] > 1),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_HR_IDLE": lambda df, days: drop_per_hour(df, "REMAINING_CAPACITY", df["VEHICLE_SPEED"] == 0),
    "AVG_CHARGING_DURATION": lambda df, days: avg_charging_duration(df),
    "AVG_CHARGING_DURATION_PER_SOC_INCREASE": lambda df, days: avg_charging_duration_per_soc(df),
    "SOH_DEGRADATION_PER_DAY": lambda df, days: soh_degradation_per_day(df, days),
    "BATTERY_OVERTEMP_DURATION": lambda df, days: duration_above_threshold(df, "BATTERY_TEMPERATURE", 60),
    "BATTERY_OVERVOLTAGE_DURATION": lambda df, days: duration_above_threshold(df, "BATTERY_VOLTAGE", 100),
    "BATTERY_OVERCURRENT_DURATION": lambda df, days: duration_above_threshold(df, "BATTERY_CURRENT", 150),
    "DISTANCE_TRAVELLED": lambda df, days: df["ODOMETER"].iloc[-1] - df["ODOMETER"].iloc[0] if len(df) > 0 else None,
    "DISTANCE_PER_MONTH": lambda df, days: distance_per_month(df),
    "LATEST_ODOMETER": lambda df, days: df["ODOMETER"].iloc[-1] if len(df) > 0 else None,
    "BATTERY_POWER_KWH_PER_KM": lambda df, days: power_per_km(df),
    "BATTERY_POWER_KWH_PER_HR": lambda df, days: power_per_hour(df),
    "NIGHT_LOCATION": lambda df, days: night_location(df),
}


def _compute_features_chunk(args):
    """Worker for ProcessPoolExecutor: (df_std_chunk, days) -> DataFrame."""
    df_chunk, days = args
    return compute_features(df_chunk, days)


def compute_features(df_std, days):
    final_rows = []

    stamped_date = pd.Timestamp.now().normalize()

    for vin, subdf in df_std.groupby("VIN"):
        oem = subdf["OEM"].iloc[0] if "OEM" in subdf else None

        if "EVENT_AT" in subdf.columns:
            subdf = subdf.sort_values("EVENT_AT")
        subdf["ACCELERATION"] = subdf["VEHICLE_SPEED"].diff()/(subdf["EVENT_AT"].diff().dt.total_seconds() / 3600)
        subdf["ACCELERATION"] = subdf["ACCELERATION"].mask(
            (subdf["ACCELERATION"] < SANITIZATION_RULES["ACCELERATION"]["min"]) |
            (subdf["ACCELERATION"] > SANITIZATION_RULES["ACCELERATION"]["max"])
        )
        start_soc = charging_start_soc(subdf)
        end_soc = charging_end_soc(subdf)
        n_start, n_end = len(start_soc), len(end_soc)
        subdf["CHARGING_START_SOC"] = float("nan")
        subdf["CHARGING_END_SOC"] = float("nan")
        if n_start:
            subdf.iloc[:n_start, subdf.columns.get_loc("CHARGING_START_SOC")] = start_soc.values
        if n_end:
            subdf.iloc[:n_end, subdf.columns.get_loc("CHARGING_END_SOC")] = end_soc.values
        charging_mask = detect_charging_state(subdf)
        running_mask = subdf["VEHICLE_SPEED"] > 1
        subdf["BATTERY_TEMP_DURING_CHARGE"] = subdf.loc[charging_mask, "BATTERY_TEMPERATURE"]
        subdf["BATTERY_TEMP_DURING_RUNNING"] = subdf.loc[running_mask, "BATTERY_TEMPERATURE"]

        row = {
            "VIN": vin,
            "OEM": oem,
            "STAMPED_DATE": stamped_date
        }

        for feat, spec in FEATURES.items():
            stats = compute_stats(subdf[spec["col"]], spec["stats"])
            for stat_name, stat_val in stats.items():
                row[f"{feat}_{stat_name}_L{days}"] = stat_val

        for feat_name, func in COMPLEX_FEATURES.items():
            try:
                row[f"{feat_name}_L{days}"] = func(subdf, days)
            except Exception as e:
                row[f"{feat_name}_L{days}"] = None

        try:
            soc_feats = compute_soc_threshold_features(subdf)
            for feat_name, val in soc_feats.items():
                row[f"{feat_name}_L{days}"] = val
        except Exception:
            pass

        try:
            cached_feats = compute_cached_features(subdf, days)
            for feat_name, val in cached_feats.items():
                row[f"{feat_name}_L{days}"] = val
        except Exception:
            pass

        final_rows.append(row)

    return pd.DataFrame(final_rows)


def convert_to_long(df_features):
    id_cols = ["VIN", "OEM", "STAMPED_DATE"]
    df_long = df_features.melt(
        id_vars=id_cols,
        var_name="Feature_Name",
        value_name="Feature_Value"
    )
    df_long = df_long.dropna(subset=["Feature_Value"])
    return df_long


def compute_triggers(df_features, days):

    try:
        df_app = pd.read_excel(APPLICANT_EXCEL, usecols=["vin", "applicant_address.latitude", "applicant_address.longitude"])
        df_app = df_app.rename(columns={"vin": "VIN", "applicant_address.latitude": "APP_LAT", "applicant_address.longitude": "APP_LON"})
        df_app = df_app.dropna(subset=["APP_LAT", "APP_LON"])
        df_app = df_app.drop_duplicates(subset=["VIN"], keep="first")
        app_lookup = df_app.set_index("VIN")[["APP_LAT", "APP_LON"]].to_dict("index")
    except Exception as e:
        log(f"Could not load applicant Excel: {e}")
        app_lookup = {}

    night_col = f"NIGHT_LOCATION_L{days}"
    dist_col = f"DISTANCE_TRAVELLED_L{days}"
    min_distance = 1.0 * days

    rows = []
    for _, r in df_features.iterrows():
        vin, oem = r["VIN"], r.get("OEM")
        night_val = r.get(night_col)
        if night_val and vin in app_lookup:
            try:
                nlat, nlon = map(float, str(night_val).split(","))
                app = app_lookup[vin]
                dist_m = round(haversine_meters(nlat, nlon, app["APP_LAT"], app["APP_LON"]), 2)
                if dist_m > 60000:
                    rows.append({"VIN": vin, "OEM": oem, "Feature_Name": f"FAR_FROM_NIGHT_LOCATION_L{days}", "Feature_Value": dist_m})
            except Exception:
                pass

        dist_val = r.get(dist_col)
        if dist_val is not None:
            try:
                dist_km = round(float(dist_val), 2)
                if dist_km < min_distance:
                    rows.append({"VIN": vin, "OEM": oem, "Feature_Name": f"LOW_DISTANCE_TRAVELLED_L{days}", "Feature_Value": dist_km})
            except Exception:
                pass

    return pd.DataFrame(rows, columns=["VIN", "OEM", "Feature_Name", "Feature_Value"])
