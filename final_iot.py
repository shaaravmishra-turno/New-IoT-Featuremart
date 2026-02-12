import psycopg2
import psycopg2.errors
from psycopg2.pool import ThreadedConnectionPool
import pandas as pd
from datetime import datetime
import time
import os
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# Parallel workers: fetch (threads), compute_features (processes). Optimal = CPU count.
N_WORKERS = os.cpu_count() or 4

# Total max threads used for fetching (shared across all tables). Pool and executor use this limit.
MAX_FETCH_THREADS = min(32, os.cpu_count() or 4)

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
days = None
column_mappings = {
    'montra_location_data': {'vin': 'VIN', 'event_at': 'EVENT_AT', 'soc': 'SOC', 'battery_pack_voltage': 'BATTERY_VOLTAGE', 'current': 'BATTERY_CURRENT', 'temperature': 'BATTERY_TEMPERATURE', 'odometer': 'ODOMETER', 'latitude': 'LATITUDE', 'longitude': 'LONGITUDE', 'max_speed': 'VEHICLE_SPEED', 'ignition_status': 'IGNITION_STATUS', 'gps_validity': 'GPS_VALIDITY', 'ride_mode': 'RIDE_MODE', 'rescapacity': 'REMAINING_CAPACITY', 'vehicle_status': 'VEHICLE_STATUS'},
    'mahindra_vehicle_data': {'vin': 'VIN', 'last_connected': 'EVENT_AT', 'soc': 'SOC', 'battery_temp': 'BATTERY_TEMPERATURE', 'kwh': 'POWER_DRAWN_KWH', 'odometer': 'ODOMETER', 'latitude': 'LATITUDE', 'longitude': 'LONGITUDE', 'vehicle_speed': 'VEHICLE_SPEED', 'key_status': 'IGNITION_STATUS', 'vehicle_mode': 'RIDE_MODE', 'distance_to_empty': 'DISTANCE_TO_EMPTY', 'vehicle_status': 'VEHICLE_STATUS', 'license_plate': 'LICENSE_PLATE', 'vehicle_model': 'VEHICLE_MODEL', 'vehicle_variant': 'VEHICLE_VARIANT', 'color': 'COLOR', 'gear_position': 'GEAR_POSITION', 'state': 'STATE', 'latitude_direction': 'LATITUDE_DIRECTION', 'longitude_direction': 'LONGITUDE_DIRECTION', 'gps_validity_flag': 'GPS_VALIDITY'},
    'euler_vehicle_data': {'vin': 'VIN', 'location_data_last_updated_at': 'EVENT_AT', 'battery_soc': 'SOC', 'battery_soh': 'SOH', 'battery_temperature': 'BATTERY_TEMPERATURE', 'battery_voltage': 'BATTERY_VOLTAGE', 'battery_current': 'BATTERY_CURRENT', 'battery_remaining_capacity': 'REMAINING_CAPACITY', 'odometer': 'ODOMETER', 'latitude': 'LATITUDE', 'longitude': 'LONGITUDE', 'speed': 'VEHICLE_SPEED', 'vehicle_mode': 'RIDE_MODE', 'registration_number': 'LICENSE_PLATE', 'imei_number': 'IMEI_NUMBER', 'cell_imbalance': 'CELL_IMBALANCE', 'controller_temperature': 'CONTROLLER_TEMPERATURE', 'motor_temperature': 'MOTOR_TEMPERATURE'},
    'piaggio_vehicle_data': {'vin': 'VIN', 'gps_data_timestamp': 'EVENT_AT', 'soc': 'SOC', 'battery_temperature': 'BATTERY_TEMPERATURE', 'battery_voltage': 'BATTERY_VOLTAGE', 'battery_discharge_current': 'BATTERY_CURRENT', 'odometer': 'ODOMETER', 'latitude': 'LATITUDE', 'longitude': 'LONGITUDE', 'speed': 'VEHICLE_SPEED', 'key_on': 'IGNITION_STATUS', 'drive_mode': 'RIDE_MODE', 'distance_till_empty': 'DISTANCE_TO_EMPTY', 'battery_fault': 'BATTERY_FAULT', 'battery_charging': 'BATTERY_CHARGING', 'controller_temperature': 'CONTROLLER_TEMPERATURE', 'motor_temperature': 'MOTOR_TEMPERATURE', 'hand_throttle': 'HAND_THROTTLE', 'battery_over_voltage': 'BATTERY_OVER_VOLTAGE', 'fault_controller_over_temp': 'FAULT_CONTROLLER_OVER_TEMP', 'fault_controller_under_temp': 'FAULT_CONTROLLER_UNDER_TEMP', 'fault_controller_over_current': 'FAULT_CONTROLLER_OVER_CURRENT', 'motor_no': 'MOTOR_NO'}
}

std_columns = [
            'VIN', 'EVENT_AT', 'SOC', 'SOH', 'BATTERY_TEMPERATURE', 'BATTERY_VOLTAGE', 
            'BATTERY_CURRENT', 'REMAINING_CAPACITY', 'ODOMETER', 'LATITUDE', 'LONGITUDE', 
            'VEHICLE_SPEED', 'IGNITION_STATUS', 'GPS_VALIDITY', 'RIDE_MODE', 'DISTANCE_TO_EMPTY',
            'VEHICLE_STATUS', 'LICENSE_PLATE', 'VEHICLE_MODEL', 'VEHICLE_VARIANT', 'COLOR',
            'GEAR_POSITION', 'STATE', 'LATITUDE_DIRECTION', 'LONGITUDE_DIRECTION', 
            'POWER_DRAWN_KWH', 'IMEI_NUMBER', 'CELL_IMBALANCE', 
            'CONTROLLER_TEMPERATURE', 'MOTOR_TEMPERATURE', 'BATTERY_FAULT', 'BATTERY_CHARGING',
            'HAND_THROTTLE', 'BATTERY_OVER_VOLTAGE', 'FAULT_CONTROLLER_OVER_TEMP',
            'FAULT_CONTROLLER_UNDER_TEMP', 'FAULT_CONTROLLER_OVER_CURRENT', 'MOTOR_NO',
            'ACCELERATION'
        ]

def connect():
    try:
        return psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    except Exception as e:
        print("DB error:", e)

VIN_TABLE_MAP = {
    "MBX": "piaggio_vehicle_data",
    "MA":  "mahindra_vehicle_data",
    "MD9": "euler_vehicle_data",
    "P60": "montra_location_data"
}


def get_table_for_vin(vin):
    return (
        VIN_TABLE_MAP.get(vin[:3]) or  
        VIN_TABLE_MAP.get(vin[:2]) or  
        None
    )


# Vectorized OEM/table lookup (faster than .apply on large series)
OEM_FROM_PREFIX3 = {"MD9": "Euler", "MBX": "Piaggio", "P60": "Montra"}
OEM_FROM_PREFIX2 = {"MA": "Mahindra"}


def oem_from_vin_series(vin_series):
    v = vin_series.astype(str)
    out = v.str[:3].map(OEM_FROM_PREFIX3)
    return out.fillna(v.str[:2].map(OEM_FROM_PREFIX2)).fillna("Unknown")


def table_from_vin_series(vin_series):
    v = vin_series.astype(str)
    out = v.str[:3].map(VIN_TABLE_MAP)
    return out.fillna(v.str[:2].map(VIN_TABLE_MAP))


TS_COLUMN_MAP = {
    'piaggio_vehicle_data': 'gps_data_timestamp',
    'mahindra_vehicle_data': 'last_connected',
    'euler_vehicle_data': 'location_data_last_updated_at',
    'montra_location_data': 'event_at'
}


def _select_columns_for_table(table_name):
    """Return comma-separated column list for SELECT; only columns we use in standardize."""
    mapping = column_mappings.get(table_name, {})
    return ", ".join(mapping.keys()) if mapping else "*"


def fetch_iot_data(conn, batch_id, table_name, vins, days, quiet=False):
    if not quiet:
        t0 = time.time()
        log(f"fetch_iot_data batch_id={batch_id} START table={table_name} ({len(vins)} VINs)")
    ts_col = TS_COLUMN_MAP.get(table_name, "timestamp")
    cols = _select_columns_for_table(table_name)

    sql = f"""
        SELECT {cols}
        FROM {table_name}
        WHERE vin = ANY(%s)
        AND {ts_col} >= NOW() - INTERVAL '{int(days)} days'
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_sql(sql, conn, params=(vins,))
    df["table_name"] = table_name
    df["OEM"] = oem_from_vin_series(df["vin"])
    if not quiet:
        log(f"fetch_iot_data batch_id={batch_id} END table={table_name}", time.time() - t0)
        log_df_size(f"fetch_iot_data batch_id={batch_id} {table_name}", df)
    return df


def _is_serialization_failure(e):
    """Check if exception is SerializationFailure (direct or wrapped by pandas)."""
    if isinstance(e, psycopg2.errors.SerializationFailure):
        return True
    cause = getattr(e, "__cause__", None)
    return cause is not None and isinstance(cause, psycopg2.errors.SerializationFailure)


def _fetch_one_table(pool, batch_id, table_name, vins, days):
    """Fetch one chunk (table + VIN list). On SerializationFailure, retry each VIN individually and skip only the one(s) that error."""
    if not vins:
        return pd.DataFrame()
    conn = pool.getconn()
    try:
        return fetch_iot_data(conn, batch_id, table_name, vins, days)
    except (psycopg2.errors.SerializationFailure, pd.errors.DatabaseError) as e:
        if not _is_serialization_failure(e):
            raise
        log(f"fetch_iot_data batch_id={batch_id} table={table_name} ({len(vins)} VINs) conflict with recovery, retrying VIN-by-VIN.")
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            pool.putconn(conn, close=True)
        except Exception:
            pass
        conn = pool.getconn()
        dfs = []
        skipped = 0
        for i, vin in enumerate(vins, 1):
            try:
                df_one = fetch_iot_data(conn, batch_id, table_name, [vin], days, quiet=True)
                if df_one is not None and not df_one.empty:
                    dfs.append(df_one)
                    log(f"fetch_iot_data batch_id={batch_id} table={table_name} VIN-by-VIN [{i}/{len(vins)}] {vin} ok ({len(df_one)} rows)")
                else:
                    log(f"fetch_iot_data batch_id={batch_id} table={table_name} VIN-by-VIN [{i}/{len(vins)}] {vin} empty")
            except (psycopg2.errors.SerializationFailure, pd.errors.DatabaseError, psycopg2.InterfaceError) as e2:
                is_retryable = _is_serialization_failure(e2) or isinstance(e2, psycopg2.InterfaceError)
                if is_retryable:
                    skipped += 1
                    reason = "conflict" if _is_serialization_failure(e2) else "connection closed"
                    log(f"fetch_iot_data batch_id={batch_id} table={table_name} VIN-by-VIN [{i}/{len(vins)}] skipping VIN {vin} ({reason}).")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        pool.putconn(conn, close=True)
                    except Exception:
                        pass
                    conn = pool.getconn()
                else:
                    raise
            finally:
                try:
                    conn.rollback()
                except Exception:
                    pass
        log(f"fetch_iot_data batch_id={batch_id} table={table_name} VIN-by-VIN done: {len(dfs)} fetched, {skipped} skipped")
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        pool.putconn(conn)


def pick_and_sanitize(df, cols, rule):
    # Pick first non-null across cols using combine_first (faster than reindex+bfill(axis=1))
    s = None
    for c in cols:
        if c not in df.columns:
            continue
        if s is None:
            s = df[c].copy()
        else:
            s = s.combine_first(df[c])
    if s is None:
        s = pd.Series(index=df.index, dtype=object)

    if rule:
        s = s.mask((s < rule["min"]) | (s > rule["max"]))
    return s

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

def map_vins_to_tables(vin_list):
    df = pd.DataFrame({"VIN": vin_list})
    v = df["VIN"].astype(str)
    df["OEM"] = oem_from_vin_series(v)
    df["table_name"] = table_from_vin_series(v)
    return df

def get_oem_from_vin(vin):
    if vin.startswith("MA"):
        return "Mahindra"
    if vin.startswith("MD9"):
        return "Euler"
    if vin.startswith("MBX"):
        return "Piaggio"
    if vin.startswith("P60"):
        return "Montra"
    return "Unknown"

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

def soh_degradation_per_day(subdf):
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
                row[f"{feat_name}_L{days}"] = func(subdf)
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
    "SOC_DROP_PER_HR_RUNNING": lambda df: drop_per_hour(df, "SOC", df["VEHICLE_SPEED"] > 1),
    "SOC_DROP_PER_HR_IDLE": lambda df: drop_per_hour(df, "SOC", df["VEHICLE_SPEED"] == 0),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_HR_RUNNING": lambda df: drop_per_hour(df, "REMAINING_CAPACITY", df["VEHICLE_SPEED"] > 1),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_HR_IDLE": lambda df: drop_per_hour(df, "REMAINING_CAPACITY", df["VEHICLE_SPEED"] == 0),
    "AVG_CHARGING_DURATION": lambda df: avg_charging_duration(df),
    "AVG_CHARGING_DURATION_PER_SOC_INCREASE": lambda df: avg_charging_duration_per_soc(df),
    "SOH_DEGRADATION_PER_DAY": lambda df: soh_degradation_per_day(df),
    "BATTERY_OVERTEMP_DURATION": lambda df: duration_above_threshold(df, "BATTERY_TEMPERATURE", 60),
    "BATTERY_OVERVOLTAGE_DURATION": lambda df: duration_above_threshold(df, "BATTERY_VOLTAGE", 100),
    "BATTERY_OVERCURRENT_DURATION": lambda df: duration_above_threshold(df, "BATTERY_CURRENT", 150),
    "DISTANCE_TRAVELLED": lambda df: df["ODOMETER"].iloc[-1] - df["ODOMETER"].iloc[0] if len(df) > 0 else None,
    "DISTANCE_PER_MONTH": lambda df: distance_per_month(df),
    "LATEST_ODOMETER": lambda df: df["ODOMETER"].iloc[-1] if len(df) > 0 else None,
    "BATTERY_POWER_KWH_PER_KM": lambda df: power_per_km(df),
    "BATTERY_POWER_KWH_PER_HR": lambda df: power_per_hour(df),
}


def standardize(df_final):
    df_std = pd.DataFrame(columns=std_columns)
    df_std = df_std.reindex(range(len(df_final)))

    df_std["VIN"] = pick_and_sanitize(
        df_final,
        ["vin"],
        None
    )

    df_std["EVENT_AT"] = pick_and_sanitize(
        df_final,
        ["event_at", "last_connected", "location_data_last_updated_at", "gps_data_timestamp"],
        None
    )

    df_std["SOC"] = pick_and_sanitize(
        df_final,
        ["soc", "battery_soc"],
        SANITIZATION_RULES.get("SOC")
    )

    df_std["SOH"] = pick_and_sanitize(
        df_final,
        ["battery_soh"],
        SANITIZATION_RULES.get("SOH")
    )

    df_std["BATTERY_VOLTAGE"] = pick_and_sanitize(
        df_final,
        ["battery_pack_voltage", "battery_voltage"],
        SANITIZATION_RULES.get("BATTERY_VOLTAGE")
    )

    df_std["BATTERY_CURRENT"] = pick_and_sanitize(
        df_final,
        ["current", "battery_current", "battery_discharge_current"],
        SANITIZATION_RULES.get("BATTERY_CURRENT")
    )

    df_std["BATTERY_TEMPERATURE"] = pick_and_sanitize(
        df_final,
        ["temperature", "battery_temp", "battery_temperature"],
        SANITIZATION_RULES.get("BATTERY_TEMPERATURE")
    )

    df_std["REMAINING_CAPACITY"] = pick_and_sanitize(
        df_final,
        ["rescapacity", "battery_remaining_capacity"],
        SANITIZATION_RULES.get("REMAINING_CAPACITY")
    )

    df_std["ODOMETER"] = pick_and_sanitize(
        df_final,
        ["odometer"],
        SANITIZATION_RULES.get("ODOMETER")
    )

    df_std["LATITUDE"] = pick_and_sanitize(
        df_final,
        ["latitude"],
        None
    )

    df_std["LONGITUDE"] = pick_and_sanitize(
        df_final,
        ["longitude"],
        None
    )

    df_std["VEHICLE_SPEED"] = pick_and_sanitize(
        df_final,
        ["speed", "vehicle_speed", "max_speed"],
        SANITIZATION_RULES.get("VEHICLE_SPEED")
    )

    df_std["IGNITION_STATUS"] = pick_and_sanitize(
        df_final,
        ["ignition_status", "key_on", "key_status"],
        None
    )

    df_std["GPS_VALIDITY"] = pick_and_sanitize(
        df_final,
        ["gps_validity", "gps_validity_flag"],
        None
    )

    df_std["RIDE_MODE"] = pick_and_sanitize(
        df_final,
        ["ride_mode", "vehicle_mode", "drive_mode"],
        None
    )

    df_std["DISTANCE_TO_EMPTY"] = pick_and_sanitize(
        df_final,
        ["distance_to_empty", "distance_till_empty"],
        SANITIZATION_RULES.get("DISTANCE_TO_EMPTY")
    )

    df_std["VEHICLE_STATUS"] = pick_and_sanitize(
        df_final,
        ["vehicle_status"],
        None
    )

    df_std["LICENSE_PLATE"] = pick_and_sanitize(
        df_final,
        ["license_plate", "registration_number"],
        None
    )

    df_std["VEHICLE_MODEL"] = pick_and_sanitize(
        df_final,
        ["vehicle_model"],
        None
    )

    df_std["VEHICLE_VARIANT"] = pick_and_sanitize(
        df_final,
        ["vehicle_variant"],
        None
    )

    df_std["COLOR"] = pick_and_sanitize(
        df_final,
        ["color"],
        None
    )

    df_std["GEAR_POSITION"] = pick_and_sanitize(
        df_final,
        ["gear_position"],
        None
    )

    df_std["STATE"] = pick_and_sanitize(
        df_final,
        ["state"],
        None
    )

    df_std["LATITUDE_DIRECTION"] = pick_and_sanitize(
        df_final,
        ["latitude_direction"],
        None
    )

    df_std["LONGITUDE_DIRECTION"] = pick_and_sanitize(
        df_final,
        ["longitude_direction"],
        None
    )

    df_std["POWER_DRAWN_KWH"] = pick_and_sanitize(
        df_final,
        ["kwh"],
        None
    )

    df_std["IMEI_NUMBER"] = pick_and_sanitize(
        df_final,
        ["imei_number"],
        None
    )

    df_std["CELL_IMBALANCE"] = pick_and_sanitize(
        df_final,
        ["cell_imbalance"],
        None
    )

    df_std["CONTROLLER_TEMPERATURE"] = pick_and_sanitize(
        df_final,
        ["controller_temperature"],
        SANITIZATION_RULES.get("CONTROLLER_TEMPERATURE")
    )

    df_std["MOTOR_TEMPERATURE"] = pick_and_sanitize(
        df_final,
        ["motor_temperature"],
        SANITIZATION_RULES.get("MOTOR_TEMPERATURE")
    )

    df_std["BATTERY_FAULT"] = pick_and_sanitize(
        df_final,
        ["battery_fault"],
        None
    )

    df_std["BATTERY_CHARGING"] = pick_and_sanitize(
        df_final,
        ["battery_charging"],
        None
    )

    df_std["HAND_THROTTLE"] = pick_and_sanitize(
        df_final,
        ["hand_throttle"],
        None
    )

    df_std["BATTERY_OVER_VOLTAGE"] = pick_and_sanitize(
        df_final,
        ["battery_over_voltage"],
        None
    )

    df_std["FAULT_CONTROLLER_OVER_TEMP"] = pick_and_sanitize(
        df_final,
        ["fault_controller_over_temp"],
        None
    )

    df_std["FAULT_CONTROLLER_UNDER_TEMP"] = pick_and_sanitize(
        df_final,
        ["fault_controller_under_temp"],
        None
    )

    df_std["FAULT_CONTROLLER_OVER_CURRENT"] = pick_and_sanitize(
        df_final,
        ["fault_controller_over_current"],
        None
    )

    df_std["MOTOR_NO"] = pick_and_sanitize(
        df_final,
        ["motor_no"],
        None
    )

    df_std["ACCELERATION"] = pick_and_sanitize(
        df_final,
        ["acceleration"],
        SANITIZATION_RULES.get("ACCELERATION")
    )

    df_std["OEM"] = df_final["OEM"].values
    df_std["table_name"] = df_final["table_name"].values

    return df_std


def main():
    global days
    main_start = time.time()
    log("START main")
    log(f"N_WORKERS={N_WORKERS} MAX_FETCH_THREADS={MAX_FETCH_THREADS}")
    vins = open('vins.txt').read().splitlines()
    log(f"Loaded {len(vins)} VINs from vins.txt")
    days = input("Enter number of days: ")

    t0 = time.time()
    df_map = map_vins_to_tables(vins)
    table_groups = df_map.groupby("table_name")["VIN"]
    # Build one task per (table, VIN chunk). Total concurrency for fetch is capped by MAX_FETCH_THREADS.
    tasks = []
    batch_id = 0
    for table_name, vin_series in table_groups:
        vlist = vin_series.to_list()
        n_chunks = min(MAX_FETCH_THREADS, max(1, len(vlist)))
        for vin_chunk in np.array_split(vlist, n_chunks):
            if len(vin_chunk) > 0:
                batch_id += 1
                tasks.append((batch_id, table_name, list(vin_chunk)))
    try:
        pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=MAX_FETCH_THREADS,
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
        )
    except Exception as e:
        log(f"DB connection pool failed: {e}")
        log("Total time", time.time() - main_start)
        return
    log("DB connection pool created", time.time() - t0)

    dfs = []
    t0_fetch = time.time()
    with ThreadPoolExecutor(max_workers=MAX_FETCH_THREADS) as ex:
        futures = {ex.submit(_fetch_one_table, pool, bid, tbl, vlist, days): bid for bid, tbl, vlist in tasks}
        for fut in as_completed(futures):
            df_part = fut.result()
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)
    log("fetch_iot_data (all tables, chunks per table in parallel)", time.time() - t0_fetch)

    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
    else:
        df_final = pd.DataFrame()

    if df_final.empty:
        log("No IoT data retrieved. Exiting.")
        pool.closeall()
        log("Total time", time.time() - main_start)
        return

    pool.closeall()
    log("DB connection pool closed (fetch done).")
    log_df_size("df_final (raw IoT)", df_final)

    t0 = time.time()
    df_std = standardize(df_final)
    log("standardize", time.time() - t0)
    log_df_size("df_std (standardized)", df_std)

    t0 = time.time()
    unique_vins = df_std["VIN"].unique()
    n_workers = min(N_WORKERS, len(unique_vins))
    if n_workers <= 1:
        df_features = compute_features(df_std, days)
    else:
        vin_splits = np.array_split(unique_vins, n_workers)
        chunks = [df_std[df_std["VIN"].isin(s)] for s in vin_splits]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_compute_features_chunk, [(c, days) for c in chunks]))
        df_features = pd.concat(results, ignore_index=True)
    log("compute_features", time.time() - t0)
    log_df_size("df_features", df_features)

    t0 = time.time()
    df_long = convert_to_long(df_features)
    fv = df_long["Feature_Value"]
    numeric = pd.to_numeric(fv, errors="coerce")
    df_long["Feature_Value"] = numeric.round(2).fillna(fv)
    log("convert_to_long + round", time.time() - t0)
    log_df_size("df_long", df_long)

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"computed_features_{timestamp}.csv"
    df_long.to_csv(output_file, index=False)
    log(f"to_csv -> {output_file}", time.time() - t0)

    log("Total time", time.time() - main_start)
    log("END main")

if __name__ == "__main__":
    main()
