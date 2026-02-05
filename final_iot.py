import psycopg2
import pandas as pd
from datetime import datetime
import time
import os
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

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

TS_COLUMN_MAP = {
    'piaggio_vehicle_data': 'gps_data_timestamp',
    'mahindra_vehicle_data': 'last_connected',
    'euler_vehicle_data': 'location_data_last_updated_at',
    'montra_location_data': 'event_at'
}

def fetch_iot_data(conn, table_name, vins, days):
    t0 = time.time()
    log(f"fetch_iot_data START table={table_name} ({len(vins)} VINs)")
    ts_col = TS_COLUMN_MAP.get(table_name, "timestamp")

    sql = f"""
        SELECT *
        FROM {table_name}
        WHERE vin = ANY(%s)
        AND {ts_col} >= NOW() - INTERVAL '{int(days)} days'
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_sql(sql, conn, params=(vins,))
    df["table_name"] = table_name
    df["OEM"] = df["vin"].apply(get_oem_from_vin)
    log(f"fetch_iot_data END table={table_name}", time.time() - t0)
    log_df_size(f"fetch_iot_data {table_name}", df)
    return df


def _fetch_one_table(table_name, vins, days):
    """Fetch one table using its own connection (for parallel execution)."""
    conn = connect()
    if not conn:
        return pd.DataFrame()
    try:
        return fetch_iot_data(conn, table_name, vins, days)
    finally:
        conn.close()


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
    df["OEM"] = df["VIN"].apply(get_oem_from_vin)
    df["table_name"] = df["VIN"].apply(get_table_for_vin)
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

def drop_per_km(subdf, col):
    df = subdf.copy()
    df["delta_val"] = (-df[col].diff()).clip(lower=0)
    df["delta_km"] = df["ODOMETER"].diff()
    mask = (df["delta_km"] > 0)
    return (df.loc[mask, "delta_val"] / df.loc[mask, "delta_km"]).mean()

def drop_per_hour(subdf, col, mask):
    df = subdf.copy()
    df["delta_val"] = (-df[col].diff()).clip(lower=0)     #clip neg values to 0
    df["delta_t"] = df["EVENT_AT"].diff().dt.total_seconds() / 3600
    valid = mask & (df["delta_t"] > 0)
    return (df.loc[valid, "delta_val"] / df.loc[valid, "delta_t"]).mean()

def distance_per_month(subdf):
    df = subdf
    total_distance = df["ODOMETER"].iloc[-1] - df["ODOMETER"].iloc[0]
    days = (df["EVENT_AT"].dt.date.nunique())
    if days == 0:
        return None
    return total_distance * (30 / days)

def detect_charging_state(subdf):
    df = subdf.copy()
    df["soc_diff"] = df["SOC"].diff()
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
    df["soc_diff"] = df["SOC"].diff()
    is_charging = df["soc_diff"] > 0
    df["charging_start"] = is_charging & (~is_charging.shift(fill_value=False))
    starts = df.loc[df["charging_start"], "SOC"]
    return starts

def charging_end_soc(subdf):
    df = subdf.copy()
    df["soc_diff"] = df["SOC"].diff()
    charging = False
    end_socs = []

    for idx, diff in enumerate(df["soc_diff"]):
        if diff > 0 and not charging:
            charging = True
        elif diff < 0 and charging:
            # We consider the previous SOC as the end of charging
            end_soc = df["SOC"].iloc[idx - 1] if idx > 0 else None
            if end_soc is not None:
                end_socs.append(end_soc)
            charging = False

    return pd.Series(end_socs, name="SOC")

def charging_cycle_count(subdf):
    df = subdf.copy()
    df["soc_diff"] = df["SOC"].diff()
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
    df = subdf.copy()
    df["soc_diff"] = df["SOC"].diff()
    df["is_charging"] = df["soc_diff"] > 0
    df["charging_session"] = (~df["is_charging"]).cumsum()
    
    durations = []
    for session, group in df[df["is_charging"]].groupby("charging_session"):
        if len(group) > 1:
            duration = (group["EVENT_AT"].iloc[-1] - group["EVENT_AT"].iloc[0]).total_seconds()
            durations.append(duration)
    
    return sum(durations) / len(durations) if len(durations) > 0 else None

def avg_charging_duration_per_soc(subdf):
    df = subdf.copy()
    df["soc_diff"] = df["SOC"].diff()
    df["time_diff"] = df["EVENT_AT"].diff().dt.total_seconds()
    df["is_charging"] = df["soc_diff"] > 0
    
    charging_df = df[df["is_charging"] & (df["soc_diff"] > 0)]
    if len(charging_df) == 0:
        return None
    
    time_per_pct = charging_df["time_diff"] / charging_df["soc_diff"]
    return time_per_pct.mean()

def soh_degradation_per_day(subdf):
    df = subdf.copy()
    soh_values = df["SOH"].dropna()
    if len(soh_values) < 2:
        return None
    
    first_soh = soh_values.iloc[0]
    last_soh = soh_values.iloc[-1]
    days = (df["EVENT_AT"].iloc[-1] - df["EVENT_AT"].iloc[0]).days
    
    if days == 0:
        return None
    
    return (first_soh - last_soh) / days

def temperature_during_condition(subdf, condition_mask):
    df = subdf.copy()
    temps = df.loc[condition_mask, "BATTERY_TEMPERATURE"].dropna()
    return temps.mean() if len(temps) > 0 else None

def days_vehicle_used(subdf):
    return subdf["EVENT_AT"].dt.date.nunique()

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

def ratio_event_per_charge(event_count, charge_cycles):
    if charge_cycles in (None, 0):
        return None
    return event_count / charge_cycles

def ignition_on_count_per_day(subdf):
    df = subdf.copy()
    if "IGNITION_STATUS" not in df.columns:
        return None
    
    df["ignition_on"] = df["IGNITION_STATUS"] == 1
    df["ignition_start"] = df["ignition_on"] & ~df["ignition_on"].shift(fill_value=False)
    ignition_count = df["ignition_start"].sum()
    
    days = df["EVENT_AT"].dt.date.nunique()
    if days == 0:
        return None
    
    return ignition_count / days

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
        subdf["CHARGING_START_SOC"] = charging_start_soc(subdf)
        subdf["CHARGING_END_SOC"] = charging_end_soc(subdf)

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
    "POWER_DRAWN_KWH": {
        "col": "POWER_DRAWN_KWH",
        "stats": ["latest", "avg", "min", "max", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "var"]
    }
}

COMPLEX_FEATURES = {
    "SOC_ABOVE_95_COUNT": lambda df: count_threshold_crossings(df["SOC"], 95),
    "SOC_ABOVE_90_COUNT": lambda df: count_threshold_crossings(df["SOC"], 90),
    "SOC_ABOVE_80_COUNT": lambda df: count_threshold_crossings(df["SOC"], 80),
    "SOC_BELOW_20_COUNT": lambda df: count_threshold_crossings_below(df["SOC"], 20),
    "SOC_BELOW_10_COUNT": lambda df: count_threshold_crossings_below(df["SOC"], 10),
    "SOC_BELOW_5_COUNT": lambda df: count_threshold_crossings_below(df["SOC"], 5),
    "SOC_DROP_PER_KM_RUNNING": lambda df: drop_per_km(df[df["VEHICLE_SPEED"] > 1], "SOC"),
    "SOC_DROP_PER_HR_RUNNING": lambda df: drop_per_hour(df, "SOC", df["VEHICLE_SPEED"] > 1),
    "SOC_DROP_PER_HR_IDLE": lambda df: drop_per_hour(df, "SOC", df["VEHICLE_SPEED"] == 0),
    "RATIO_SOC_ABOVE_95_PER_CHARGE": lambda df: ratio_event_per_charge(count_threshold_crossings(df["SOC"], 95), charging_cycle_count(df)),
    "RATIO_SOC_ABOVE_90_PER_CHARGE": lambda df: ratio_event_per_charge(count_threshold_crossings(df["SOC"], 90), charging_cycle_count(df)),
    "RATIO_SOC_ABOVE_80_PER_CHARGE": lambda df: ratio_event_per_charge(count_threshold_crossings(df["SOC"], 80), charging_cycle_count(df)),
    "RATIO_SOC_BELOW_20_PER_CHARGE": lambda df: ratio_event_per_charge(count_threshold_crossings_below(df["SOC"], 20), charging_cycle_count(df)),
    "RATIO_SOC_BELOW_10_PER_CHARGE": lambda df: ratio_event_per_charge(count_threshold_crossings_below(df["SOC"], 10), charging_cycle_count(df)),
    "RATIO_SOC_BELOW_5_PER_CHARGE": lambda df: ratio_event_per_charge(count_threshold_crossings_below(df["SOC"], 5), charging_cycle_count(df)),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_KM_RUNNING": lambda df: drop_per_km(df[df["VEHICLE_SPEED"] > 1], "REMAINING_CAPACITY"),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_HR_RUNNING": lambda df: drop_per_hour(df, "REMAINING_CAPACITY", df["VEHICLE_SPEED"] > 1),
    "BATTERY_REMAINING_CAPACITY_DROP_PER_HR_IDLE": lambda df: drop_per_hour(df, "REMAINING_CAPACITY", df["VEHICLE_SPEED"] == 0),
    "CHARGING_CYCLE_COUNT": lambda df: charging_cycle_count(df),
    "AVG_CHARGING_DURATION": lambda df: avg_charging_duration(df),
    "AVG_CHARGING_DURATION_PER_SOC_INCREASE": lambda df: avg_charging_duration_per_soc(df),
    "SOH_DEGRADATION_PER_DAY": lambda df: soh_degradation_per_day(df),
    "BATTERY_OVERTEMP_COUNT": lambda df: count_threshold_crossings(df["BATTERY_TEMPERATURE"], 60),
    "BATTERY_OVERTEMP_DURATION": lambda df: duration_above_threshold(df, "BATTERY_TEMPERATURE", 60),
    "BATTERY_TEMPERATURE_DURING_CHARGE": lambda df: temperature_during_condition(df, detect_charging_state(df)),
    "BATTERY_TEMPERATURE_DURING_RUNNING": lambda df: temperature_during_condition(df, df["VEHICLE_SPEED"] > 1),
    "BATTERY_OVERVOLTAGE_COUNT": lambda df: count_threshold_crossings(df["BATTERY_VOLTAGE"], 100),
    "BATTERY_OVERVOLTAGE_DURATION": lambda df: duration_above_threshold(df, "BATTERY_VOLTAGE", 100),
    "BATTERY_OVERCURRENT_COUNT": lambda df: count_threshold_crossings(df["BATTERY_CURRENT"], 150),
    "BATTERY_OVERCURRENT_DURATION": lambda df: duration_above_threshold(df, "BATTERY_CURRENT", 150),
    "SPEED_ABOVE_50_COUNT": lambda df: count_threshold_crossings(df["VEHICLE_SPEED"], 50),
    "DISTANCE_TRAVELLED": lambda df: df["ODOMETER"].iloc[-1] - df["ODOMETER"].iloc[0] if len(df) > 0 else None,
    "DISTANCE_PER_MONTH": lambda df: distance_per_month(df),
    "LATEST_ODOMETER": lambda df: df["ODOMETER"].iloc[-1] if len(df) > 0 else None,
    "DAYS_VEHICLE_USED": lambda df: days_vehicle_used(df),
    "IGNITION_ON_COUNT_PER_DAY": lambda df: ignition_on_count_per_day(df),
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
    log("START main")
    vins = open('vins.txt').read().splitlines()
    log(f"Loaded {len(vins)} VINs from vins.txt")
    days = input("Enter number of days: ")

    t0 = time.time()
    conn = connect()
    log("DB connect", time.time() - t0)
    if not conn:
        log("DB connection failed.")
        return

    t0 = time.time()
    df_map = map_vins_to_tables(vins)
    table_groups = df_map.groupby("table_name")["VIN"]
    tasks = [(table_name, s.to_list()) for table_name, s in table_groups]
    dfs = []
    with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as ex:
        futures = {ex.submit(_fetch_one_table, table_name, vlist, days): table_name for table_name, vlist in tasks}
        for fut in as_completed(futures):
            df_part = fut.result()
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)
    log("fetch_iot_data (all tables, parallel)", time.time() - t0)

    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
    else:
        df_final = pd.DataFrame()

    if df_final.empty:
        log("No IoT data retrieved. Exiting.")
        conn.close()
        return

    conn.close()
    log("DB connection closed (fetch done).")
    log_df_size("df_final (raw IoT)", df_final)

    t0 = time.time()
    df_std = standardize(df_final)
    log("standardize", time.time() - t0)
    log_df_size("df_std (standardized)", df_std)

    t0 = time.time()
    unique_vins = df_std["VIN"].unique()
    n_workers = min(4, len(unique_vins))
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
    # df_features.to_csv("df_features.csv", index=False)

    t0 = time.time()
    df_long = convert_to_long(df_features)
    df_long['Feature_Value'] = df_long['Feature_Value'].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
    log("convert_to_long + round", time.time() - t0)
    log_df_size("df_long", df_long)

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"computed_features_{timestamp}.csv"
    df_long.to_csv(output_file, index=False)
    log(f"to_csv -> {output_file}", time.time() - t0)

    log("END main")

if __name__ == "__main__":
    main()
