import sys
import psycopg2
import psycopg2.errors
from psycopg2.pool import ThreadedConnectionPool
import pandas as pd
from datetime import datetime
import time
import os
import warnings
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from compute_features import (
    log,
    log_df_size,
    SANITIZATION_RULES,
    compute_features,
    _compute_features_chunk,
    convert_to_long,
    compute_triggers,
)

SKIPPED_VINS_FILE = "skipped_vins.txt"
_skipped_vins_lock = threading.Lock()

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# Parallel workers: fetch (threads), compute_features (processes). Optimal = CPU count.
N_WORKERS = os.cpu_count() or 4

# Total max threads used for fetching (shared across all tables). Pool and executor use this limit.
MAX_FETCH_THREADS = min(32, os.cpu_count() or 4)

# Max VINs per fetch chunk (smaller chunks reduce load and conflict-with-recovery errors).
MAX_VINS_PER_CHUNK = 50


def append_skipped_vin(vin):
    """Append a skipped VIN to skipped_vins.txt (thread-safe)."""
    with _skipped_vins_lock:
        try:
            with open(SKIPPED_VINS_FILE, "a") as f:
                f.write(vin + "\n")
        except Exception as e:
            log(f"Failed to append skipped VIN to {SKIPPED_VINS_FILE}: {e}")


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
                    append_skipped_vin(vin)
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
    main_start = time.time()
    log("START main")
    log(f"N_WORKERS={N_WORKERS} MAX_FETCH_THREADS={MAX_FETCH_THREADS}")
    vins = open('vins.txt').read().splitlines()
    log(f"Loaded {len(vins)} VINs from vins.txt")
    days_input = input("Enter number of days: ")
    days = int(days_input)
    if days is None or days <= 0:
        log("Invalid days; must be a positive integer. Exiting.")
        return

    t0 = time.time()
    df_map = map_vins_to_tables(vins)
    table_groups = df_map.groupby("table_name")["VIN"]
    tasks = []
    batch_id = 0
    for table_name, vin_series in table_groups:
        vlist = vin_series.to_list()
        n_chunks = max(1, (len(vlist) + MAX_VINS_PER_CHUNK - 1) // MAX_VINS_PER_CHUNK)
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

    t0 = time.time()
    df_triggers = compute_triggers(df_features, days)
    if not df_triggers.empty:
        triggers_file = f"triggers_{timestamp}.csv"
        df_triggers.to_csv(triggers_file, index=False)
        log(f"triggers -> {triggers_file} ({len(df_triggers)} rows)", time.time() - t0)
    else:
        log("No triggers generated.", time.time() - t0)

    log("Total time", time.time() - main_start)
    log("END main")

if __name__ == "__main__":
    main()
