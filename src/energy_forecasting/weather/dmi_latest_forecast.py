#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://opendataapi.dmi.dk/v1/forecastedr/collections/harmonie_dini_sf/position"
COORDS = "POINT(9.97304 57.01469)"
PARAMETERS = "wind-speed-10m,wind-dir-10m,temperature-2m,pressure-surface"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = PROJECT_ROOT / "data" / "dmi_latest_forecast.csv"


def fetch_latest_forecast(retries=0, delay=1800):
    params = {
        "coords": COORDS,
        "crs": "crs84",
        "parameter-name": PARAMETERS,
        "f": "GeoJSON",
    }
    url = f"{API_URL}?{urlencode(params)}"

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                if attempt <= retries:
                    wait_time = delay * (2 ** (attempt - 1))
                    print(f"Rate limited by DMI API (attempt {attempt}/{retries + 1}). Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(
                    "DMI API rate limit hit. Wait at least 3 hours before retrying, or use --loop with a large interval."
                ) from exc
            if exc.code in {502, 503, 504} and attempt <= retries:
                wait_time = delay
                print(f"Temporary server error {exc.code} from DMI API. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise
        except URLError as exc:
            raise RuntimeError(f"Network error while contacting DMI API: {exc}") from exc
    else:
        if last_error is not None:
            raise RuntimeError(f"DMI API rejected the request after {retries} attempt(s): {last_error}")
        raise RuntimeError("Failed to reach the DMI API after retries.")

    features = payload.get("features", [])
    if not features:
        raise RuntimeError("No forecast features returned from the DMI API.")

    cleaned = []
    for feature in features:
        properties = feature.get("properties", {})
        record = {
            "step": properties.get("step"),
            "wind_speed_10m_mps": properties.get("wind-speed-10m"),
            "wind_dir_10m_deg": properties.get("wind-dir-10m"),
            "temperature_2m_k": properties.get("temperature-2m"),
            "pressure_surface_pa": properties.get("pressure-surface"),
        }

        if record["temperature_2m_k"] is not None:
            record["temperature_2m_celsius"] = record["temperature_2m_k"] - 273.15

        if record["pressure_surface_pa"] is not None:
            record["pressure_surface_hpa"] = record["pressure_surface_pa"] / 100.0

        cleaned.append(record)

    cleaned.sort(key=lambda item: item["step"])
    return cleaned


def write_forecast_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "wind_speed_10m_mps",
        "temperature_2m_k",
        "pressure_surface_pa",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp": row["step"],
                "wind_speed_10m_mps": row["wind_speed_10m_mps"],
                "temperature_2m_k": row["temperature_2m_k"],
                "pressure_surface_pa": row["pressure_surface_pa"],
            })

    print(f"Saved DMI weather data to: {output_path}")


def load_weather_dataframe(csv_path=OUTPUT_FILE):
    """Load the flat CSV and return the MultiIndex DataFrame required by windpowerlib."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to build the windpowerlib weather DataFrame") from exc

    frame = pd.read_csv(csv_path)
    required = [
        "timestamp",
        "wind_speed_10m_mps",
        "temperature_2m_k",
        "pressure_surface_pa",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()

    weather = pd.DataFrame(index=frame.index)
    weather[("wind_speed", 10)] = pd.to_numeric(frame["wind_speed_10m_mps"], errors="coerce")
    weather[("temperature", 2)] = pd.to_numeric(frame["temperature_2m_k"], errors="coerce")
    weather[("pressure", 0)] = pd.to_numeric(frame["pressure_surface_pa"], errors="coerce")
    weather.columns = pd.MultiIndex.from_tuples(weather.columns)

    if weather.isna().any().any():
        raise ValueError("Weather CSV contains missing or invalid numeric values")

    return weather


def fetch_and_write_once(retries=2, delay=60):
    data = fetch_latest_forecast(retries=retries, delay=delay)
    write_forecast_csv(data, OUTPUT_FILE)

    print(f"Latest forecast entries: {len(data)}")
    print("First row:")
    print(data[0])
    print("\nNewest row:")
    print(data[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the newest DMI forecast for a single point and save it to a CSV file.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Explicitly poll the API on a low-frequency schedule and overwrite the same CSV file every N hours.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=3.0,
        help="How often to poll when --loop is enabled (default: 3 hours).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Maximum number of retries for a single request. Default is 0 to avoid repeated rate-limit hits.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=int,
        default=1800,
        help="Base wait time in seconds before retrying after a temporary API error (default: 1800 = 30 minutes).",
    )
    args = parser.parse_args()

    if args.interval_hours < 1.0:
        raise ValueError("--interval-hours must be at least 1.0 to avoid excessive API calls.")

    if args.loop:
        while True:
            fetch_and_write_once(retries=args.retries, delay=args.retry_delay_seconds)
            print(f"Waiting {args.interval_hours} hours before the next fetch...")
            time.sleep(args.interval_hours * 60 * 60)
    else:
        fetch_and_write_once(retries=args.retries, delay=args.retry_delay_seconds)
