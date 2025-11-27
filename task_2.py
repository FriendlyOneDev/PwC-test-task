import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def parse_time_to_seconds(time_str):
    """Convert GTFS time format (HH:MM:SS) to seconds, handling times >= 24:00:00"""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_hms(seconds):
    """Convert seconds to hours, minutes, seconds"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, minutes, secs


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth (in km)"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371

    return c * r


def find_longest_route():
    """Find the longest train route in terms of duration"""
    print("=" * 80)
    print("TASK 2.1: Finding the longest train route (by duration)")
    print("=" * 80)

    # Load data
    stop_times = pd.read_csv("input/stop_times.txt")
    trips = pd.read_csv("input/trips.txt")
    stops = pd.read_csv("input/stops.txt")

    # For each trip, find the first and last stop times
    trip_durations = []

    for trip_id in stop_times["trip_id"].unique():
        trip_stops = stop_times[stop_times["trip_id"] == trip_id].sort_values(
            "stop_sequence"
        )

        if len(trip_stops) >= 2:
            first_stop = trip_stops.iloc[0]
            last_stop = trip_stops.iloc[-1]

            # Parse times
            departure_time = parse_time_to_seconds(first_stop["departure_time"])
            arrival_time = parse_time_to_seconds(last_stop["arrival_time"])

            # Calculate duration
            duration = arrival_time - departure_time

            # Handle cases where arrival is next day (if negative, add 24 hours)
            if duration < 0:
                duration += 24 * 3600

            trip_durations.append(
                {
                    "trip_id": trip_id,
                    "duration_seconds": duration,
                    "first_stop_id": first_stop["stop_id"],
                    "last_stop_id": last_stop["stop_id"],
                    "departure_time": first_stop["departure_time"],
                    "arrival_time": last_stop["arrival_time"],
                    "num_stops": len(trip_stops),
                }
            )

    # Convert to DataFrame and find the longest
    durations_df = pd.DataFrame(trip_durations)
    longest = durations_df.loc[durations_df["duration_seconds"].idxmax()]

    # Get trip details
    trip_info = trips[trips["trip_id"] == longest["trip_id"]].iloc[0]

    # Get station names
    first_station = stops[stops["stop_id"] == longest["first_stop_id"]][
        "stop_name"
    ].values[0]
    last_station = stops[stops["stop_id"] == longest["last_stop_id"]][
        "stop_name"
    ].values[0]

    # Convert duration to readable format
    hours, minutes, secs = seconds_to_hms(longest["duration_seconds"])

    print(f"\nLongest train route:")
    print(f"  Trip ID: {longest['trip_id']}")
    print(f"  Train: {trip_info['trip_short_name']}")
    print(f"  From: {first_station} (departure: {longest['departure_time']})")
    print(f"  To: {last_station} (arrival: {longest['arrival_time']})")
    print(
        f"  Duration: {hours}h {minutes}m {secs}s ({longest['duration_seconds']} seconds)"
    )
    print(f"  Number of stops: {longest['num_stops']}")
    print()

    return longest


def find_furthest_stations():
    """Find the two train stations that are furthest apart geographically"""
    print("=" * 80)
    print(
        "TASK 2.2: Finding the two train stations furthest apart (geographical distance)"
    )
    print("=" * 80)

    # Load stops data
    stops = pd.read_csv("input/stops.txt")

    # Clean up lat/lon data (remove extra spaces)
    stops["stop_lat"] = stops["stop_lat"].astype(str).str.strip().astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(str).str.strip().astype(float)

    # Remove any invalid coordinates
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    print(f"\nAnalyzing {len(stops)} stations...")

    # Calculate distances between all pairs of stations
    # For efficiency, we'll use vectorized operations
    max_distance = 0
    station1_idx = 0
    station2_idx = 0

    # Using numpy arrays for faster computation
    lats = stops["stop_lat"].values
    lons = stops["stop_lon"].values

    # This is O(n^2) but necessary to find the maximum
    for i in range(len(stops)):
        for j in range(i + 1, len(stops)):
            distance = haversine_distance(lats[i], lons[i], lats[j], lons[j])
            if distance > max_distance:
                max_distance = distance
                station1_idx = i
                station2_idx = j

    # Get station details
    station1 = stops.iloc[station1_idx]
    station2 = stops.iloc[station2_idx]

    print(f"\nTwo furthest stations:")
    print(f"  Station 1: {station1['stop_name']}")
    print(f"    Coordinates: ({station1['stop_lat']:.5f}, {station1['stop_lon']:.5f})")
    print(f"  Station 2: {station2['stop_name']}")
    print(f"    Coordinates: ({station2['stop_lat']:.5f}, {station2['stop_lon']:.5f})")
    print(f"  Distance: {max_distance:.2f} km")
    print()

    return {
        "station1": station1["stop_name"],
        "station2": station2["stop_name"],
        "distance_km": max_distance,
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SLOVAK RAILWAYS DATA ANALYSIS - TASK 2")
    print("=" * 80 + "\n")

    # Task 2.1: Longest route by duration
    longest_route = find_longest_route()

    # Task 2.2: Furthest stations by geographical distance
    furthest_stations = find_furthest_stations()

    print("=" * 80)
    print("TASK 2 COMPLETE")
    print("=" * 80)
