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


def analyze_trip_metrics(stop_times, trips, stops, calendar_dates):
    """Calculate key metrics for each trip to support optimization decisions"""

    print("Analyzing trip metrics...")

    trip_metrics = []

    for trip_id in trips["trip_id"].unique():
        # Get all stops for this trip
        trip_stops = stop_times[stop_times["trip_id"] == trip_id].sort_values(
            "stop_sequence"
        )

        if len(trip_stops) < 2:
            continue

        trip_info = trips[trips["trip_id"] == trip_id].iloc[0]

        # Calculate trip duration
        first_stop = trip_stops.iloc[0]
        last_stop = trip_stops.iloc[-1]
        departure_time = parse_time_to_seconds(first_stop["departure_time"])
        arrival_time = parse_time_to_seconds(last_stop["arrival_time"])
        duration = arrival_time - departure_time
        if duration < 0:
            duration += 24 * 3600

        # Calculate how many days this trip operates
        service_id = trip_info["service_id"]
        days_operating = len(
            calendar_dates[
                (calendar_dates["service_id"] == service_id)
                & (calendar_dates["exception_type"] == 1)
            ]
        )

        # Get origin and destination
        origin_id = first_stop["stop_id"]
        dest_id = last_stop["stop_id"]

        # Extract train type
        trip_short_name = trip_info["trip_short_name"]
        train_type = "".join(filter(str.isalpha, trip_short_name.split()[0]))

        # Count stops
        num_stops = len(trip_stops)

        # Calculate average dwell time (how long train waits at intermediate stops)
        dwell_times = []
        for idx in range(1, len(trip_stops) - 1):  # Exclude first and last
            stop = trip_stops.iloc[idx]
            arrival = parse_time_to_seconds(stop["arrival_time"])
            departure = parse_time_to_seconds(stop["departure_time"])
            dwell = departure - arrival
            if dwell >= 0:
                dwell_times.append(dwell)

        avg_dwell = np.mean(dwell_times) if dwell_times else 0

        # Determine peak hour departure (6-9 AM or 4-7 PM)
        departure_hour = departure_time // 3600
        is_peak = (6 <= departure_hour < 9) or (16 <= departure_hour < 19)

        trip_metrics.append(
            {
                "trip_id": trip_id,
                "trip_short_name": trip_short_name,
                "train_type": train_type,
                "num_stops": num_stops,
                "duration_minutes": duration / 60,
                "days_operating": days_operating,
                "origin_id": origin_id,
                "dest_id": dest_id,
                "departure_time_seconds": departure_time,
                "departure_hour": departure_hour,
                "is_peak": is_peak,
                "avg_dwell_seconds": avg_dwell,
            }
        )

    return pd.DataFrame(trip_metrics)


def calculate_station_importance(stop_times, trips, calendar_dates):
    """
    Calculate importance score for each station based on:
    - Number of daily trips
    - Number of unique routes
    """
    print("Calculating station importance...")

    station_stats = []

    for stop_id in stop_times["stop_id"].unique():
        # Get all trips serving this station
        trips_at_station = stop_times[stop_times["stop_id"] == stop_id][
            "trip_id"
        ].unique()

        # Calculate daily trips (weighted by days operating)
        daily_trips = 0
        for trip_id in trips_at_station:
            trip = trips[trips["trip_id"] == trip_id].iloc[0]
            service_id = trip["service_id"]
            days = len(
                calendar_dates[
                    (calendar_dates["service_id"] == service_id)
                    & (calendar_dates["exception_type"] == 1)
                ]
            )
            # Assume analysis over 365 days
            daily_trips += days / 365 if days > 0 else 0

        # Number of unique routes
        routes_at_station = trips[trips["trip_id"].isin(trips_at_station)][
            "route_id"
        ].nunique()

        station_stats.append(
            {
                "stop_id": stop_id,
                "daily_trips": daily_trips,
                "num_routes": routes_at_station,
                "importance_score": daily_trips * np.log1p(routes_at_station),
            }
        )

    return pd.DataFrame(station_stats)


def calculate_route_redundancy(trip_metrics):
    """
    Find trips that are redundant - same origin/destination with similar departure times
    """
    print("Analyzing route redundancy...")

    # Group by origin-destination pairs
    trip_metrics["route_pair"] = (
        trip_metrics["origin_id"].astype(str)
        + "_"
        + trip_metrics["dest_id"].astype(str)
    )

    redundancy_scores = []

    for route_pair in trip_metrics["route_pair"].unique():
        route_trips = trip_metrics[trip_metrics["route_pair"] == route_pair]

        if len(route_trips) <= 1:
            # No redundancy if only one trip
            for _, trip in route_trips.iterrows():
                redundancy_scores.append(
                    {"trip_id": trip["trip_id"], "redundancy_score": 0}
                )
            continue

        # Sort by departure time
        route_trips = route_trips.sort_values("departure_time_seconds")

        for idx, trip in route_trips.iterrows():
            # Find trips within 2 hours before/after
            time_window = 2 * 3600  # 2 hours in seconds
            similar_trips = route_trips[
                (
                    abs(
                        route_trips["departure_time_seconds"]
                        - trip["departure_time_seconds"]
                    )
                    < time_window
                )
                & (route_trips["trip_id"] != trip["trip_id"])
            ]

            redundancy_score = len(similar_trips)
            redundancy_scores.append(
                {"trip_id": trip["trip_id"], "redundancy_score": redundancy_score}
            )

    return pd.DataFrame(redundancy_scores)


def score_trips_for_cutting(trip_metrics, station_importance, redundancy):
    """
    Score each trip based on how suitable it is for cutting.
    Higher score = better candidate to cut (less impact on passengers)
    """
    print("Scoring trips for potential cuts...")

    # Merge all data
    df = trip_metrics.copy()

    # Add origin/destination importance
    df = df.merge(
        station_importance[["stop_id", "importance_score"]].rename(
            columns={
                "stop_id": "origin_id",
                "importance_score": "origin_importance",
            }
        ),
        on="origin_id",
        how="left",
    )

    df = df.merge(
        station_importance[["stop_id", "importance_score"]].rename(
            columns={"stop_id": "dest_id", "importance_score": "dest_importance"}
        ),
        on="dest_id",
        how="left",
    )

    # Add redundancy
    df = df.merge(redundancy, on="trip_id", how="left")
    df["redundancy_score"] = df["redundancy_score"].fillna(0)

    # Calculate cutting score components

    # 1. Low frequency service (operates fewer days) = easier to cut
    max_days = df["days_operating"].max()
    frequency_score = 1 - (df["days_operating"] / max_days)

    # 2. Short trips = lower impact (people can drive/use alternatives)
    max_duration = df["duration_minutes"].max()
    duration_score = 1 - (df["duration_minutes"] / max_duration)

    # 3. Low importance endpoints = fewer affected passengers
    max_origin_imp = df["origin_importance"].max()
    max_dest_imp = df["dest_importance"].max()
    endpoint_score = 1 - (
        (df["origin_importance"] / max_origin_imp)
        + (df["dest_importance"] / max_dest_imp)
    ) / 2

    # 4. High redundancy = alternatives exist
    max_redundancy = df["redundancy_score"].max()
    if max_redundancy > 0:
        redundancy_score = df["redundancy_score"] / max_redundancy
    else:
        redundancy_score = 0

    # 5. Off-peak trips = fewer passengers affected
    peak_penalty = df["is_peak"].astype(float) * -0.3

    # 6. Many stops = local service, often lower ridership per trip
    max_stops = df["num_stops"].max()
    stops_score = df["num_stops"] / max_stops

    # 7. Premium train types (IC, EC, SC, RJ) should be preserved
    premium_types = ["IC", "EC", "SC", "RJ", "EN", "Ex"]
    premium_penalty = df["train_type"].isin(premium_types).astype(float) * -0.5

    # Combine scores (weighted)
    df["cutting_score"] = (
        0.25 * frequency_score  # Low frequency
        + 0.20 * duration_score  # Short duration
        + 0.20 * endpoint_score  # Unimportant endpoints
        + 0.15 * redundancy_score  # Has alternatives
        + 0.10 * stops_score  # Many stops
        + peak_penalty  # Off-peak bonus
        + premium_penalty  # Preserve premium services
    )

    return df


def recommend_cuts(scored_trips, target_percentage=0.05):
    """
    Recommend which trips to cut based on scoring.
    Also calculates estimated savings and impact.
    """
    print(f"\nRecommending {target_percentage*100}% of trips to cut...")

    total_trips = len(scored_trips)
    trips_to_cut = int(total_trips * target_percentage)

    # Sort by cutting score (highest = best to cut)
    sorted_trips = scored_trips.sort_values("cutting_score", ascending=False)

    # Select top candidates
    recommended_cuts = sorted_trips.head(trips_to_cut)

    # Calculate impact metrics
    total_service_days = scored_trips["days_operating"].sum()
    cut_service_days = recommended_cuts["days_operating"].sum()

    total_route_minutes = (
        scored_trips["duration_minutes"] * scored_trips["days_operating"]
    ).sum()
    cut_route_minutes = (
        recommended_cuts["duration_minutes"] * recommended_cuts["days_operating"]
    ).sum()

    print(f"\n{'='*80}")
    print("RECOMMENDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total trips in system: {total_trips}")
    print(f"Trips to cut: {trips_to_cut} ({target_percentage*100:.1f}%)")
    print(f"\nService day reduction: {cut_service_days:.0f} / {total_service_days:.0f}")
    print(
        f"  ({cut_service_days/total_service_days*100:.2f}% of total service days)"
    )
    print(
        f"\nRoute-minutes reduction: {cut_route_minutes:.0f} / {total_route_minutes:.0f}"
    )
    print(
        f"  ({cut_route_minutes/total_route_minutes*100:.2f}% of operational time)"
    )

    print(f"\n{'='*80}")
    print("BREAKDOWN BY TRAIN TYPE")
    print(f"{'='*80}")
    cuts_by_type = recommended_cuts["train_type"].value_counts()
    total_by_type = scored_trips["train_type"].value_counts()

    breakdown = pd.DataFrame(
        {
            "Total": total_by_type,
            "Cut": cuts_by_type,
            "% Cut": (cuts_by_type / total_by_type * 100).round(1),
        }
    ).fillna(0)

    print(breakdown.sort_values("% Cut", ascending=False))

    print(f"\n{'='*80}")
    print("TOP 20 RECOMMENDED CUTS")
    print(f"{'='*80}")

    display_cols = [
        "trip_short_name",
        "train_type",
        "num_stops",
        "duration_minutes",
        "days_operating",
        "is_peak",
        "redundancy_score",
        "cutting_score",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(recommended_cuts[display_cols].head(20).to_string(index=False))

    print(f"\n{'='*80}")
    print("CHARACTERISTICS OF TRIPS TO CUT")
    print(f"{'='*80}")

    print(f"\nAverage statistics:")
    print(f"  Duration: {recommended_cuts['duration_minutes'].mean():.1f} minutes")
    print(f"  Stops: {recommended_cuts['num_stops'].mean():.1f}")
    print(f"  Days operating: {recommended_cuts['days_operating'].mean():.1f}")
    print(f"  Peak hour trips: {recommended_cuts['is_peak'].sum()} / {len(recommended_cuts)}")
    print(
        f"  Trips with alternatives (redundancy > 0): {(recommended_cuts['redundancy_score'] > 0).sum()}"
    )

    print(f"\n{'='*80}")
    print("IMPACT ANALYSIS")
    print(f"{'='*80}")

    # Check if we're cutting any premium services
    premium_cuts = recommended_cuts[
        recommended_cuts["train_type"].isin(["IC", "EC", "SC", "RJ", "EN"])
    ]
    if len(premium_cuts) > 0:
        print(f"\n⚠ WARNING: {len(premium_cuts)} premium services in cut list:")
        print(premium_cuts[["trip_short_name", "train_type"]].to_string(index=False))
    else:
        print("\n✓ Good: No premium services (IC/EC/SC/RJ/EN) in cut list")

    # Check peak hour impact
    peak_cuts = recommended_cuts[recommended_cuts["is_peak"]]
    print(
        f"\n{'⚠' if len(peak_cuts) > trips_to_cut * 0.3 else '✓'} Peak hour impact: {len(peak_cuts)} / {trips_to_cut} trips ({len(peak_cuts)/trips_to_cut*100:.1f}%)"
    )
    if len(peak_cuts) > trips_to_cut * 0.3:
        print(
            "  Consider: Over 30% of cuts are peak hour services - may impact commuters"
        )

    # Check redundancy
    no_alternative = recommended_cuts[recommended_cuts["redundancy_score"] == 0]
    print(
        f"\n{'⚠' if len(no_alternative) > trips_to_cut * 0.5 else '✓'} Routes with no alternatives: {len(no_alternative)} / {trips_to_cut} trips ({len(no_alternative)/trips_to_cut*100:.1f}%)"
    )
    if len(no_alternative) > trips_to_cut * 0.5:
        print(
            "  Consider: Over 50% of cuts have no alternative services on same route"
        )

    return recommended_cuts


def save_results(recommended_cuts, output_file="output/recommended_cuts.csv"):
    """Save the recommended cuts to a CSV file"""
    import os

    os.makedirs("output", exist_ok=True)

    recommended_cuts.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TASK 4: OPTIMAL TRIP CUTTING ANALYSIS")
    print("Business Problem: Cut 5% of scheduled trips to minimize passenger impact")
    print("=" * 80 + "\n")

    # Load data
    print("Loading data files...")
    stop_times = pd.read_csv("input/stop_times.txt")
    trips = pd.read_csv("input/trips.txt")
    stops = pd.read_csv("input/stops.txt")
    calendar_dates = pd.read_csv("input/calendar_dates.txt")
    print("Data loaded successfully.\n")

    # Step 1: Analyze trip metrics
    trip_metrics = analyze_trip_metrics(stop_times, trips, stops, calendar_dates)

    # Step 2: Calculate station importance
    station_importance = calculate_station_importance(stop_times, trips, calendar_dates)

    # Step 3: Calculate route redundancy
    redundancy = calculate_route_redundancy(trip_metrics)

    # Step 4: Score trips for cutting
    scored_trips = score_trips_for_cutting(trip_metrics, station_importance, redundancy)

    # Step 5: Generate recommendations
    recommended_cuts = recommend_cuts(scored_trips, target_percentage=0.05)

    # Step 6: Save results
    save_results(recommended_cuts)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSee task_4.txt for detailed explanation of the methodology.")
    print()
