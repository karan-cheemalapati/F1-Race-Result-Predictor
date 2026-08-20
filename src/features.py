import pandas as pd
import numpy as np


def add_circuit_history(df):
    """
    For each driver at each race, compute their historical
    performance at that specific circuit BEFORE this race.
    - win rate at circuit
    - podium rate at circuit
    - avg finish position at circuit
    - number of times raced at circuit
    """
    df = df.sort_values(['year', 'round']).copy()

    records = []

    for (driverId, circuitId), group in df.groupby(['driverId', 'circuitId']):
        group = group.sort_values(['year', 'round'])

        win_rate        = []
        podium_rate     = []
        avg_finish      = []
        races_at_circuit = []

        history_wins   = []
        history_podiums = []
        history_finish  = []

        for _, row in group.iterrows():
            n = len(history_wins)
            win_rate.append(np.mean(history_wins)   if n > 0 else np.nan)
            podium_rate.append(np.mean(history_podiums) if n > 0 else np.nan)
            avg_finish.append(np.mean(history_finish)   if n > 0 else np.nan)
            races_at_circuit.append(n)

            history_wins.append(1 if row['positionOrder'] == 1 else 0)
            history_podiums.append(row['podium'])
            history_finish.append(row['positionOrder'])

        group = group.copy()
        group['circuit_win_rate']      = win_rate
        group['circuit_podium_rate']   = podium_rate
        group['circuit_avg_finish']    = avg_finish
        group['circuit_races_count']   = races_at_circuit
        records.append(group)

    result = pd.concat(records).sort_values(['year', 'round', 'positionOrder'])
    return result


def add_rolling_form(df, windows=[3, 5]):
    """
    Rolling average finish position for each driver
    over the last N races (form going into the race).
    """
    df = df.sort_values(['driverId', 'year', 'round']).copy()

    for w in windows:
        col_name = f'driver_avg_finish_last{w}'
        df[col_name] = (
            df.groupby('driverId')['positionOrder']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    return df


def add_constructor_form(df, windows=[3, 5]):
    """
    Rolling average points per race for each constructor
    over the last N races.
    """
    df = df.sort_values(['constructorId', 'year', 'round']).copy()

    # constructor points per race (sum both drivers)
    constructor_pts = (
        df.groupby(['constructorId', 'raceId'])['points']
        .sum()
        .reset_index()
        .rename(columns={'points': 'constructor_race_pts'})
        .sort_values(['constructorId', 'raceId'])
    )

    for w in windows:
        col_name = f'constructor_avg_pts_last{w}'
        constructor_pts[col_name] = (
            constructor_pts.groupby('constructorId')['constructor_race_pts']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    df = df.merge(
        constructor_pts[['constructorId', 'raceId'] +
                        [f'constructor_avg_pts_last{w}' for w in windows]],
        on=['constructorId', 'raceId'],
        how='left'
    )

    return df


def add_driver_experience(df):
    """
    Total number of races a driver has started BEFORE this race.
    Experience matters — rookies perform differently.
    """
    df = df.sort_values(['driverId', 'year', 'round']).copy()
    df['driver_experience'] = (
        df.groupby('driverId').cumcount()
    )
    return df


def add_grid_vs_quali(df):
    """
    Difference between grid position and qualifying position.
    A positive value means the driver dropped back (penalty, etc).
    A negative value means they moved forward.
    """
    df = df.copy()
    df['grid_vs_quali'] = df['grid'] - df['quali_position']
    return df


def add_season_progress(df):
    """
    How far through the season is this race?
    Early season vs late season dynamics differ.
    """
    df = df.copy()
    season_length = df.groupby('year')['round'].transform('max')
    df['season_progress'] = df['round'] / season_length
    return df


def add_points_gap(df):
    """
    Gap in championship points between this driver and the leader.
    Drivers chasing the championship take different risks.
    """
    df = df.copy()

    leader_pts = (
        df.groupby('raceId')['driver_standing_pts']
        .max()
        .reset_index()
        .rename(columns={'driver_standing_pts': 'leader_pts'})
    )
    df = df.merge(leader_pts, on='raceId', how='left')
    df['pts_gap_to_leader'] = df['leader_pts'] - df['driver_standing_pts']
    df.drop(columns=['leader_pts'], inplace=True)

    return df


def build_features(df):
    """
    Full feature engineering pipeline.
    Applies all feature functions in order.
    """
    print("=== Feature Engineering ===\n")

    print("  Adding circuit history...")
    df = add_circuit_history(df)

    print("  Adding rolling driver form...")
    df = add_rolling_form(df, windows=[3, 5])

    print("  Adding constructor form...")
    df = add_constructor_form(df, windows=[3, 5])

    print("  Adding driver experience...")
    df = add_driver_experience(df)

    print("  Adding grid vs qualifying delta...")
    df = add_grid_vs_quali(df)

    print("  Adding season progress...")
    df = add_season_progress(df)

    print("  Adding points gap to leader...")
    df = add_points_gap(df)

    print(f"\n  Done! Shape: {df.shape}")
    print(f"  New features added: circuit_win_rate, circuit_podium_rate,")
    print(f"  circuit_avg_finish, circuit_races_count, driver_avg_finish_last3,")
    print(f"  driver_avg_finish_last5, constructor_avg_pts_last3,")
    print(f"  constructor_avg_pts_last5, driver_experience,")
    print(f"  grid_vs_quali, season_progress, pts_gap_to_leader\n")

    return df


if __name__ == '__main__':
    from src.data_loader import load_raw_data, build_dataset
    import os

    data = load_raw_data()
    df   = build_dataset(data)
    df   = build_features(df)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', 'f1_features.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved -> data/f1_features.csv")