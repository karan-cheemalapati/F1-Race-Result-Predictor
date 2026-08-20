import pandas as pd
import os

# Path to the data folder (works regardless of where you run the script from)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def load_raw_data():
    """
    Load all 11 CSV files into a dictionary of DataFrames.
    '\\N' is the null marker used in this dataset.
    """
    files = {
        'circuits':              'circuits.csv',
        'constructors':          'constructors.csv',
        'constructor_standings': 'constructor_standings.csv',
        'driver_standings':      'driver_standings.csv',
        'drivers':               'drivers.csv',
        'lap_times':             'lap_times.csv',
        'pit_stops':             'pit_stops.csv',
        'qualifying':            'qualifying.csv',
        'races':                 'races.csv',
        'results':               'results.csv',
        'status':                'status.csv',
    }

    data = {}
    for key, filename in files.items():
        filepath = os.path.join(DATA_PATH, filename)
        df = pd.read_csv(filepath, na_values=['\\N', 'N', ''])
        data[key] = df
        print(f"{key:<25} {df.shape[0]:>7,} rows  {df.shape[1]:>2} cols")

    print(f"\n  All files loaded successfully.\n")
    return data


def merge_core(data):
    """
    Build the base flat table by merging results with races,
    circuits, drivers, constructors, and status.
    """
    df = data['results'].copy()

    # races — year, round, circuit, date
    races_cols = ['raceId', 'year', 'round', 'circuitId', 'name', 'date']
    df = df.merge(data['races'][races_cols], on='raceId', how='left')

    # circuits
    circuits_cols = ['circuitId', 'circuitRef', 'country', 'lat', 'lng', 'alt']
    df = df.merge(data['circuits'][circuits_cols], on='circuitId', how='left')

    # drivers
    drivers_cols = ['driverId', 'driverRef', 'dob', 'nationality']
    df = df.merge(data['drivers'][drivers_cols], on='driverId', how='left')

    # constructors
    constructors_cols = ['constructorId', 'constructorRef', 'nationality']
    df = df.merge(
        data['constructors'][constructors_cols],
        on='constructorId',
        how='left',
        suffixes=('_driver', '_constructor')
    )

    # status (maps statusId → "Finished", "Engine", "Accident", etc.)
    df = df.merge(data['status'][['statusId', 'status']], on='statusId', how='left')

    # --- derived columns ---
    df['date'] = pd.to_datetime(df['date'])
    df['dob']  = pd.to_datetime(df['dob'], errors='coerce')
    df['driver_age'] = ((df['date'] - df['dob']).dt.days / 365.25).round(1)

    # DNF flag
    finished_statuses = [
        'Finished', '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps',
        '+5 Laps', '+6 Laps', '+7 Laps', '+8 Laps', '+9 Laps'
    ]
    df['is_dnf'] = (~df['status'].isin(finished_statuses)).astype(int)

    # targets
    df['podium']    = (df['positionOrder'] <= 3).astype(int)
    df['in_points'] = (df['points'] > 0).astype(int)

    return df


def merge_standings(df, data):
    """Join driver and constructor championship standings."""

    ds = data['driver_standings'][
        ['raceId', 'driverId', 'points', 'position', 'wins']
    ].copy()
    ds.columns = [
        'raceId', 'driverId',
        'driver_standing_pts', 'driver_standing_pos', 'driver_wins_so_far'
    ]
    df = df.merge(ds, on=['raceId', 'driverId'], how='left')

    cs = data['constructor_standings'][
        ['raceId', 'constructorId', 'points', 'position', 'wins']
    ].copy()
    cs.columns = [
        'raceId', 'constructorId',
        'constructor_standing_pts', 'constructor_standing_pos', 'constructor_wins_so_far'
    ]
    df = df.merge(cs, on=['raceId', 'constructorId'], how='left')

    return df


def merge_qualifying(df, data):
    """
    Join qualifying data. Convert lap time strings (m:ss.mmm) to seconds.
    Compute each driver's gap to pole position.
    """
    def to_seconds(t):
        try:
            parts = str(t).split(':')
            return float(parts[0]) * 60 + float(parts[1])
        except Exception:
            return None

    q = data['qualifying'].copy()
    for col in ['q1', 'q2', 'q3']:
        q[f'{col}_sec'] = q[col].apply(to_seconds)

    q['best_quali_sec'] = q[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)

    # pole time = fastest quali time in that race
    pole = (
        q.groupby('raceId')['best_quali_sec']
        .min()
        .reset_index()
        .rename(columns={'best_quali_sec': 'pole_time_sec'})
    )
    q = q.merge(pole, on='raceId', how='left')
    q['quali_gap_to_pole'] = q['best_quali_sec'] - q['pole_time_sec']

    keep = ['raceId', 'driverId', 'position', 'best_quali_sec', 'quali_gap_to_pole']
    q = q[keep].rename(columns={'position': 'quali_position'})

    df = df.merge(q, on=['raceId', 'driverId'], how='left')
    return df


def merge_pit_stops(df, data):
    """Add total pit stop count per driver per race."""
    pit = (
        data['pit_stops']
        .groupby(['raceId', 'driverId'])
        .size()
        .reset_index(name='pit_stop_count')
    )
    df = df.merge(pit, on=['raceId', 'driverId'], how='left')
    df['pit_stop_count'] = df['pit_stop_count'].fillna(0).astype(int)
    return df


def build_dataset(data):
    """
    Full pipeline: merge all sources into one analysis-ready DataFrame
    and save to data/f1_merged.csv.
    """
    print("--- Building merged dataset ---\n")

    df = merge_core(data)
    print(f"  After core merge:        {df.shape}")

    df = merge_standings(df, data)
    print(f"  After standings merge:   {df.shape}")

    df = merge_qualifying(df, data)
    print(f"  After qualifying merge:  {df.shape}")

    df = merge_pit_stops(df, data)
    print(f"  After pit stops merge:   {df.shape}")

    df = df.sort_values(['year', 'round', 'positionOrder']).reset_index(drop=True)

    print(f"  Final shape:    {df.shape}")
    print(f"  Years:          {df['year'].min()} – {df['year'].max()}")
    print(f"  Unique races:   {df['raceId'].nunique():,}")
    print(f"  Unique drivers: {df['driverId'].nunique():,}")
    print(f"  Podium rate:    {df['podium'].mean():.1%}")
    print(f"  DNF rate:       {df['is_dnf'].mean():.1%}")
    print(f"  Points rate:    {df['in_points'].mean():.1%}")

    out_path = os.path.join(DATA_PATH, 'f1_merged.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved → data/f1_merged.csv\n")

    return df


if __name__ == '__main__':
    data = load_raw_data()
    df   = build_dataset(data)
