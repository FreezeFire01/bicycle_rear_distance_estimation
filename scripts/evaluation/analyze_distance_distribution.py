import pandas as pd
from pathlib import Path

BASE = Path('/BP1')

# Cesty k metadátam
GARMIN_CSV = BASE / 'garmin_dataset' / 'rf_dataset.csv'
NUSC_CSV = BASE / 'bicycle_safety_dataset_final' / 'distance_regression_meta.csv'


def analyze(name, df, dist_col='distance_m'):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  N samples: {len(df)}")
    print(f"  Min:       {df[dist_col].min():.1f} m")
    print(f"  Max:       {df[dist_col].max():.1f} m")
    print(f"  Mean:      {df[dist_col].mean():.1f} m")
    print(f"  Median:    {df[dist_col].median():.1f} m")
    print(f"  P90:       {df[dist_col].quantile(0.90):.1f} m")
    print(f"  P95:       {df[dist_col].quantile(0.95):.1f} m")
    print(f"  P99:       {df[dist_col].quantile(0.99):.1f} m")

    print(f"\n  Distribúcia po pásmach:")
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 200)]
    cumulative = 0
    for lo, hi in bins:
        n = ((df[dist_col] >= lo) & (df[dist_col] < hi)).sum()
        pct = n / len(df) * 100
        cumulative += pct
        bar = '█' * int(pct / 2)
        print(f"    {lo:3d}-{hi:3d}m: {n:6d} ({pct:5.1f}%) cum={cumulative:5.1f}%  {bar}")

    print(f"\n  Strata pri rôznych MAX_DISTANCE:")
    for max_d in [80, 90, 100, 120, 140]:
        n_lost = (df[dist_col] > max_d).sum()
        pct_lost = n_lost / len(df) * 100
        marker = 'odporúča sa' if pct_lost < 1.0 and pct_lost > 0 else ''
        print(f"    MAX={max_d}m → strata {n_lost} samples ({pct_lost:.2f}%){marker}")


def main():
    # GARMIN
    if GARMIN_CSV.exists():
        df_g = pd.read_csv(GARMIN_CSV)
        analyze('GARMIN dataset', df_g)
    else:
        print(f"Garmin CSV nenajdene: {GARMIN_CSV}")

    # NUSCENES
    if NUSC_CSV.exists():
        df_n = pd.read_csv(NUSC_CSV)
        analyze('NUSCENES dataset', df_n)
    else:
        print(f"nuScenes CSV nenajdene: {NUSC_CSV}")


if __name__ == '__main__':
    main()