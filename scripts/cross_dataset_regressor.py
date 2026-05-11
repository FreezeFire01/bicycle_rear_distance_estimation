import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

GARMIN_RF = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/rf_dataset.csv'
NUSCENES_META = '/home/jozef/Documents/FIIT/5thSemester/BP1/bicycle_safety_dataset_final/distance_regression_meta.csv'
OUTPUT_DIR = '/home/jozef/Documents/FIIT/5thSemester/BP1/cross_dataset_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image dimensions
GARMIN_W, GARMIN_H = 1920, 1080
NUSCENES_W, NUSCENES_H = 1600, 900


def prepare_garmin(df):
    # Garmin features sú už normalizované (cx, cy, w, h, area, aspect_ratio, y_bottom)
    out = pd.DataFrame({
        'w_norm': df['w'],
        'h_norm': df['h'],
        'area_norm': df['area'],
        'aspect_ratio': df['aspect_ratio'],
        'y_bottom_norm': df['y_bottom'],
        'offset_x_norm': df['cx'] - 0.5,
        'distance_m': df['distance_m'],
        'source': 'garmin',
    })
    return out


def prepare_nuscenes(df):
    out = pd.DataFrame({
        'w_norm': df['w_px'] / NUSCENES_W,
        'h_norm': df['h_px'] / NUSCENES_H,
        'area_norm': (df['w_px'] / NUSCENES_W) * (df['h_px'] / NUSCENES_H),
        'aspect_ratio': df['w_px'] / df['h_px'],
        # nuScenes nemá y_bottom ale má offset_norm — odhadneme y_bottom podľa distance (blízke autá nižšie)
        # Ak nemáme y_bottom, použijeme 0.5 (stred obrazu)
        'y_bottom_norm': 0.5,
        'offset_x_norm': df['offset_norm'],
        'distance_m': df['distance_m'],
        'source': 'nuscenes',
        'split': df['split'],
    })
    return out


def train_and_evaluate(X_train, y_train, X_test, y_test, name=""):
    """Natrénuj RF na train, vyhodnoť na test."""
    rf = RandomForestRegressor(n_estimators=200, random_state=69, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Per-bin MAE
    bins = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 90), (90, 120), (120, 200)]
    bin_results = {}
    for lo, hi in bins:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            bin_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            bin_results[f'{lo}-{hi}m'] = {'mae': bin_mae, 'n': int(mask.sum())}

    print(f"  {name}: MAE={mae:.2f}m, RMSE={rmse:.2f}m")
    return {
        'mae': mae, 'rmse': rmse, 'y_pred': y_pred,
        'y_test': y_test, 'bin_results': bin_results,
    }


def main():
    df_garmin_raw = pd.read_csv(GARMIN_RF)
    df_nu_raw = pd.read_csv(NUSCENES_META)

    df_garmin = prepare_garmin(df_garmin_raw)
    df_nu = prepare_nuscenes(df_nu_raw)

    features = ['w_norm', 'h_norm', 'area_norm', 'aspect_ratio', 'y_bottom_norm', 'offset_x_norm']

    print(f"  Garmin: {len(df_garmin)} vzoriek")
    print(
        f"  nuScenes: {len(df_nu)} vzoriek (train: {(df_nu['split'] == 'train').sum()}, val: {(df_nu['split'] == 'val').sum()})")
    print()

    # Garmin
    X_g = df_garmin[features]
    y_g = df_garmin['distance_m']
    X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(X_g, y_g, test_size=0.2, random_state=69)

    # nuScenes má fixný split
    X_n_train = df_nu[df_nu['split'] == 'train'][features]
    y_n_train = df_nu[df_nu['split'] == 'train']['distance_m']
    X_n_test = df_nu[df_nu['split'] == 'val'][features]
    y_n_test = df_nu[df_nu['split'] == 'val']['distance_m']

    # Combined
    X_comb_train = pd.concat([X_g_train, X_n_train], ignore_index=True)
    y_comb_train = pd.concat([y_g_train, y_n_train], ignore_index=True)

    results = {}

    print("\n[1] BASELINES (train/test rovnaký dataset):")
    results['Garmin→Garmin'] = train_and_evaluate(X_g_train, y_g_train, X_g_test, y_g_test, 'Garmin→Garmin')
    results['nuScenes→nuScenes'] = train_and_evaluate(X_n_train, y_n_train, X_n_test, y_n_test, 'nuScenes→nuScenes')

    print("\n[2] CROSS-DATASET (train/test rôzny dataset):")
    # Garmin → nuScenes: treba nuscenes len do rozsahu garmina
    nu_mask_valid = y_n_test <= df_garmin['distance_m'].max()
    results['Garmin→nuScenes'] = train_and_evaluate(X_g_train, y_g_train, X_n_test[nu_mask_valid],
                                                    y_n_test[nu_mask_valid], 'Garmin→nuScenes')

    # nuScenes → Garmin: treba garmin len do rozsahu nuscenes
    g_mask_valid = y_g_test <= df_nu['distance_m'].max()
    results['nuScenes→Garmin'] = train_and_evaluate(X_n_train, y_n_train, X_g_test[g_mask_valid],
                                                    y_g_test[g_mask_valid], 'nuScenes→Garmin')

    print("\n[3] COMBINED TRAIN (Garmin + nuScenes):")
    results['Combined→Garmin'] = train_and_evaluate(X_comb_train, y_comb_train, X_g_test, y_g_test, 'Combined→Garmin')
    results['Combined→nuScenes'] = train_and_evaluate(X_comb_train, y_comb_train, X_n_test, y_n_test,
                                                      'Combined→nuScenes')

    # === GRAFY ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, r) in zip(axes.flat, results.items()):
        ax.scatter(r['y_test'], r['y_pred'], alpha=0.3, s=5)
        max_val = max(r['y_test'].max(), r['y_pred'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Ideál')
        ax.set_xlabel('Skutočná vzdialenosť (m)')
        ax.set_ylabel('Predikovaná vzdialenosť (m)')
        ax.set_title(f"{name}\nMAE={r['mae']:.2f}m")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cross_dataset_scatter.png'), dpi=150)
    print(f"Grafy: {OUTPUT_DIR}/cross_dataset_scatter.png")

    summary = pd.DataFrame([
        {'scenario': k, 'mae': r['mae'], 'rmse': r['rmse']}
        for k, r in results.items()
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, 'cross_dataset_summary.csv'), index=False)
    print(f"\n{summary.to_string(index=False)}")



if __name__ == '__main__':
    main()