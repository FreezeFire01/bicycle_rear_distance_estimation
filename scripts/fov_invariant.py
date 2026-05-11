import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path('/home/jozef/Documents/FIIT/5thSemester/BP1')
GARMIN_RF = BASE / 'garmin_dataset' / 'rf_dataset.csv'
NUSCENES_META = BASE / 'bicycle_safety_dataset_final' / 'distance_regression_meta.csv'
BLOCK_SPLIT_DIR = BASE / 'garmin_dataset'  # block-split mapa

MODELS_DIR = BASE / 'final_models'
FIGS_DIR = BASE / 'figures_for_bp'
MODELS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

MAX_DISTANCE = 150

GARMIN_FOV_H = 122
GARMIN_FOV_V = 68.6
GARMIN_W = 1920

NUSCENES_FOV_H = 89
NUSCENES_FOV_V = 58.6
NUSCENES_W = 1600

REAL_CAR_WIDTH_M = 1.8

plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})


def compute_focal_length_from_fov(image_width_px, fov_degrees):
    fov_rad = np.radians(fov_degrees)
    return (image_width_px / 2) / np.tan(fov_rad / 2)


def get_garmin_block_split_map():
    img_to_split = {}
    for split in ['train', 'val', 'test']:
        img_dir = BLOCK_SPLIT_DIR / split / 'images'
        if not img_dir.exists():
            continue
        for img in img_dir.glob('*.jpg'):
            img_to_split[img.stem] = split
    return img_to_split


def prepare_garmin(df):
    out = pd.DataFrame()
    out['w_norm'] = df['w']
    out['h_norm'] = df['h']
    out['area_norm'] = df['area']
    out['aspect_ratio'] = df['aspect_ratio']
    out['y_bottom_norm'] = df['y_bottom']
    out['offset_x_norm'] = df['cx'] - 0.5
    out['w_angle'] = df['w'] * GARMIN_FOV_H
    out['h_angle'] = df['h'] * GARMIN_FOV_V
    out['area_angle'] = out['w_angle'] * out['h_angle']
    out['y_angle'] = (df['y_bottom'] - 0.5) * GARMIN_FOV_V
    out['offset_angle'] = (df['cx'] - 0.5) * GARMIN_FOV_H
    focal = compute_focal_length_from_fov(GARMIN_W, GARMIN_FOV_H)
    out['d_prior'] = (REAL_CAR_WIDTH_M * focal) / (df['w'] * GARMIN_W + 1e-6)
    out['distance_m'] = df['distance_m']
    if 'split' in df.columns:
        out['split'] = df['split']
    return out


def prepare_nuscenes(df):
    out = pd.DataFrame()
    out['w_norm'] = df['w_px'] / NUSCENES_W
    out['h_norm'] = df['h_px'] / 900
    out['area_norm'] = out['w_norm'] * out['h_norm']
    out['aspect_ratio'] = df['w_px'] / df['h_px']
    out['y_bottom_norm'] = 0.5
    out['offset_x_norm'] = df['offset_norm']
    out['w_angle'] = out['w_norm'] * NUSCENES_FOV_H
    out['h_angle'] = out['h_norm'] * NUSCENES_FOV_V
    out['area_angle'] = out['w_angle'] * out['h_angle']
    out['y_angle'] = 0.0
    out['offset_angle'] = df['offset_norm'] * NUSCENES_FOV_H
    focal = compute_focal_length_from_fov(NUSCENES_W, NUSCENES_FOV_H)
    out['d_prior'] = (REAL_CAR_WIDTH_M * focal) / (df['w_px'] + 1e-6)
    out['distance_m'] = df['distance_m']
    out['split'] = df['split']
    return out


def train_eval(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=200, random_state=69, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rf, mae, rmse, y_pred


def main():
    print(f'FOV EXPERIMENT v3 (BLOCK-SPLIT, plny rozsah <={MAX_DISTANCE}m)')


    df_g_raw = pd.read_csv(GARMIN_RF)
    df_n_raw = pd.read_csv(NUSCENES_META)

    print(f'\nPovodne data:')
    print(
        f'  Garmin:   {len(df_g_raw)} vzoriek, range {df_g_raw.distance_m.min():.0f}-{df_g_raw.distance_m.max():.0f}m')
    print(f'  nuScenes: {len(df_n_raw)} vzoriek')

    # Filter na MAX_DISTANCE
    df_g_raw = df_g_raw[df_g_raw.distance_m <= MAX_DISTANCE].reset_index(drop=True)
    df_n_raw = df_n_raw[df_n_raw.distance_m <= MAX_DISTANCE].reset_index(drop=True)

    print(f'\n[1] Mapujem Garmin do block-split (z {BLOCK_SPLIT_DIR.name})')
    img_to_split = get_garmin_block_split_map()

    img_col = None
    for c in ['image', 'image_name', 'filename']:
        if c in df_g_raw.columns:
            img_col = c
            break

    df_g_raw['split'] = df_g_raw[img_col].apply(
        lambda x: img_to_split.get(x.replace('.jpg', '').replace('.png', ''), None)
    )

    n_buffer = df_g_raw.split.isna().sum()
    df_g_raw = df_g_raw[df_g_raw.split.notna()].reset_index(drop=True)

    print(f'   Po block-split: {len(df_g_raw)} vzoriek (vyhodene buffer: {n_buffer})')
    for sp in ['train', 'val', 'test']:
        n = (df_g_raw.split == sp).sum()
        print(f'   {sp}: {n}')

    df_g = prepare_garmin(df_g_raw)
    df_n = prepare_nuscenes(df_n_raw)

    features_pixel = ['w_norm', 'h_norm', 'area_norm', 'aspect_ratio', 'y_bottom_norm', 'offset_x_norm']
    features_fov = ['w_angle', 'h_angle', 'area_angle', 'aspect_ratio', 'y_angle', 'offset_angle']

    train_g = df_g[df_g.split == 'train']
    val_g = df_g[df_g.split == 'val']
    test_g = df_g[df_g.split == 'test']

    print(f'\n[2] Garmin block-split rozdelenie:')
    print(f'   train: {len(train_g)}, val: {len(val_g)}, test: {len(test_g)}')

    X_g_train = train_g[features_pixel]
    X_g_train_fov = train_g[features_fov]
    y_g_train = train_g['distance_m']

    X_g_test = test_g[features_pixel]
    X_g_test_fov = test_g[features_fov]
    y_g_test = test_g['distance_m']

    print(f'\n[3/6] Garmin (PIXEL features) - HLAVNY MODEL')
    rf_gg_pix, mae_gg_pix, rmse_gg_pix, y_pred_gg_pix = train_eval(
        X_g_train, y_g_train, X_g_test, y_g_test)
    joblib.dump(rf_gg_pix, MODELS_DIR / 'rf_garmin_pixel_v3.pkl')
    print(f'   MAE={mae_gg_pix:.2f}m, RMSE={rmse_gg_pix:.2f}m')

    print(f'\n[4/6] Garmin (FOV-invariant features)')
    rf_gg_fov, mae_gg_fov, _, _ = train_eval(
        X_g_train_fov, y_g_train, X_g_test_fov, y_g_test)
    joblib.dump(rf_gg_fov, MODELS_DIR / 'rf_garmin_fov_v3.pkl')
    print(f'   MAE={mae_gg_fov:.2f}m')

    print(f'\n[5/6] nuScenes (PIXEL features)')
    train_n = df_n[df_n.split == 'train']
    val_n = df_n[df_n.split == 'val']
    rf_nn_pix, mae_nn_pix, _, _ = train_eval(
        train_n[features_pixel], train_n['distance_m'],
        val_n[features_pixel], val_n['distance_m'])
    print(f'   MAE={mae_nn_pix:.2f}m')

    print(f'\n[6/6] nuScenes (FOV-invariant features) - CROSS-DATASET MODEL')
    rf_nn_fov, mae_nn_fov, _, _ = train_eval(
        train_n[features_fov], train_n['distance_m'],
        val_n[features_fov], val_n['distance_m'])
    joblib.dump(rf_nn_fov, MODELS_DIR / 'rf_nuscenes_fov_v3.pkl')
    print(f'   MAE={mae_nn_fov:.2f}m')

    print(f'\n=== CROSS-DATASET nuScenes -> Garmin (block-split test) ===')
    y_pred_n2g_pix = rf_nn_pix.predict(X_g_test)
    y_pred_n2g_fov = rf_nn_fov.predict(X_g_test_fov)
    mae_n2g_pix = mean_absolute_error(y_g_test, y_pred_n2g_pix)
    mae_n2g_fov = mean_absolute_error(y_g_test, y_pred_n2g_fov)
    improvement = (mae_n2g_pix - mae_n2g_fov) / mae_n2g_pix * 100

    print(f'   PIXEL features:         MAE = {mae_n2g_pix:.2f}m')
    print(f'   FOV-invariant features: MAE = {mae_n2g_fov:.2f}m')
    print(f'   ZLEPSENIE: {improvement:.1f}%')


    # GRAF 1: Bar comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = ['Garmin->Garmin', 'nuScenes->nuScenes', 'nuScenes->Garmin\n(cross-dataset)']
    pixel_maes = [mae_gg_pix, mae_nn_pix, mae_n2g_pix]
    fov_maes = [mae_gg_fov, mae_nn_fov, mae_n2g_fov]

    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width / 2, pixel_maes, width, label='Pixelove crty', color='steelblue')
    ax.bar(x + width / 2, fov_maes, width, label='FOV-invariantne crty', color='coral')

    for i, (p, f) in enumerate(zip(pixel_maes, fov_maes)):
        ax.text(i - width / 2, p + 0.15, f'{p:.2f}m', ha='center', fontsize=9)
        ax.text(i + width / 2, f + 0.15, f'{f:.2f}m', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('MAE (m)')
    ax.set_title(f'FOV experiment v3 (BLOCK-SPLIT, plny rozsah <={MAX_DISTANCE}m)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'fig_cross_dataset_mae_v3.png', dpi=150)
    plt.close()
    print(f'fig_cross_dataset_mae_v3.png')

    # GRAF 2: Per-bin
    bins = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 140)]
    bin_labels = [f'{lo}-{hi}' for lo, hi in bins]
    y_test_arr = y_g_test.values
    pix_bin_maes, fov_bin_maes, bin_counts = [], [], []
    for lo, hi in bins:
        bmask = (y_test_arr >= lo) & (y_test_arr < hi)
        if bmask.sum() > 0:
            pix_bin_maes.append(mean_absolute_error(y_test_arr[bmask], y_pred_n2g_pix[bmask]))
            fov_bin_maes.append(mean_absolute_error(y_test_arr[bmask], y_pred_n2g_fov[bmask]))
            bin_counts.append(int(bmask.sum()))
        else:
            pix_bin_maes.append(0)
            fov_bin_maes.append(0)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bins))
    ax.bar(x - width / 2, pix_bin_maes, width, label='Pixelove', color='steelblue')
    ax.bar(x + width / 2, fov_bin_maes, width, label='FOV-invariantne', color='coral')
    for i, (p, f, n) in enumerate(zip(pix_bin_maes, fov_bin_maes, bin_counts)):
        ax.text(i - width / 2, p + 0.5, f'{p:.1f}', ha='center', fontsize=9)
        ax.text(i + width / 2, f + 0.5, f'{f:.1f}', ha='center', fontsize=9)
        ax.text(i, -2.0, f'n={n}', ha='center', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Vzdialenostne pasmo (m)')
    ax.set_ylabel('MAE (m)')
    ax.set_title('Per-bin MAE: nuScenes -> Garmin (block-split test)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'fig_cross_dataset_per_bin_v3.png', dpi=150)
    plt.close()

    # GRAF 3: Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = rf_gg_pix.feature_importances_
    sorted_idx = np.argsort(importances)
    ax.barh([features_pixel[i] for i in sorted_idx],
            importances[sorted_idx], color='steelblue')
    for i, v in enumerate(importances[sorted_idx]):
        ax.text(v + 0.005, i, f'{100 * v:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Doleitost')
    ax.set_title('Feature importance - RF Garmin (block-split)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'fig_feature_importance_v3.png', dpi=150)
    plt.close()

    # GRAF 4: Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_g_test, y_pred_gg_pix, alpha=0.4, s=30, color='steelblue', label='Predikcie')
    ax.plot([0, MAX_DISTANCE], [0, MAX_DISTANCE], 'r--', label='Idealna predikcia')
    ax.set_xlabel('Skutocna vzdialenost (m)')
    ax.set_ylabel('Predikovana vzdialenost (m)')
    ax.set_title(f'RF Garmin (block-split) - MAE {mae_gg_pix:.2f}m')
    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_ylim(0, MAX_DISTANCE)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'fig_scatter_predictions_v3.png', dpi=150)
    plt.close()

    info = {
        'version': 'v3 (block-split)',
        'max_distance_m': MAX_DISTANCE,
        'description': 'RF trained on block-split data (no clip-boundary leakage)',
        'garmin_split_source': str(BLOCK_SPLIT_DIR),
        'samples': {
            'garmin_train': len(train_g),
            'garmin_val': len(val_g),
            'garmin_test': len(test_g),
            'nuscenes_train': len(train_n),
            'nuscenes_val': len(val_n),
        },
        'results': {
            'garmin_pixel_mae': float(mae_gg_pix),
            'garmin_pixel_rmse': float(rmse_gg_pix),
            'garmin_fov_mae': float(mae_gg_fov),
            'nuscenes_pixel_mae': float(mae_nn_pix),
            'nuscenes_fov_mae': float(mae_nn_fov),
            'cross_dataset_pixel_mae': float(mae_n2g_pix),
            'cross_dataset_fov_mae': float(mae_n2g_fov),
            'fov_improvement_pct': float(improvement),
        }
    }
    with open(MODELS_DIR / 'model_info_v3.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f'FINALNE VYSLEDKY (block-split, plny rozsah <={MAX_DISTANCE}m)')
    print(f'Garmin->Garmin PIXEL:     MAE {mae_gg_pix:.2f}m')
    print(f'Garmin->Garmin FOV:       MAE {mae_gg_fov:.2f}m')
    print(f'nuScenes->nuScenes PIXEL: MAE {mae_nn_pix:.2f}m')
    print(f'nuScenes->nuScenes FOV:   MAE {mae_nn_fov:.2f}m')
    print(f'nuScenes->Garmin PIXEL:   MAE {mae_n2g_pix:.2f}m')
    print(f'nuScenes->Garmin FOV:     MAE {mae_n2g_fov:.2f}m (zlepsenie {improvement:.1f}%)')


if __name__ == '__main__':
    main()