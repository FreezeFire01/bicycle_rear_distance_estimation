import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path('/home/jozef/Documents/FIIT/5thSemester/BP1')
ORIG_RF_CSV = BASE / 'garmin_dataset' / 'rf_dataset.csv'

BLOCK_SPLIT_DIR = BASE / 'garmin_dataset'


def get_image_to_split_map():
    img_to_split = {}
    for split in ['train', 'val', 'test']:
        img_dir = BLOCK_SPLIT_DIR / split / 'images'
        if not img_dir.exists():
            continue
        for img in img_dir.glob('*.jpg'):
            img_to_split[img.stem] = split
    return img_to_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='rf_garmin_full_v3.pkl')
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=20)
    args = parser.parse_args()


    if not ORIG_RF_CSV.exists():
        print(f'Povodny rf_dataset.csv: {ORIG_RF_CSV}')
        return

    df_orig = pd.read_csv(ORIG_RF_CSV)
    print(f'\nPovodny rf_dataset: {len(df_orig)} riadkov')
    print(f'Distance range: {df_orig.distance_m.min():.1f}m - {df_orig.distance_m.max():.1f}m')

    img_to_split = get_image_to_split_map()
    print(f'\nBlock-split mapa (z {BLOCK_SPLIT_DIR.name}): {len(img_to_split)} obrazkov')

    img_col = None
    for col in ['image', 'image_name', 'filename', 'img', 'frame']:
        if col in df_orig.columns:
            img_col = col
            break

    df_orig['new_split'] = df_orig[img_col].apply(
        lambda x: img_to_split.get(x.replace('.jpg', '').replace('.png', ''), None)
    )

    print(f'\nMapovanie:')
    for split in ['train', 'val', 'test']:
        n = (df_orig.new_split == split).sum()
        print(f'  {split}: {n}')
    n_buffer = df_orig.new_split.isna().sum()
    print(f'  buffer/vyhodene: {n_buffer}')

    df = df_orig[df_orig.new_split.notna()].copy()
    print(f'\nPo block-split: {len(df)} riadkov')

    feat_cols = ['cx', 'cy', 'w', 'h', 'area', 'aspect_ratio', 'y_bottom']

    train_df = df[df.new_split == 'train']
    val_df = df[df.new_split == 'val']
    test_df = df[df.new_split == 'test']

    print(f'\nTrenovacie samples: {len(train_df)}')
    print(f'Val samples:        {len(val_df)}')
    print(f'Test samples:       {len(test_df)}')

    X_train = train_df[feat_cols].values
    y_train = train_df['distance_m'].values

    print(f'\nTrenujem RF...')
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Eval
    print('VAL RESULTS')
    X_val = val_df[feat_cols].values
    y_val = val_df['distance_m'].values
    print(f'{"Range":<15} {"MAE":<10} {"RMSE":<10} {"n":<10}')
    for max_d in [50, 60, 90, 140, 200]:
        mask = y_val <= max_d
        if mask.sum() == 0:
            continue
        y_pred = rf.predict(X_val[mask])
        mae = mean_absolute_error(y_val[mask], y_pred)
        rmse = np.sqrt(mean_squared_error(y_val[mask], y_pred))
        print(f'<={max_d}m       {mae:>6.2f}m   {rmse:>6.2f}m   {int(mask.sum())}')

    print('TEST RESULTS')

    X_test = test_df[feat_cols].values
    y_test = test_df['distance_m'].values
    print(f'{"Range":<15} {"MAE":<10} {"RMSE":<10} {"n":<10}')

    for max_d in [50, 60, 90, 140, 200]:
        mask = y_test <= max_d
        if mask.sum() == 0:
            continue
        y_pred = rf.predict(X_test[mask])
        mae = mean_absolute_error(y_test[mask], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred))
        print(f'<={max_d}m       {mae:>6.2f}m   {rmse:>6.2f}m   {int(mask.sum())}')

    # Per-bin
    print('\nPer-bin MAE (test):')
    y_pred_test = rf.predict(X_test)
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 90), (90, 140), (140, 200)]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            bin_mae = mean_absolute_error(y_test[mask], y_pred_test[mask])
            print(f'  {lo}-{hi}m: MAE={bin_mae:.2f}m (n={int(mask.sum())})')

    print(f'\nFeature importance:')
    for fc, imp in sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f'  {fc:<14s}: {imp * 100:>5.1f}% {bar}')

    joblib.dump(rf, args.output)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()