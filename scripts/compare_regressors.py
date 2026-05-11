from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path('/home/jozef/Documents/FIIT/5thSemester/BP1')
ORIG_RF_CSV = BASE / 'garmin_dataset' / 'rf_dataset.csv'
BLOCK_SPLIT_DIR = BASE / 'garmin_dist_dataset'
OUTPUT_DIR = BASE / 'regressor_comparison'
OUTPUT_DIR.mkdir(exist_ok=True)


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
    print('POROVNANIE REGRESOROV NA BLOCK-SPLIT DATASETE')


    df = pd.read_csv(ORIG_RF_CSV)
    print(f'Povodny rf_dataset: {len(df)} riadkov, range {df.distance_m.min():.0f}-{df.distance_m.max():.0f}m')

    # Mapuj na block-split
    img_to_split = get_image_to_split_map()

    img_col = None
    for c in ['image', 'image_name', 'filename', 'img', 'frame']:
        if c in df.columns:
            img_col = c
            break

    df['new_split'] = df[img_col].apply(
        lambda x: img_to_split.get(x.replace('.jpg', '').replace('.png', ''), None)
    )

    df = df[df.new_split.notna()].copy()
    print(f'Po block-split mapovani: {len(df)} riadkov')

    train_df = df[df.new_split == 'train']
    val_df = df[df.new_split == 'val']
    test_df = df[df.new_split == 'test']

    print(f'  Train: {len(train_df)}')
    print(f'  Val:   {len(val_df)}')
    print(f'  Test:  {len(test_df)}')

    # Features
    feat_cols = ['cx', 'cy', 'w', 'h', 'area', 'aspect_ratio', 'y_bottom']
    X_train = train_df[feat_cols].values
    y_train = train_df['distance_m'].values
    X_val = val_df[feat_cols].values
    y_val = val_df['distance_m'].values
    X_test = test_df[feat_cols].values
    y_test = test_df['distance_m'].values

    # Modely
    MODELS = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10),
        'SVR (RBF)': SVR(kernel='rbf', C=10),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20,
                                                random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200,
                                                        random_state=42),
    }

    print(f'{"Model":<22} | {"≤50m":>8} | {"≤60m":>8} | {"≤90m":>8} | {"Cely":>8}')


    results_data = {}

    for name, model in MODELS.items():
        # Trenuj na train
        model.fit(X_train, y_train)
        # Predikuj na test
        y_pred_test = model.predict(X_test)

        # MAE pre rozne rozsahy
        results_data[name] = {}
        row_str = f'{name:<22} |'
        for max_d in [50, 60, 90, 999]:
            mask = y_test <= max_d
            if mask.sum() > 0:
                mae = mean_absolute_error(y_test[mask], y_pred_test[mask])
                results_data[name][f'≤{max_d}m'] = mae
                row_str += f' {mae:>7.2f}m |'
            else:
                row_str += f' {"-":>8} |'
        print(row_str)

    rows = []
    for name, vals in results_data.items():
        rows.append({'Model': name, **vals})
    df_results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'comparison_block_split.csv'
    df_results.to_csv(csv_path, index=False)
    print(f'\nCSV: {csv_path}')

if __name__ == '__main__':
    main()