import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path('/BP1')
GARMIN_CSV = BASE / 'garmin_dataset' / 'rf_dataset.csv'
NUSC_CSV = BASE / 'bicycle_safety_dataset_final' / 'distance_regression_meta.csv'

# Block-split adresare (na zistenie ktore obrazky su v ktorom splite)
BLOCK_SPLIT_DIR = BASE / 'garmin_dataset_v2'


def get_block_split_images():
    images = set()
    for split in ['train', 'val', 'test']:
        img_dir = BLOCK_SPLIT_DIR / split / 'images'
        if not img_dir.exists():
            continue
        for img in img_dir.glob('*.jpg'):
            images.add(img.stem)
    return images


# Nacitaj data
df_g = pd.read_csv(GARMIN_CSV)
df_n = pd.read_csv(NUSC_CSV)

print(f'Povodny Garmin rf_dataset: {len(df_g)} rámikov')

block_split_images = get_block_split_images()
print(f'Block-split obrázky (bez buffer): {len(block_split_images)}')

# Stlpec s nazvom obrazka
img_col = 'image' if 'image' in df_g.columns else 'image_name'

df_g['stem'] = df_g[img_col].apply(lambda x: x.replace('.jpg', '').replace('.png', ''))
df_g_filtered = df_g[df_g['stem'].isin(block_split_images)].copy()

print(f'Po block-split filtri: {len(df_g_filtered)} rámikov')
print(f'Vyhodené buffer rámiky: {len(df_g) - len(df_g_filtered)}')

# Setup figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

# Bins
bins = np.arange(0, 145, 5)

# Garmin histogram (po block-split)
ax = axes[0]
ax.hist(df_g_filtered['distance_m'], bins=bins, color='#2E86AB',
        edgecolor='white', alpha=0.85)
ax.set_xlabel('Vzdialenosť (m)', fontsize=11)
ax.set_ylabel('Počet rámikov', fontsize=11)
ax.set_title(f'Garmin (n={len(df_g_filtered):,})', fontsize=12, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Bezpečnostný rozsah (50 m)')
ax.axvline(90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Dist-YOLO max (90 m)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# nuScenes histogram (zostava povodny)
ax = axes[1]
ax.hist(df_n['distance_m'], bins=bins, color='#A23B72',
        edgecolor='white', alpha=0.85)
ax.set_xlabel('Vzdialenosť (m)', fontsize=11)
ax.set_ylabel('Počet rámikov', fontsize=11)
ax.set_title(f'nuScenes (n={len(df_n):,})', fontsize=12, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Bezpečnostný rozsah (50 m)')
ax.axvline(90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Dist-YOLO max (90 m)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
out = BASE / 'figures_for_bp' / 'fig_distance_distribution.png'
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.show()


# Statistiky pre kontrolu
print(f'Garmin (po block-split):')
print(f'  Total: {len(df_g_filtered)}')
print(f'  Range: {df_g_filtered.distance_m.min():.1f} - {df_g_filtered.distance_m.max():.1f}m')
print(f'  Median: {df_g_filtered.distance_m.median():.1f}m')
print(f'  ≤50m: {(df_g_filtered.distance_m <= 50).sum()} ({100*(df_g_filtered.distance_m <= 50).mean():.1f}%)')
print(f'  ≤90m: {(df_g_filtered.distance_m <= 90).sum()} ({100*(df_g_filtered.distance_m <= 90).mean():.1f}%)')

print(f'\nnuScenes:')
print(f'  Total: {len(df_n)}')
print(f'  Range: {df_n.distance_m.min():.1f} - {df_n.distance_m.max():.1f}m')
print(f'  ≤50m: {(df_n.distance_m <= 50).sum()} ({100*(df_n.distance_m <= 50).mean():.1f}%)')