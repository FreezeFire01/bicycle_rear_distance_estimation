import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path('/BP1')
OUTPUT_DIR = BASE_DIR / 'yolo_experiments'
RESULTS_DIR = BASE_DIR / 'yolo_comparison'
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = ['nuscenes_only', 'garmin_only', 'combined', 'nuscenes_to_garmin']
MODELS = ['yolo11n', 'yolo11s', 'yolov8n', 'yolo26n']

STRATEGY_LABELS = {
    'nuscenes_only': 'nuScenes',
    'garmin_only': 'Garmin',
    'combined': 'Combined',
    'nuscenes_to_garmin': 'nuScenes→Garmin',
}


def evaluate_model(run_dir, eval_dataset_yaml, imgsz=832):
    from ultralytics import YOLO

    best_pt = run_dir / 'weights' / 'best.pt'
    if not best_pt.exists():
        return None

    model = YOLO(str(best_pt))
    try:
        metrics = model.val(
            data=str(eval_dataset_yaml),
            split='val',
            imgsz=imgsz,
            rect=True,
            verbose=False,
            plots=False,
        )
        return {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'per_class': {
                'car': {
                    'mAP50': float(metrics.box.maps[0]) if len(metrics.box.maps) > 0 else None,
                },
                'truck': {
                    'mAP50': float(metrics.box.maps[1]) if len(metrics.box.maps) > 1 else None,
                },
            }
        }
    except Exception as e:
        print(f"  Chyba: {e}")
        return None


def benchmark_latency(run_dir, imgsz=832, n_warmup=10, n_runs=100):
    from ultralytics import YOLO

    best_pt = run_dir / 'weights' / 'best.pt'
    if not best_pt.exists():
        return None

    model = YOLO(str(best_pt))
    dummy = np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(dummy, imgsz=imgsz, verbose=False)

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy, imgsz=imgsz, verbose=False)
        times.append(time.perf_counter() - t0)

    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'fps': float(1.0 / np.mean(times)),
    }


def main():
    results = []

    for strategy in STRATEGIES:
        for model_name in MODELS:
            run_name = f"{strategy}_{model_name}"
            run_dir = OUTPUT_DIR / run_name

            if not run_dir.exists():
                print(f"  ✗ Chýba: {run_name}")
                continue

            results_csv = run_dir / 'results.csv'
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                last_row = df.iloc[-1]

                row = {
                    'strategy': strategy,
                    'model': model_name,
                    'epochs_trained': int(last_row['epoch']) if 'epoch' in df.columns else None,
                    'train_box_loss': float(last_row.get('train/box_loss', 0)),
                    'val_box_loss': float(last_row.get('val/box_loss', 0)),
                    'mAP50_train': float(last_row.get('metrics/mAP50(B)', 0)),
                    'mAP50-95_train': float(last_row.get('metrics/mAP50-95(B)', 0)),
                    'precision_train': float(last_row.get('metrics/precision(B)', 0)),
                    'recall_train': float(last_row.get('metrics/recall(B)', 0)),
                }
                print(f"{run_name}: mAP50={row['mAP50_train']:.3f}, mAP50-95={row['mAP50-95_train']:.3f}")
            else:
                print(f"Chýba results.csv pre {run_name}")
                continue

            # Benchmark
            bench = benchmark_latency(run_dir)
            if bench:
                row.update({f'bench_{k}': v for k, v in bench.items()})
                print(f"    Latency: {bench['mean_ms']:.1f}ms → {bench['fps']:.1f} FPS (desktop GPU)")

            results.append(row)

    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'yolo_comparison.csv', index=False)
    print(f"\nCSV: {RESULTS_DIR / 'yolo_comparison.csv'}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(STRATEGIES))
    width = 0.25

    for i, model_name in enumerate(MODELS):
        values = []
        for strategy in STRATEGIES:
            row = df[(df['strategy'] == strategy) & (df['model'] == model_name)]
            values.append(row['mAP50_train'].iloc[0] if not row.empty else 0)
        ax.bar(x + i * width - width, values, width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], rotation=15, ha='right')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Porovnanie YOLO modelov naprieč tréningovými stratégiami')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'map50_comparison.png', dpi=150)
    print(f"Graf: {RESULTS_DIR / 'map50_comparison.png'}")

    # Latency vs Accuracy scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for strategy in STRATEGIES:
        strat_df = df[df['strategy'] == strategy]
        if not strat_df.empty:
            ax.scatter(strat_df.get('bench_mean_ms', [0] * len(strat_df)),
                       strat_df['mAP50_train'],
                       s=100, label=STRATEGY_LABELS[strategy], alpha=0.7)
            for _, row in strat_df.iterrows():
                ax.annotate(row['model'],
                            (row.get('bench_mean_ms', 0), row['mAP50_train']),
                            fontsize=8)

    ax.set_xlabel('Latencia (ms)')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Kompromis presnosť vs. latencia')
    ax.legend(title='Stratégia')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'latency_vs_accuracy.png', dpi=150)
    print(f"Graf: {RESULTS_DIR / 'latency_vs_accuracy.png'}")

    print(df.to_string(index=False))


if __name__ == '__main__':
    main()