# pip install fitparse pandas

import fitparse
import pandas as pd
from datetime import timezone


def fit_to_csv(fit_path, csv_path):
    fitfile = fitparse.FitFile(fit_path)

    records = []
    for record in fitfile.get_messages():
        row = {"message_type": record.name}
        for field in record:
            row[field.name] = field.value
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Uložené: {csv_path}")
    print(df["message_type"].value_counts())
    return df


df = fit_to_csv("/GARMIN_CSV/2026-04-12-11-21-54.fit",
                "vyuzitie_neskor/activity_20260412.csv")