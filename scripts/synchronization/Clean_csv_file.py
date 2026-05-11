#!/usr/bin/env python3
"""
python Clean_csv_file.py activity_202xxxxx.csv
"""

import csv
import sys
import os


INPUT = sys.argv[1]
OUTPUT = INPUT.replace('.csv', '_clean.csv')

with open(INPUT, 'r') as f_in, open(OUTPUT, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    header = next(reader)

    idx = {col: header.index(col) for col in [
        'timestamp', 'enhanced_speed', 'radar_current', 'radar_ranges',
        'radar_speeds', 'passing_speed', 'passing_speedabs',
        'position_lat', 'position_long', 'distance',
    ]}

    out_fields = ['timestamp', 'speed_kmh', 'num_cars', 'closest_m',
                  'radar_ranges', 'passing_speedabs', 'lat', 'lon', 'distance_m']

    writer = csv.DictWriter(f_out, fieldnames=out_fields)
    writer.writeheader()

    count = 0
    for row in reader:
        if row[0] != 'record':
            continue
        ts = row[idx['timestamp']]
        if not ts:
            continue

        ranges = []
        for val in row[idx['radar_ranges']].strip('()').split(','):
            val = val.strip()
            if val not in ('None', ''):
                try: ranges.append(int(float(val)))
                except: ranges.append(0)
            else: ranges.append(0)
        non_zero = [r for r in ranges if r > 0]

        spd = row[idx['enhanced_speed']]
        lat_raw = row[idx['position_lat']]
        lon_raw = row[idx['position_long']]

        def to_dms(decimal_deg):
            d = int(decimal_deg)
            m_float = abs(decimal_deg - d) * 60
            m = int(m_float)
            s = (m_float - m) * 60
            return f"{d}°{m}'{s:.2f}\""

        lat_dec = float(lat_raw) * 180 / 2**31 if lat_raw else None
        lon_dec = float(lon_raw) * 180 / 2**31 if lon_raw else None

        writer.writerow({
            'timestamp': ts,
            'speed_kmh': round(float(spd) * 3.6, 1) if spd else '',
            'num_cars': len(non_zero),
            'closest_m': min(non_zero) if non_zero else '',
            'radar_ranges': str(non_zero) if non_zero else '[]',
            'passing_speedabs': row[idx['passing_speedabs']] or '',
            'lat': to_dms(lat_dec) if lat_dec else '',
            'lon': to_dms(lon_dec) if lon_dec else '',
            'distance_m': row[idx['distance']] or '',
        })
        count += 1

print(f"{count} záznamov → {OUTPUT}")