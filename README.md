# GroundWater Table Statistics Tool (GWTST)

A small Windows GUI tool to process WRIS groundwater time-series Excel files downloaded from the India WRIS platform ( https://indiawris.gov.in/wris/), compute seasonal statistics, and perform Mann–Kendall (MK) trend analysis with Sen's slope. Note that the Excel format is hard-coded for WRIS only.

## Overview
GWTST reads Excel files whose filename contains "Ground Water Level", extracts location and time-series data, removes outliers, and produces CSV outputs:
- Well_locations.csv — header/location info per well
- well_data_all.csv — cleaned time-series for all wells
- GWT_mean_fluctuations.csv — average pre-monsoon / post-monsoon values and mean fluctuation
- GWT_trends.csv — MK trend results and Sen's slope (m/yr) for pre-monsoon, post-monsoon and overall

## Features
- GUI to select input and output directories
- Button 1: Process raw files → saves Well_locations.csv and well_data_all.csv
- Button 2: Statistics → computes PreMon / PostMon averages and mean fluctuation; saves GWT_mean_fluctuations.csv
- Button 3: Trends → runs Mann–Kendall + Sen's slope; saves GWT_trends.csv
- Button 4: Exit

Each button runs independently (you must run Button 1 first to create processed CSVs used by Buttons 2 & 3).

## Requirements
- Windows
- Python 3.8+
- Packages:
  - pandas
  - numpy
  - tkinter (usually included with Python on Windows)

Install packages (if needed):
```powershell
python -m pip install pandas numpy
```

## How to run
1. Open a terminal (PowerShell or cmd).
2. Run:
```powershell
python "c:\Users\Ranveer\Desktop\WRIS_WELL_Management\python\WRIS_well_GUI.py"
```
3. In the GUI:
   - Select Input Directory (folder containing Excel files).
   - Select Output Directory (folder where CSVs will be saved).
   - Use buttons as described above.

## Input expectations
- Filenames must contain the substring `Ground Water Level` (case-sensitive filter used in the tool).
- Time-series are expected on the second sheet (index 1) with columns `Data Time` and `Data Value`.
- Some header values are read from fixed row indices (tool assumes WRIS Excel layout).

## Seasonal definitions
- Pre-monsoon months: March, April, May (3,4,5)
- Post-monsoon months: October, November, December (10,11,12)

## Trend analysis details
- Mann–Kendall (MK) test with a default alpha used in the script.
- Sen's slope is computed as the median pairwise slope and converted to meters per year (m/yr) in the output CSV.
- If insufficient data (< 3 points for a series), the tool records "no trend" and NaN slope.

## Output columns
- Well_locations.csv: WellID, Latt, Long, Site, Data_Acquisition, Aq_type, Well_depth, Data_From, Data_to
- well_data_all.csv: WellID, DateTime, Ground Water Level (m)
- GWT_mean_fluctuations.csv: WellID, Latitude, Longitude, Aq_Type, PreMon_Mean, PostMon_Mean, Mean_Fluctuation
- GWT_trends.csv: WellID, Latitude, Longitude, premon_trend, premon_slope (m/yr), postmon_trend, postmon_slope (m/yr), GW_trend, GW_slope (m/yr)

## Troubleshooting
- "Processed files not found" — run Button 1 first to generate Well_locations.csv and well_data_all.csv.
- If no files are found, ensure Excel filenames contain `Ground Water Level` and end with `.xlsx` or `.xls`.
- Date parsing errors: verify `Data Time` column values are valid date/time strings.
- If GUI hangs while processing large datasets, wait — work is run on the main thread (future improvement: add background threading).

## Notes & limitations
- Header row indices and sheet layout assume WRIS export format. Adjust code if your files differ.
- MK implementation uses a normal approximation (tie corrections omitted); use specialized libraries for rigorous statistical needs.
- Outlier removal uses IQR method (1.5×IQR).

## License
CC0
- MK implementation uses a normal approximation (tie corrections omitted); use specialized libraries for rigorous statistical needs.
- Outlier removal uses IQR method (1.5×IQR).

## License
CC0 Global
