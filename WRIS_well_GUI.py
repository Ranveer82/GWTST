import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import warnings
import numpy as np
import math

warnings.filterwarnings('ignore')

def log_message(msg):
    """Appends a message to the GUI's log box."""
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    root.update_idletasks()

def mk_test(x, t=None, alpha=0.15):
    """
    Simple Mann-Kendall test with Sen's slope.
    x : 1D array-like of values (ordered by t if provided)
    t : 1D array-like of times (numeric). If None, uses 0..n-1
    Returns: (trend_str, sen_slope, p_value)
    """
    x = np.array(x, dtype=float)
    if t is None:
        t = np.arange(len(x))
    else:
        t = np.array(t, dtype=float)

    n = len(x)
    # Handle insufficient data or all NaNs
    if n < 3 or np.all(np.isnan(x)):
        return 'no trend', np.nan, 1.0

    # Remove NaNs while preserving corresponding times
    mask = ~np.isnan(x)
    x, t = x[mask], t[mask]
    n = len(x)
    if n < 3:
        return 'no trend', np.nan, 1.0

    # S statistic
    S = 0
    for k in range(n - 1):
        S += np.sum(np.sign(x[k+1:] - x[k]))

    # Variance (approximation, tie correction omitted for simplicity)
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if var_s == 0:
        z = 0.0
    else:
        z = (S - 1) / math.sqrt(var_s) if S > 0 else (S + 1) / math.sqrt(var_s) if S < 0 else 0.0

    # Two-sided p-value from normal approximation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    # Sen's slope (median of pairwise slopes)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            denom = t[j] - t[i]
            if denom != 0:
                slopes.append((x[j] - x[i]) / denom)
    sen_slope = np.median(slopes) if slopes else np.nan

    trend = 'increasing' if p < alpha and z > 0 else 'decreasing' if p < alpha and z < 0 else 'no trend'

    return trend, sen_slope, p

# --- Helper routines split so buttons are independent ---

def _list_files(input_dir):
    """Finds all relevant 'Ground Water Level' Excel files."""
    try:
        return sorted([f for f in os.listdir(input_dir) if 'Ground Water Level' in f and f.endswith(('.xlsx', '.xls'))])
    except Exception as e:
        log_message(f"Error listing files: {e}")
        return []

def read_well_locations(input_dir, output_dir):
    """Reads header info from each Excel file and saves Well_locations.csv."""
    files = _list_files(input_dir)
    if not files:
        log_message("No 'Ground Water Level' Excel files found in the input directory.")
        return pd.DataFrame(), []

    well_loc = []
    for n, f in enumerate(files, 1):
        try:
            file = pd.read_excel(os.path.join(input_dir, f), 0)
            lat = -999 if pd.isna(file.iloc[17, 1]) else float(file.iloc[17, 1])
            long = -888 if pd.isna(file.iloc[18, 1]) else float(file.iloc[18, 1])
            if lat == -999 or long == -888:
                log_message(f"Invalid coordinates in {f}, skipping.")
                continue
            # Read specific header rows
            site = file.iloc[8, 1]
            data_aq_m = file.iloc[14, 1]
            aq_type = file.iloc[22, 1]
            well_depth = file.iloc[23, 1]
            data_from = file.iloc[19, 1]
            data_to = file.iloc[20, 1]
            well_id = f'W{n}'
            well_loc.append([well_id, lat, long, site, data_aq_m, aq_type, well_depth, data_from, data_to])
        except Exception as e:
            log_message(f"Error reading header from {f}: {e}")
            continue

    cols = ['WellID', 'Latt', 'Long', 'Site', 'Data_Acquisition', 'Aq_type', 'Well_depth', 'Data_From', 'Data_to']
    well_loc_df = pd.DataFrame(well_loc, columns=cols)

    try:
        well_loc_df.to_csv(os.path.join(output_dir, 'Well_locations.csv'), index=False)
        log_message("Saved well location data to Well_locations.csv.")
    except Exception as e:
        log_message(f"Error saving Well_locations.csv: {e}")
    return well_loc_df, files

def read_and_clean_timeseries(input_dir, output_dir, files):
    """Reads timeseries sheets, cleans data, and saves well_data_all.csv."""
    if not files:
        return pd.DataFrame()

    well_data = []
    for n, f in enumerate(files, 1):
        try:
            well_id = f'W{n}'
            # Read the second sheet for time series data
            file = pd.read_excel(os.path.join(input_dir, f), sheet_name=1, skiprows=6, skipfooter=5)
            file['Datetime'] = pd.to_datetime(file['Data Time'], errors='coerce', format='mixed')
            file['GWT'] = pd.to_numeric(file['Data Value'], errors='coerce').abs()
            file.dropna(subset=['Datetime', 'GWT'], inplace=True)

            for _, row in file.iterrows():
                well_data.append([well_id, row.Datetime, row.GWT])
        except Exception as e:
            log_message(f"Error reading time series from {f}: {e}")
            continue

    data_df = pd.DataFrame(well_data, columns=['WellID', 'DateTime', 'Ground Water Level (m)'])
    if data_df.empty:
        log_message("No valid time series data could be extracted.")
        return pd.DataFrame()

    log_message("Removing outliers for each well using IQR method...")
    cleaned_data = []
    for well in data_df['WellID'].unique():
        df_well = data_df[data_df['WellID'] == well].copy()
        series = df_well['Ground Water Level (m)']
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_clean = df_well[(series.between(lower, upper))]
        cleaned_data.append(df_clean)
        removed = len(df_well) - len(df_clean)
        if removed > 0:
            log_message(f" - {well}: Removed {removed} outliers.")

    data_df_clean = pd.concat(cleaned_data, ignore_index=True)
    try:
        data_df_clean.to_csv(os.path.join(output_dir, 'well_data_all.csv'), index=False)
        log_message("Saved cleaned time series to well_data_all.csv.")
    except Exception as e:
        log_message(f"Error saving well_data_all.csv: {e}")
    return data_df_clean

# --- Button callbacks ---

def process_data_button():
    """Button 1: Processes raw Excel files into clean CSVs."""
    in_dir, out_dir = input_dir_var.get(), output_dir_var.get()
    if not os.path.isdir(in_dir) or not os.path.isdir(out_dir):
        messagebox.showerror("Error", "Please select valid input and output directories.")
        return
    log_box.delete(1.0, tk.END)
    log_message("Starting Step 1: Processing raw data...")
    _, files = read_well_locations(in_dir, out_dir)
    read_and_clean_timeseries(in_dir, out_dir, files)
    log_message("Step 1 complete.")
    messagebox.showinfo("Done", "Data processing completed. Well_locations.csv and well_data_all.csv saved.")

def statistics_button():
    """Button 2: Calculates statistics from the processed CSV files."""
    out_dir = output_dir_var.get()
    if not os.path.isdir(out_dir):
        messagebox.showerror("Error", "Please select a valid output directory.")
        return
    log_box.delete(1.0, tk.END)
    log_message("Starting Step 2: Calculating statistics...")

    # --- MODIFIED: Load from processed CSV files ---
    loc_path = os.path.join(out_dir, 'Well_locations.csv')
    data_path = os.path.join(out_dir, 'well_data_all.csv')

    try:
        well_loc_df = pd.read_csv(loc_path)
        data_df_clean = pd.read_csv(data_path)
        # IMPORTANT: Convert DateTime column back to datetime objects
        data_df_clean['DateTime'] = pd.to_datetime(data_df_clean['DateTime'])
        log_message("Successfully loaded processed data files.")
    except FileNotFoundError:
        log_message("Error: Processed files not found.")
        messagebox.showerror("Error", "Could not find 'Well_locations.csv' or 'well_data_all.csv'.\nPlease run Step 1 first.")
        return

    data_df_clean['Month'] = data_df_clean['DateTime'].dt.month
    data_df_clean['Year'] = data_df_clean['DateTime'].dt.year

    # Use WellID as index for easy lookup
    well_loc_df.set_index('WellID', inplace=True)

    results = []
    for well_id in data_df_clean['WellID'].unique():
        well_data = data_df_clean[data_df_clean['WellID'] == well_id]
        premon = well_data[well_data['Month'].isin([3, 4, 5])]
        postmon = well_data[well_data['Month'].isin([10, 11, 12])]

        premon_grouped = premon.groupby('Year')['Ground Water Level (m)'].mean()
        postmon_grouped = postmon.groupby('Year')['Ground Water Level (m)'].mean()

        common_years = premon_grouped.index.intersection(postmon_grouped.index)
        if common_years.empty:
            log_message(f" - {well_id}: No common years for pre/post comparison. Skipping.")
            continue

        premon_mean = premon_grouped.loc[common_years].mean()
        postmon_mean = postmon_grouped.loc[common_years].mean()
        fluctuation = postmon_mean - premon_mean # GWT is depth, so positive means water level rose

        try:
            loc_info = well_loc_df.loc[well_id]
            lat, lon, aq_type = loc_info['Latt'], loc_info['Long'], loc_info['Aq_type']
        except KeyError:
            lat, lon, aq_type = np.nan, np.nan, 'Unknown'

        results.append([well_id, lat, lon, aq_type, premon_mean, postmon_mean, fluctuation])
        log_message(f" - {well_id}: Pre-monsoon mean={premon_mean:.2f}, Post-monsoon mean={postmon_mean:.2f}, Fluctuation={fluctuation:.2f}")

    cols = ['WellID', 'Latitude', 'Longitude', 'Aq_Type', 'PreMon_Mean', 'PostMon_Mean', 'Mean_Fluctuation']
    result_df = pd.DataFrame(results, columns=cols)
    try:
        result_df.to_csv(os.path.join(out_dir, 'GWT_mean_fluctuations.csv'), index=False)
        log_message("Saved mean GWT fluctuation analysis to GWT_mean_fluctuations.csv.")
        messagebox.showinfo("Done", "Statistics saved to GWT_mean_fluctuations.csv.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save statistics: {e}")

def trends_button():
    """Button 3: Calculates trends from the processed CSV files."""
    out_dir = output_dir_var.get()
    if not os.path.isdir(out_dir):
        messagebox.showerror("Error", "Please select a valid output directory.")
        return
    log_box.delete(1.0, tk.END)
    log_message("Starting Step 3: Performing Mann-Kendall trend analysis...")

    # Load from processed CSV files
    loc_path = os.path.join(out_dir, 'Well_locations.csv')
    data_path = os.path.join(out_dir, 'well_data_all.csv')

    try:
        well_loc_df = pd.read_csv(loc_path)
        data_df_clean = pd.read_csv(data_path)
        data_df_clean['DateTime'] = pd.to_datetime(data_df_clean['DateTime'])
        log_message("Successfully loaded processed data files.")
    except FileNotFoundError:
        log_message("Error: Processed files not found.")
        messagebox.showerror("Error", "Could not find 'Well_locations.csv' or 'well_data_all.csv'.\nPlease run Step 1 first.")
        return

    # Keep month and year for seasonal filtering
    data_df_clean['Month'] = data_df_clean['DateTime'].dt.month
    data_df_clean['Year'] = data_df_clean['DateTime'].dt.year

    well_loc_df.set_index('WellID', inplace=True)
    trend_rows = []
    
    for well_id in data_df_clean['WellID'].unique():
        wdf = data_df_clean[data_df_clean['WellID'] == well_id].copy()
        
        # Sort by date just in case, and find the start date for this well
        wdf.sort_values('DateTime', inplace=True)
        well_start_date = wdf['DateTime'].min()

        # NEW: Create the time vector 't' in days from the start date
        wdf['t_days'] = (wdf['DateTime'] - well_start_date).dt.days
        
        # Overall trend
        # MODIFIED: Use all data points, not yearly means
        x_overall = wdf['Ground Water Level (m)'].values
        t_overall = wdf['t_days'].values
        gw_trend, gw_slope_daily, _ = mk_test(x_overall, t=t_overall)
        gw_slope_yearly = gw_slope_daily * 365.25 # Convert slope to meters/year

        # Pre-monsoon trend
        pre = wdf[wdf['Month'].isin([3, 4, 5])]
        if len(pre) > 3: # Check for sufficient data
            x_pre = pre['Ground Water Level (m)'].values
            t_pre = pre['t_days'].values
            pre_trend, pre_slope_daily, _ = mk_test(x_pre, t=t_pre)
            pre_slope_yearly = pre_slope_daily * 365.25
        else:
            pre_trend, pre_slope_yearly = 'no trend', np.nan

        # Post-monsoon trend
        post = wdf[wdf['Month'].isin([10, 11, 12])]
        if len(post) > 3: # Check for sufficient data
            x_post = post['Ground Water Level (m)'].values
            t_post = post['t_days'].values
            post_trend, post_slope_daily, _ = mk_test(x_post, t=t_post)
            post_slope_yearly = post_slope_daily * 365.25
        else:
            post_trend, post_slope_yearly = 'no trend', np.nan
            
        try:
            loc_info = well_loc_df.loc[well_id]
            lat, lon = loc_info['Latt'], loc_info['Long']
        except KeyError:
            lat, lon = np.nan, np.nan
            
        # MODIFIED: Save the yearly slope for better interpretation
        trend_rows.append([
            well_id, lat, lon, 
            pre_trend, pre_slope_yearly, 
            post_trend, post_slope_yearly, 
            gw_trend, gw_slope_yearly
        ])
        log_message(f" - {well_id}: Pre-monsoon={pre_trend}, Post-monsoon={post_trend}, Overall={gw_trend}")

    # MODIFIED: Update column names to reflect yearly slope
    cols = [
        'WellID', 'Latitude', 'Longitude', 
        'premon_trend', 'premon_slope (m/yr)', 
        'postmon_trend', 'postmon_slope (m/yr)', 
        'GW_trend', 'GW_slope (m/yr)'
    ]
    trend_df = pd.DataFrame(trend_rows, columns=cols)
    try:
        trend_df.to_csv(os.path.join(out_dir, 'GWT_trends.csv'), index=False)
        log_message("Saved Mann-Kendall trend analysis to GWT_trends.csv.")
        messagebox.showinfo("Done", "Trend analysis saved to GWT_trends.csv.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save trends: {e}")

def browse_input():
    folder = filedialog.askdirectory()
    if folder:
        input_dir_var.set(folder)

def browse_output():
    folder = filedialog.askdirectory()
    if folder:
        output_dir_var.set(folder)

def exit_app():
    root.destroy()

# --- GUI Setup ---
root = tk.Tk()
root.title("Groundwater Time Series Processor for WRIS")
root.geometry("700x550")

input_dir_var = tk.StringVar()
output_dir_var = tk.StringVar()

# --- Main Frame ---
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Input/Output Frames
io_frame = tk.Frame(main_frame)
io_frame.pack(fill=tk.X, pady=5)
tk.Label(io_frame, text="Input Directory (raw Excel files):").pack(anchor='w')
in_entry_frame = tk.Frame(io_frame)
in_entry_frame.pack(fill=tk.X)
tk.Entry(in_entry_frame, textvariable=input_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
tk.Button(in_entry_frame, text="Browse...", command=browse_input).pack(side=tk.RIGHT, padx=(5,0))

tk.Label(io_frame, text="Output Directory (for CSV results):").pack(anchor='w', pady=(10,0))
out_entry_frame = tk.Frame(io_frame)
out_entry_frame.pack(fill=tk.X)
tk.Entry(out_entry_frame, textvariable=output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
tk.Button(out_entry_frame, text="Browse...", command=browse_output).pack(side=tk.RIGHT, padx=(5,0))

# Buttons Frame
btn_frame = tk.Frame(main_frame)
btn_frame.pack(fill=tk.X, pady=15)
btn_style = {'bg': "#4CAF50", 'fg': "white", 'height': 2, 'font': ('Helvetica', 9, 'bold')}
tk.Button(btn_frame, text="1: Process Raw WRIS Data", command=process_data_button, **btn_style).pack(fill=tk.X, pady=3)
btn_style['bg'] = "#2196F3"
tk.Button(btn_frame, text="2: Calculate Statistics", command=statistics_button, **btn_style).pack(fill=tk.X, pady=3)
btn_style['bg'] = "#9C27B0"
tk.Button(btn_frame, text="3: Analyze Trends (MK test)", command=trends_button, **btn_style).pack(fill=tk.X, pady=3)
btn_style['bg'] = "#f44336"
tk.Button(btn_frame, text="Exit", command=exit_app, **btn_style).pack(fill=tk.X, pady=(10,0))

# Log Display Box
tk.Label(main_frame, text="Processing Log:").pack(anchor='w')
log_box = scrolledtext.ScrolledText(main_frame, height=10, wrap=tk.WORD, state='normal')
log_box.pack(fill=tk.BOTH, expand=True)

root.mainloop()