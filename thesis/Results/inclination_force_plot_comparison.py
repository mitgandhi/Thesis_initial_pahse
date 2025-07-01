import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import math
from matplotlib import cm

# --- Correct column names from file format ---
correct_columns = [
    "%time", "revolution", "shaft_angle", "FAKz", "FSKx", "FSKy", "FSKz",
    "FKx", "FKy", "FK", "FaK", "FDK", "FwK", "FAKy"
]

# --- File Paths ---
zero_friction_paths = [p for p in glob.glob(
    'G:/Inline/Highspeed_pump/HP_VD/results/Results_for_inclination_forces_firction_0/forces_piston_*.txt')
                       if os.path.splitext(os.path.basename(p))[0].split('_')[-1] in ['0', '5','10','15']]

with_friction_paths = [p for p in glob.glob(
    'G:/Inline/Highspeed_pump/HP_VD/results/Results_for_inclination_forces/forces_piston_*.txt')
                       if os.path.splitext(os.path.basename(p))[0].split('_')[-1] in ['0', '5', '10','15']]

# Flag to determine if with_friction data should be used
use_with_friction = len(with_friction_paths) > 0  # Will be False if the glob doesn't find any files

# --- Output folders ---
output_folder = 'G:/Inline/Highspeed_pump/HP_VD/Inclination_Plots_Last_Revolution'
csv_output_folder = 'G:/Inline/Highspeed_pump/HP_VD/Inclination_CSV_Last_Revolution'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)

# --- Load and Tag Data ---
data_by_friction = {"zero": {}}
if use_with_friction:
    data_by_friction["with"] = {}

# Load zero friction data
for path in zero_friction_paths:
    angle = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
    try:
        df_full = pd.read_csv(path, sep='\t', skiprows=1, names=correct_columns)

        # Get the maximum revolution value
        max_rev = df_full["revolution"].max()

        # Find the last complete revolution (floor value of max_rev to the last integer)
        last_rev = math.floor(max_rev)

        # Filter for data in the last revolution
        df = df_full[(df_full["revolution"] > last_rev - 1) &
                     (df_full["revolution"] <= last_rev)].reset_index(drop=True)

        # If there's no complete last revolution, take the highest available
        if df.empty and max_rev > 0:
            second_last_rev = last_rev - 1
            df = df_full[(df_full["revolution"] > second_last_rev) &
                         (df_full["revolution"] <= last_rev)].reset_index(drop=True)

        data_by_friction["zero"][angle] = df
        print(f"Loaded zero friction angle {angle}°: Using revolution {last_rev - 1} to {last_rev}")

    except Exception as e:
        print(f"Error reading {path} [zero friction]: {e}")

# Load with friction data if available and flag is set
if use_with_friction:
    for path in with_friction_paths:
        angle = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
        try:
            df_full = pd.read_csv(path, sep='\t', skiprows=1, names=correct_columns)

            # Get the maximum revolution value
            max_rev = df_full["revolution"].max()

            # Find the last complete revolution (floor value of max_rev to the last integer)
            last_rev = math.floor(max_rev)

            # Filter for data in the last revolution
            df = df_full[(df_full["revolution"] > last_rev - 1) &
                         (df_full["revolution"] <= last_rev)].reset_index(drop=True)

            # If there's no complete last revolution, take the highest available
            if df.empty and max_rev > 0:
                second_last_rev = last_rev - 1
                df = df_full[(df_full["revolution"] > second_last_rev) &
                             (df_full["revolution"] <= last_rev)].reset_index(drop=True)

            data_by_friction["with"][angle] = df
            print(f"Loaded with friction angle {angle}°: Using revolution {last_rev - 1} to {last_rev}")

        except Exception as e:
            print(f"Error reading {path} [with friction]: {e}")

# Determine which friction types to include in the plots
friction_types = ["zero"]
if use_with_friction:
    friction_types.append("with")

# --- Get all unique angles and set color map ---
all_angles = set()
for friction_type in friction_types:
    all_angles.update(data_by_friction[friction_type].keys())

angles = sorted(list(all_angles), key=int)
num_angles = len(angles)

# ------------ NEW CODE: CREATE CSV FILES ------------

# Create CSV files for each angle (separate files for with and without friction)
for angle in angles:
    # Export data without friction
    if angle in data_by_friction["zero"]:
        df_no_friction = data_by_friction["zero"][angle].copy()

        # Make sure shaft_angle is in 0-360 range
        df_no_friction['shaft_angle'] = df_no_friction['shaft_angle'] % 360

        # Add custom column with FSK calculated value
        cos14 = math.cos(math.radians(14))
        df_no_friction["FSK_calculated"] = df_no_friction["FAKz"] / cos14

        # Add custom column for Querkraft
        df_no_friction["Querkraft"] = df_no_friction["FAKy"] + df_no_friction["FSKy"]

        # Save to CSV - don't rename columns for individual files to keep it simple
        no_friction_csv_path = os.path.join(csv_output_folder, f'angle_{angle}_no_friction.csv')
        df_no_friction.to_csv(no_friction_csv_path, index=False)
        print(f"Saved no-friction CSV for angle {angle}° to {no_friction_csv_path}")

    # Export data with friction if available
    if use_with_friction and angle in data_by_friction["with"]:
        df_with_friction = data_by_friction["with"][angle].copy()

        # Make sure shaft_angle is in 0-360 range
        df_with_friction['shaft_angle'] = df_with_friction['shaft_angle'] % 360

        # Add custom column with FSK calculated value
        cos14 = math.cos(math.radians(14))
        df_with_friction["FSK_calculated"] = df_with_friction["FAKz"] / cos14

        # Add custom column for Querkraft
        df_with_friction["Querkraft"] = df_with_friction["FAKy"] + df_with_friction["FSKy"]

        # Save to CSV - don't rename columns for individual files to keep it simple
        with_friction_csv_path = os.path.join(csv_output_folder, f'angle_{angle}_with_friction.csv')
        df_with_friction.to_csv(with_friction_csv_path, index=False)
        print(f"Saved with-friction CSV for angle {angle}° to {with_friction_csv_path}")

    # Create a combined CSV with both friction and no-friction data if both are available
    if use_with_friction and angle in data_by_friction["zero"] and angle in data_by_friction["with"]:
        # Get the data frames
        df_no_friction = data_by_friction["zero"][angle].copy()
        df_with_friction = data_by_friction["with"][angle].copy()

        # Make sure shaft_angle is in 0-360 range
        df_no_friction['shaft_angle'] = df_no_friction['shaft_angle'] % 360
        df_with_friction['shaft_angle'] = df_with_friction['shaft_angle'] % 360

        # Create new column names with suffixes, excluding shaft_angle
        no_friction_cols = {col: f"{col}_no_friction" for col in df_no_friction.columns if col != 'shaft_angle'}
        with_friction_cols = {col: f"{col}_with_friction" for col in df_with_friction.columns if col != 'shaft_angle'}

        # Create copies with renamed columns
        df_no_friction_renamed = df_no_friction.rename(columns=no_friction_cols)
        df_with_friction_renamed = df_with_friction.rename(columns=with_friction_cols)

        # Merge on shaft_angle
        combined_df = pd.merge(df_no_friction_renamed, df_with_friction_renamed,
                               on='shaft_angle', how='outer')

        # Sort by shaft angle
        combined_df = combined_df.sort_values('shaft_angle').reset_index(drop=True)

        # Add calculated columns for both friction cases
        cos14 = math.cos(math.radians(14))
        if "FAKz_no_friction" in combined_df.columns:
            combined_df["FSK_calculated_no_friction"] = combined_df["FAKz_no_friction"] / cos14
        if "FAKz_with_friction" in combined_df.columns:
            combined_df["FSK_calculated_with_friction"] = combined_df["FAKz_with_friction"] / cos14

        # Add Querkraft calculations
        if "FAKy_no_friction" in combined_df.columns and "FSKy_no_friction" in combined_df.columns:
            combined_df["Querkraft_no_friction"] = combined_df["FAKy_no_friction"] + combined_df["FSKy_no_friction"]
        if "FAKy_with_friction" in combined_df.columns and "FSKy_with_friction" in combined_df.columns:
            combined_df["Querkraft_with_friction"] = combined_df["FAKy_with_friction"] + combined_df[
                "FSKy_with_friction"]

        # Save combined CSV
        combined_csv_path = os.path.join(csv_output_folder, f'angle_{angle}_combined.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined CSV for angle {angle}° to {combined_csv_path}")

# Continue with the original plotting code
cmap = plt.get_cmap('turbo', num_angles)
# Define linestyle and label format based on available data
linestyles = {"zero": '-'}
if use_with_friction:
    linestyles["with"] = '--'

## Create and save plots

# For the first set of plots (all columns):
for col in correct_columns:
    if col == "%time":
        continue

    # Create a square figure
    plt.figure(figsize=(10, 10))  # Equal width and height for square

    for idx, angle in enumerate(angles):
        color = cmap(idx)
        for tag in friction_types:
            df = data_by_friction[tag].get(angle)
            if df is not None and not df.empty:
                # Convert shaft_angle to 0-360 range
                shaft_degrees = df["shaft_angle"] % 360

                # label = f"{angle}° ({tag} friction)"
                label = f"{angle}°"
                plt.plot(shaft_degrees, df[col], label=label, color=color, linestyle=linestyles[tag])

    plt.title(f'{col} ', fontsize=18)
    plt.xlabel('Shaft Angle (°)', fontsize=18)
    plt.ylabel(col, fontsize=18)

    # Increased legend size
    plt.legend(title='Inclination', fontsize=15, title_fontsize=18,
               loc='upper right')

    plt.grid(True)
    plt.xticks(np.linspace(0, 360, 13), fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout to fit everything
    plt.tight_layout()

    # Save with a square aspect ratio
    plt.savefig(os.path.join(output_folder, f'{col}_comparison.png'),
                dpi=300,
                transparent=True
                )  # This ensures all elements fit
    plt.close()

# For the FSK plot:
cos14 = math.cos(math.radians(14))
plt.figure(figsize=(10, 10))  # Square figure
for idx, angle in enumerate(angles):
    color = cmap(idx)
    for tag in friction_types:
        df = data_by_friction[tag].get(angle)
        if df is not None and not df.empty:
            # Convert shaft_angle to 0-360 range
            shaft_degrees = df["shaft_angle"] % 360

            fsk_values = df["FAKz"] / cos14
            label = f"{angle}° ({tag} friction)"
            plt.plot(shaft_degrees, fsk_values, label=label, color=color, linestyle=linestyles[tag])

plt.title('Comparison of FSK (FAKz / cos(14°)) over Shaft Angle', fontsize=14)
plt.xlabel('Shaft Angle (degrees)', fontsize=12)
plt.ylabel('FSK', fontsize=12)

# Increased legend size
plt.legend(title='Inclination', fontsize=12, title_fontsize=14, markerscale=1.5,
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.xticks(np.linspace(0, 360, 13), fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Fsk_comparison.png'),
            dpi=300,
            bbox_inches='tight',
            transparent=True
            )
plt.close()

# For the FSKYY plot:
cmap = plt.get_cmap('turbo', len(angles))

plt.figure(figsize=(10, 10))  # Square figure
for idx, angle in enumerate(angles):
    color = cmap(idx)
    for tag in friction_types:
        df = data_by_friction[tag].get(angle)
        if df is not None and not df.empty:
            # Convert shaft_angle to 0-360 range
            shaft_degrees = df["shaft_angle"] % 360

            fskyy = df["FAKy"] + df["FSKy"]
            label = f"{angle}° ({tag} friction)"
            plt.plot(shaft_degrees, fskyy, label=label, color=color, linestyle=linestyles[tag])

plt.title('Querkraft', fontsize=18)
plt.xlabel('Shaft Angle (°)', fontsize=18)
plt.ylabel('Querkraft', fontsize=18)

# Increased legend size
plt.legend(title='Inclination', fontsize=12, loc="upper right")

plt.grid(True)
plt.xticks(np.linspace(0, 360, 13), fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'FSKYY_comparison.png'),
            transparent=True)
plt.close()

print("Processing complete. All plots saved to:", output_folder)
print("All CSV files saved to:", csv_output_folder)