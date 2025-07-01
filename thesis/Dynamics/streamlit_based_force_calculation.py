import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os

# === User Input Section ===
pdc_file_path = "P_DC_subset.csv"  # 🔁 Change this path to your actual file location

# Static input parameters
R = 42.08* 0.001         # meters
r = 9.73 * 0.001         # meters
r_inner =1.5 * 0.001    # meters
Alpha = 14             # degrees
gamma_values = [0, 5, 10, 15]
N = 4400                 # RPM
mass = 0.103117             # kg

# === Load PDC Data ===
try:
    pdc_df = pd.read_csv(pdc_file_path)
    if 'PDC' not in pdc_df.columns:
        raise ValueError("CSV file must have a column named 'PDC'.")
    pdc_data = pdc_df['PDC'].to_numpy()
except Exception as e:
    print(f"Error loading PDC data: {e}")
    exit(1)

phi = np.linspace(0, 360, len(pdc_data))


# === Main Calculation Function ===
def calculate_piston_dynamics(R, r, r_inner, Alpha, gamma_values, N, mass, pdc_data):
    data = {
        'R': R, 'r': r, 'r_inner': r_inner, 'Alpha': Alpha, 'gamma': gamma_values,
        'N': N, 'mass': mass, 'PDC': pdc_data, 'phi': phi
    }

    pCase = 1.0
    area_k = np.pi * (r ** 2 - r_inner ** 2)
    data['Pressure_Force'] = (pdc_data - pCase) * area_k

    beta_rad = np.radians(Alpha)
    omega = (N * np.pi / 30)
    phi_rad = np.radians(phi)

    data['Acceleration'] = {}
    data['Inertial_Force'] = {}
    data['FAKz'] = {}
    data['FAKy'] = {}
    data['FSK'] = {}
    data['FSKy'] = {}
    data['FSKx'] = {}
    data['QuerKraft'] = {}

    for g in gamma_values:
        zeta_rad = np.radians(g)
        K = np.tan(beta_rad) * np.tan(zeta_rad)
        term1 = (omega ** 2) * R * (np.tan(beta_rad) / np.cos(zeta_rad)) * (1 + K)
        num = K * (1 + np.sin(phi_rad) ** 2) + np.cos(phi_rad)
        denom = (1 + K * np.cos(phi_rad)) ** 3
        acc = term1 * (num / denom)

        data['Acceleration'][g] = acc
        data['Inertial_Force'][g] = mass * acc

        F1wkz = mass * (omega ** 2) * R * np.sin(zeta_rad)
        F1AK = data['Pressure_Force'] + data['Inertial_Force'][g] + F1wkz

        data['FAKz'][g] = F1AK * np.cos(zeta_rad)
        data['FAKy'][g] = F1AK * np.sin(zeta_rad)

        data['FSK'][g] = data['FAKz'][g] / np.cos(beta_rad)
        data['FSKy'][g] = data['FSK'][g] * np.sin(beta_rad)
        data['FSKx'][g] = -data['FSK'][g] * np.sin(zeta_rad)

        data['QuerKraft'][g] = data['FAKy'][g] + data['FSKy'][g]

    return data


# === Export to ZIP ===
def export_results_to_zip(results, output_path="piston_dynamics_results.zip"):
    with zipfile.ZipFile(output_path, 'w') as zf:
        params_df = pd.DataFrame({
            'Parameter': ['R (m)', 'r (m)', 'r_inner (m)', 'Alpha (degrees)', 'N (RPM)', 'mass (kg)'],
            'Value': [results['R'], results['r'], results['r_inner'],
                      results['Alpha'], results['N'], results['mass']]
        })
        zf.writestr('parameters.csv', params_df.to_csv(index=False))

        for g in results['gamma']:
            gamma_data = {'phi': results['phi']}

            for key in ['Acceleration', 'Inertial_Force', 'FAKz', 'FAKy',
                        'FSK', 'FSKy', 'FSKx', 'QuerKraft']:
                gamma_data[key] = results[key][g]

            gamma_data['Pressure_Force'] = results['Pressure_Force']
            df = pd.DataFrame(gamma_data)
            zf.writestr(f'gamma_{g}_data.csv', df.to_csv(index=False))

        consolidated = {'phi': results['phi'], 'Pressure_Force': results['Pressure_Force']}
        for key in ['Acceleration', 'Inertial_Force', 'FAKz', 'FAKy', 'FSK', 'FSKy', 'FSKx', 'QuerKraft']:
            for g in results['gamma']:
                consolidated[f'{key}_gamma_{g}'] = results[key][g]

        df_all = pd.DataFrame(consolidated)
        zf.writestr('all_gamma_data.csv', df_all.to_csv(index=False))

    print(f"✅ Data exported to {output_path}")


# === Plotting ===
def plot_series(results, key, ylabel, title):
    plt.figure(figsize=(8, 5))
    for g in gamma_values:
        plt.plot(results['phi'], results[key][g], label=f'γ={g}')
    plt.xlabel("Shaft Angle (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# === Execute Everything ===
results = calculate_piston_dynamics(R, r, r_inner, Alpha, gamma_values, N, mass, pdc_data)
export_results_to_zip(results)

# === Optional: Plot Results ===
plot_series(results, 'FAKz', 'FAKz (N)', 'Axial Force vs Shaft Angle')
plot_series(results, 'QuerKraft', 'QuerKraft (N)', 'Total Lateral Force vs Shaft Angle')
