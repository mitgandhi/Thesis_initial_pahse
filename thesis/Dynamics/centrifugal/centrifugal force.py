import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

class CentrifugalForceAnalyzer:
    def __init__(self):
        self.mK = 0.103  # Mass piston/slipper assembly (kg)
        self.rB = 42.08  # Pitch circle radius (mm)
        self.beta_rad = np.radians(14)  # Swashplate angle (14 degrees in radians)

    def calculate_centrifugal_force(self, phi_deg, speed_rpm, gamma_deg):
        phi = math.radians(phi_deg)
        gamma = math.radians(gamma_deg)
        omega = np.pi * speed_rpm / 30  # Convert RPM to rad/s
        rB_m = self.rB / 1000.0  # Convert mm to m

        r_temp = rB_m - (2 * rB_m * math.tan(self.beta_rad) * math.tan(gamma) * (1 - math.cos(phi)))
        FwK_inclined = self.mK * (omega * omega) * r_temp

        Fwkz = FwK_inclined * math.tan(gamma)  # Z-component
        Fwky = FwK_inclined * math.cos(gamma)  # Y-component

        return {
            'Fwkz': Fwkz,
            'Fwky': Fwky,
            'speed_rpm': speed_rpm,
            'gamma_deg': gamma_deg
        }

def analyze_centrifugal_forces(speed_range_rpm, gamma_angles_deg, phi_angles_deg=None):
    if phi_angles_deg is None:
        phi_angles_deg = [0]  # Simplified to just one phi angle

    analyzer = CentrifugalForceAnalyzer()
    results = []

    for speed_rpm in speed_range_rpm:
        for gamma_deg in gamma_angles_deg:
            for phi_deg in phi_angles_deg:
                forces = analyzer.calculate_centrifugal_force(phi_deg, speed_rpm, gamma_deg)
                results.append(forces)

    return pd.DataFrame(results)





def plot_force_vs_speed(results_df, plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)

    # Plot for Fwkz
    fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor='white')
    ax1.set_facecolor('white')

    for gamma in results_df['gamma_deg'].unique():
        gamma_data = results_df[results_df['gamma_deg'] == gamma]
        ax1.plot(gamma_data['speed_rpm'], gamma_data['Fwkz'], linewidth=3, label=f'γ = {gamma}°')

    ax1.set_xlabel('Speed [RPM]', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Fwkz [N]', fontsize=16, fontweight='bold')
    ax1.set_title('Attaching centrifugal force component: Fwkz vs Speed', fontsize=18, fontweight='bold', pad=20)

    ax1.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Legend below plot in a row format to avoid overlapping
    legend = ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                        frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust the rect to make space for the legend
    plt.savefig(f'{plot_dir}/fwkz_vs_speed.png', dpi=300, bbox_inches='tight', transparent=True, facecolor='none',
                edgecolor='none')
    plt.close()

    # Plot for Fwky
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='white')
    ax2.set_facecolor('white')

    for gamma in results_df['gamma_deg'].unique():
        gamma_data = results_df[results_df['gamma_deg'] == gamma]
        ax2.plot(gamma_data['speed_rpm'], gamma_data['Fwky'], linewidth=3, label=f'γ = {gamma}°')

    ax2.set_xlabel('Speed [RPM]', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Fwky [N]', fontsize=16, fontweight='bold')
    ax2.set_title('Fwky vs Speed', fontsize=18, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Legend below plot in a row format to avoid overlapping
    legend = ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                        frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust the rect to make space for the legend
    plt.savefig(f'{plot_dir}/fwky_vs_speed.png', dpi=300, bbox_inches='tight', transparent=True, facecolor='none',
                edgecolor='none')
    plt.close()

# Example usage
if __name__ == "__main__":
    speed_range_rpm = np.linspace(500, 5000, 10)  # 500 to 5000 RPM, 10 points
    gamma_angles_deg = [0, 5, 10, 15, 20]  # Gamma angles in degrees

    results = analyze_centrifugal_forces(speed_range_rpm, gamma_angles_deg)
    plot_force_vs_speed(results)
