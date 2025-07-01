import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_data(piston_folder_path, csv_file_path, zeta_values=[0, 5]):
    """Load piston and CSV data"""
    piston_data_dict = {}

    # Load piston files
    for zeta in zeta_values:
        piston_file = f"piston_{zeta}.txt"
        piston_file_path = os.path.join(piston_folder_path, piston_file)

        try:
            piston_data = pd.read_csv(piston_file_path, sep='\t')
            # Filter to one revolution and convert to degrees
            piston_data = piston_data[piston_data['revolution'] <= 1.0].copy()
            piston_data['phi_deg_piston'] = piston_data['revolution'] * 360.0
            piston_data_dict[zeta] = piston_data
        except:
            piston_data_dict[zeta] = pd.DataFrame()

    # Load CSV data
    csv_data = pd.read_csv(csv_file_path)
    csv_filtered_dict = {}
    for zeta in zeta_values:
        csv_filtered_dict[zeta] = csv_data[csv_data['zeta_deg'] == zeta].copy()

    return piston_data_dict, csv_filtered_dict


def create_querkraft_plots(piston_data_dict, csv_data_dict, output_dir="plots"):
    """Create 3 separate Querkraft plots: Python only, Caspar only, and comparison"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zeta_colors = {
        0: 'red',
        5: 'green'
    }

    # Plot 1: Python only
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0)  # Transparent background

    for zeta in sorted(csv_data_dict.keys()):
        color = zeta_colors[zeta]
        csv_data = csv_data_dict[zeta]

        if len(csv_data) > 0:
            if 'FSKy' in csv_data.columns and 'FAKy' in csv_data.columns:
                querkraft_data = csv_data['FSKy'] + csv_data['FAKy']
                ax.plot(csv_data['phi_deg'], querkraft_data,
                        color=color, linestyle='--',
                        linewidth=2.5, label=f'Python (γ={zeta}°)', alpha=1.0)

    ax.set_xlabel('Phi [°]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Querkraft [N]', fontsize=14, fontweight='bold')
    ax.set_title('Querkraft - Python Results', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    legend = ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax.set_xlim(0, 360)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    ax.tick_params(axis='x', bottom=True, top=False)

    plot_path = os.path.join(output_dir, "Querkraft_python_only.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 2: Caspar only
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0)  # Transparent background

    for zeta in sorted(piston_data_dict.keys()):
        color = zeta_colors[zeta]
        piston_data = piston_data_dict[zeta]

        if len(piston_data) > 0:
            if 'FSKy' in piston_data.columns and 'FAKy' in piston_data.columns:
                querkraft_data = piston_data['FSKy'] + piston_data['FAKy']
                ax.plot(piston_data['phi_deg_piston'], querkraft_data,
                        color=color, linestyle='-',
                        linewidth=2.5, label=f'Caspar (γ={zeta}°)', alpha=1.0)

    ax.set_xlabel('Phi [°]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Querkraft [N]', fontsize=14, fontweight='bold')
    ax.set_title('Querkraft - Caspar Results', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    legend = ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax.set_xlim(0, 360)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    ax.tick_params(axis='x', bottom=True, top=False)

    plot_path = os.path.join(output_dir, "Querkraft_caspar_only.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 3: Comparison (Python vs Caspar)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0)  # Transparent background

    for zeta in sorted(piston_data_dict.keys()):
        color = zeta_colors[zeta]

        # Plot Caspar data
        piston_data = piston_data_dict[zeta]
        if len(piston_data) > 0:
            if 'FSKy' in piston_data.columns and 'FAKy' in piston_data.columns:
                querkraft_data = piston_data['FSKy'] + piston_data['FAKy']
                ax.plot(piston_data['phi_deg_piston'], querkraft_data,
                        color=color, linestyle='-',
                        linewidth=2.5, label=f'Caspar (γ={zeta}°)', alpha=1.0)

        # Plot Python data
        csv_data = csv_data_dict[zeta]
        if len(csv_data) > 0:
            if 'FSKy' in csv_data.columns and 'FAKy' in csv_data.columns:
                querkraft_data = csv_data['FSKy'] + csv_data['FAKy']
                ax.plot(csv_data['phi_deg'], querkraft_data,
                        color=color, linestyle='--',
                        linewidth=2.5, label=f'Python (γ={zeta}°)', alpha=1.0)

    ax.set_xlabel('Phi [°]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Querkraft [N]', fontsize=14, fontweight='bold')
    ax.set_title('Querkraft - Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    legend = ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax.set_xlim(0, 360)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    ax.tick_params(axis='x', bottom=True, top=False)

    plot_path = os.path.join(output_dir, "Querkraft_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {plot_path}")
    plt.close()


def create_plots(piston_data_dict, csv_data_dict, force_columns, output_dir="plots"):
    """Create and save individual plots (excluding Querkraft)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zeta_colors = {
        0: 'red',
        5: 'green'
    }

    # Filter out Querkraft since it's handled separately
    force_columns_filtered = [col for col in force_columns if col != 'Querkraft']

    for force_col in force_columns_filtered:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_alpha(0)  # Transparent background

        for zeta in sorted(piston_data_dict.keys()):
            color = zeta_colors[zeta]

            # Plot piston data
            piston_data = piston_data_dict[zeta]
            if len(piston_data) > 0:
                # Create FSK from FAK_inclined if this is FAK_inclined column
                if force_col == 'FAK_inclined':
                    if 'FAK_inclined' in piston_data.columns:
                        # Calculate FSK = FAK_inclined * cos(gamma) / cos(14°)
                        gamma_rad = np.radians(zeta)  # gamma angle in radians
                        cos_14_deg = np.cos(np.radians(14))
                        fsk_data = piston_data['FAK_inclined'] * np.cos(gamma_rad) / cos_14_deg
                        ax.plot(piston_data['phi_deg_piston'], fsk_data,
                                color=color, linestyle='-',
                                linewidth=2.5, label=f'Caspar (γ={zeta}°)', alpha=1.0)
                # Plot regular columns
                elif force_col in piston_data.columns:
                    ax.plot(piston_data['phi_deg_piston'], piston_data[force_col],
                            color=color, linestyle='-',
                            linewidth=2.5, label=f'Caspar (γ={zeta}°)', alpha=1.0)

            # Plot CSV data
            csv_data = csv_data_dict[zeta]
            if len(csv_data) > 0:
                # Create FSK from FAK_inclined if this is FAK_inclined column
                if force_col == 'FAK_inclined':
                    if 'FAK_inclined' in csv_data.columns:
                        # Calculate FSK = FAK_inclined * cos(gamma) / cos(14°)
                        gamma_rad = np.radians(zeta)  # gamma angle in radians
                        cos_14_deg = np.cos(np.radians(14))
                        fsk_data = csv_data['FAK_inclined'] * np.cos(gamma_rad) / cos_14_deg
                        ax.plot(csv_data['phi_deg'], fsk_data,
                                color=color, linestyle='--',
                                linewidth=2.5, label=f'Python (γ={zeta}°)', alpha=1.0)
                # Plot regular columns
                elif force_col in csv_data.columns:
                    ax.plot(csv_data['phi_deg'], csv_data[force_col],
                            color=color, linestyle='--',
                            linewidth=2.5, label=f'Python (γ={zeta}°)', alpha=1.0)

        ax.set_xlabel('Phi [°]', fontsize=14, fontweight='bold')
        ax.xaxis.set_label_position('bottom')  # Ensure x-axis label is at bottom

        # Set appropriate ylabel and title
        if force_col == 'FAK_inclined':
            ax.set_ylabel('FSK [N]', fontsize=14, fontweight='bold')
            ax.set_title('Total Axial Force', fontsize=16, fontweight='bold')
            plot_filename = "FSK_comparison.png"
        elif force_col == 'FSKy':
            ax.set_ylabel('FSKy [N]', fontsize=14, fontweight='bold')
            ax.set_title('Radial Reaction Force', fontsize=16, fontweight='bold')
            plot_filename = "FSKy_comparison.png"
        elif force_col == 'FAKy':
            ax.set_ylabel('FAKy [N]', fontsize=14, fontweight='bold')
            ax.set_title('Radial Piston Force', fontsize=16, fontweight='bold')
            plot_filename = "FAKy_comparison.png"
        else:
            ax.set_ylabel(f'{force_col} [N]', fontsize=14, fontweight='bold')
            ax.set_title(f'{force_col} Comparison', fontsize=16, fontweight='bold')
            plot_filename = f"{force_col}_comparison.png"

        ax.grid(True, alpha=0.3)

        # Legend in column format (2 columns instead of 4)
        legend = ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                           frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
        legend.get_frame().set_facecolor('white')

        ax.set_xlim(0, 360)

        # Move x-axis ticks and labels to bottom
        ax.tick_params(axis='x', labelbottom=True, labeltop=False)
        ax.tick_params(axis='x', bottom=True, top=False)

        # Save plot
        plot_path = os.path.join(output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Saved: {plot_path}")

        plt.close()


def main():
    # File paths
    piston_folder_path = "C:/Users/MIT/Desktop/thesis_temp/Results_inclined_piston_forces_caspar/Results/Friction"
    csv_file_path = "piston_forces_multiple_zeta.csv"

    # Force columns (Querkraft will be handled separately)
    force_columns = ['FAK_inclined', 'FSKy', 'FAKy', 'Querkraft']

    # Load data
    piston_data_dict, csv_data_dict = load_data(piston_folder_path, csv_file_path)

    # Create regular plots (excluding Querkraft)
    create_plots(piston_data_dict, csv_data_dict, force_columns)

    # Create the 3 separate Querkraft plots
    create_querkraft_plots(piston_data_dict, csv_data_dict)

    print("Done! Created plots:")
    print("- FSK_comparison.png")
    print("- FSKy_comparison.png")
    print("- FAKy_comparison.png")
    print("- Querkraft_python_only.png")
    print("- Querkraft_caspar_only.png")
    print("- Querkraft_comparison.png")


if __name__ == "__main__":
    main()