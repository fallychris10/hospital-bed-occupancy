import pandas as pd
import matplotlib.pyplot as plt
import os

# --- File Paths ---
FILE_PATH = 'simulation_results.csv'
OUTPUT_DIR = 'report'

def calculate_kpis(df_scenario):
    """Calculates key performance indicators for a given scenario's data."""
    total_arrivals = len(df_scenario)
    blocked_patients = df_scenario['blocked'].sum()
    treated_patients = total_arrivals - blocked_patients
    
    if total_arrivals == 0:
        return {}

    blocking_prob = (blocked_patients / total_arrivals) * 100 if total_arrivals > 0 else 0
    
    df_treated = df_scenario[df_scenario['blocked'] == 0]
    avg_wait_time = df_treated['wait_time'].mean() if not df_treated.empty else 0
    avg_treatment_time = df_treated['treatment_time'].mean() if not df_treated.empty else 0
    # Using the same formula as the dashboard for consistency (10 beds, 7 days)
    avg_utilization = (df_treated['treatment_time'].sum() / (10 * 7 * 24 * 60)) * 100 

    return {
        'Total Arrivals': total_arrivals,
        'Patients Treated': treated_patients,
        'Patients Blocked (Turned Away)': blocked_patients,
        'Blocking Probability (%)': blocking_prob,
        'Average Wait Time (minutes)': avg_wait_time,
        'Average Treatment Time (minutes)': avg_treatment_time,
        'Bed Utilization (%)': avg_utilization
    }

def main():
    """Main function to generate the static report files."""
    print("Generating static report...")

    # Load data
    if not os.path.exists(FILE_PATH):
        print(f"Error: The data file '{FILE_PATH}' was not found. Please run `simulation.py` first.")
        return
    df = pd.read_csv(FILE_PATH)

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Calculate KPIs ---
    scenarios = df['scenario'].unique()
    kpi_data = []
    for scenario in scenarios:
        kpi_row = calculate_kpis(df[df['scenario'] == scenario])
        kpi_row['Scenario'] = scenario
        kpi_data.append(kpi_row)
    
    df_kpis = pd.DataFrame(kpi_data).set_index('Scenario')
    
    # --- Generate HTML Table ---
    html_path = os.path.join(OUTPUT_DIR, 'report_table.html')
    df_kpis.to_html(html_path, float_format='{:.2f}'.format, border=1)
    print(f"KPI table saved to {html_path}")

    # --- Generate Plots ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # Blocking Probability Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df_kpis['Blocking Probability (%)'].plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Blocking Probability by Scenario', fontsize=16)
    ax1.set_ylabel('Probability (%)', fontsize=12)
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, 'blocking_probability.png')
    plt.savefig(plot1_path)
    print(f"Blocking probability plot saved to {plot1_path}")
    plt.close(fig1)

    # Average Wait Time Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_kpis['Average Wait Time (minutes)'].plot(kind='bar', ax=ax2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Average Wait Time by Scenario', fontsize=16)
    ax2.set_ylabel('Time (minutes)', fontsize=12)
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, 'wait_time.png')
    plt.savefig(plot2_path)
    print(f"Wait time plot saved to {plot2_path}")
    plt.close(fig2)

    print("\nStatic report generation complete.")
    print(f"You can find the results in the '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    main()
