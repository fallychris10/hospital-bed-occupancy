import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from simulation import run_hospital_simulation

st.set_page_config(layout="wide", page_title="Hospital Simulation Dashboard")

# --- Dashboard Title ---
st.title("🏥 Hospital Bed & Staffing Simulation Dashboard")
st.markdown("An Interactive Tool for Analyzing Patient Flow Using Queuing Theory")

# --- Sidebar (Control Panel) ---
st.sidebar.header("Control Panel")

with st.sidebar.form(key='sim_params_form'):
    st.subheader("1. Scenario Selector")
    scenario = st.selectbox(
        "Select Simulation Model",
        [
            "Baseline Model (Homogeneous Staff)", 
            "Experience-Based Model (Heterogeneous Staff)", 
            "Workload-Dependent Model (Dynamic Service Rates)"
        ]
    )

    st.subheader("2. System Configuration")
    arrival_rate = st.slider("Avg. Patient Arrivals per Day (λ)", 1, 50, 10)
    num_beds = st.slider("Total Bed Capacity (K)", 1, 100, 20)
    num_staff = st.slider("Total Staff on Duty (c)", 1, 50, 10)
    simulation_duration = st.number_input("Simulation Period (Days)", min_value=7, max_value=365, value=30)

    st.subheader("3. Model-Specific Parameters")
    # Conditional inputs based on scenario
    if 'Experience-Based' in scenario:
        senior_staff_mix = st.slider("Senior Staff Mix (%)", 0, 100, 50)
        senior_service_days = st.number_input("Avg. Days per Patient (Senior Staff)", 0.1, 10.0, 3.0, 0.1)
        junior_service_days = st.number_input("Avg. Days per Patient (Junior Staff)", 0.1, 10.0, 5.0, 0.1)
    else:
        # Use placeholders for other scenarios
        senior_staff_mix = 100
        senior_service_days = 3.0
        junior_service_days = 3.0

    if 'Workload-Dependent' in scenario:
        workload_factor_alpha = st.slider(
            "Staff Capacity Factor (α)", 0.1, 5.0, 1.5, 0.1,
            help="Max patients a staff member can manage before performance degrades."
        )
    else:
        workload_factor_alpha = 1.5

    st.subheader("4. Simulation Control")
    run_button = st.form_submit_button(label='► Run Simulation')

# Store params in a dictionary
params = {
    'scenario': scenario,
    'arrival_rate': arrival_rate,
    'num_beds': num_beds,
    'num_staff': num_staff,
    'simulation_duration': simulation_duration,
    'senior_staff_mix': senior_staff_mix,
    'senior_service_days': senior_service_days,
    'junior_service_days': junior_service_days,
    'workload_factor_alpha': workload_factor_alpha
}

# --- Main Panel (Results Display) ---
if run_button:
    with st.spinner('Simulation in progress... Please wait.'):
        st.session_state.results = run_hospital_simulation(params)

# Initialize state for comparison
if 'comparison_runs' not in st.session_state:
    st.session_state.comparison_runs = {}

if 'results' not in st.session_state:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to see the results.")
else:
    results = st.session_state.results
    kpis = results['kpis']
    patient_log_df = results['patient_log_df']
    occupancy_df = results['occupancy_df']

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Overview", "System Dynamics", "Sensitivity Analysis", "Raw Data Log"])

    with tab1:
        st.header(f"Key Performance Indicators (KPIs) for: {params['scenario']}")
        cols = st.columns(5)
        cols[0].metric("Blocking Probability", f"{kpis.get('Blocking Probability (%)', 0):.2f}%")
        cols[1].metric("Average Bed Occupancy", f"{kpis.get('Average Bed Occupancy (%)', 0):.2f}%")
        cols[2].metric("Avg. Wait for Staff", f"{kpis.get('Average Wait for Staff (days)', 0):.2f} Days")
        cols[3].metric("Avg. Length of Stay", f"{kpis.get('Average Patient Length of Stay (days)', 0):.2f} Days")
        cols[4].metric("Total Patients Served", f"{kpis.get('Total Patients Served', 0)}")

        # --- Scenario Comparison Section ---
        st.header("Scenario Comparison")
        col1, col2 = st.columns([1, 3])
        with col1:
            # Use a unique key for the button based on the scenario name to avoid conflicts
            if st.button(f"Add '{params['scenario']}' to Comparison"):
                st.session_state.comparison_runs[params['scenario']] = kpis
                st.success(f"Added '{params['scenario']}' to comparison.")
            if st.button("Clear Comparison Data"):
                st.session_state.comparison_runs = {}
                st.info("Comparison data cleared.")

        with col2:
            if len(st.session_state.comparison_runs) > 0:
                comparison_df = pd.DataFrame.from_dict(st.session_state.comparison_runs, orient='index')
                st.write("**Comparison Results:**")
                st.dataframe(comparison_df.style.format("{:.2f}"))

                # Create grouped bar chart
                df_plot = comparison_df.reset_index().rename(columns={'index': 'Scenario'})
                df_melted = df_plot.melt(id_vars='Scenario', var_name='KPI', value_name='Value')
                
                fig_comp = px.bar(
                    df_melted, 
                    x="Scenario", 
                    y="Value", 
                    color="KPI", 
                    barmode="group",
                    title="KPI Comparison Across Scenarios"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Run a simulation and click 'Add to Comparison' to build a comparison chart.")

    with tab2:
        st.header("System Dynamics Over Time")
        # Bed Occupancy Plot
        fig_occupancy = px.line(occupancy_df, x='Time', y='PatientsInSystem', title='Bed Occupancy Over Time')
        fig_occupancy.add_hline(y=params['num_beds'], line_dash="dash", line_color="red", annotation_text="Total Bed Capacity")
        st.plotly_chart(fig_occupancy, use_container_width=True)

        # Patient Outcomes Plot
        outcomes = patient_log_df['Status'].value_counts().reset_index()
        outcomes.columns = ['Status', 'Count']
        fig_outcomes = px.pie(outcomes, names='Status', values='Count', title='Patient Journey Outcomes')
        st.plotly_chart(fig_outcomes, use_container_width=True)

        st.header("Additional Report Figures")
        st.markdown("### Figure 4.1: System Performance Metrics Across Scenarios")
        
        # Create data for the bar chart
        scenarios = ['Baseline', 'Experience-Based', 'Workload-Dependent']
        utilization = [72.4, 89.2, 95.7]
        blocking_prob = [0.0, 16.67, 9.09]
        
        # Create the bar chart using Plotly
        fig_comparison = go.Figure(data=[
            go.Bar(name='System Utilization (%)', x=scenarios, y=utilization, marker_color='rgb(158,202,225)'),
            go.Bar(name='Blocking Probability (%)', x=scenarios, y=blocking_prob, marker_color='rgb(255,161,161)')
        ])
        
        # Update layout to match the paper's style
        fig_comparison.update_layout(
            barmode='group',
            title='System Performance Metrics Across Scenarios',
            yaxis_title='Performance Metrics',
            yaxis=dict(range=[0, 100]),  # Match the y-axis range from the paper
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            height=500
        )
        
        # Add value labels on top of bars
        fig_comparison.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        
        # Display the figure
        st.plotly_chart(fig_comparison, use_container_width=True)

    with tab3:
        st.header("Sensitivity Analysis")
        st.markdown("Analyze how a key performance metric changes as you vary a single input parameter.")
        
        col1, col2 = st.columns(2)
        x_axis_param = col1.selectbox("1. Select Input Parameter to Vary", list(params.keys()))
        y_axis_kpi = col2.selectbox("2. Select Performance Metric to Track", list(kpis.keys()))

        if st.button("Run Sensitivity Analysis"):
            with st.spinner(f"Running multiple simulations to vary '{x_axis_param}'..."):
                param_range_val = params[x_axis_param]
                if isinstance(param_range_val, (int, float)):
                    param_range = np.linspace(param_range_val * 0.5, param_range_val * 1.5, 10)
                    sensitivity_results = []
                    for val in param_range:
                        temp_params = params.copy()
                        temp_params[x_axis_param] = val
                        run_res = run_hospital_simulation(temp_params)
                        sensitivity_results.append({'param_val': val, 'kpi_val': run_res['kpis'][y_axis_kpi]})
                    
                    sensitivity_df = pd.DataFrame(sensitivity_results)
                    fig_sensitivity = px.line(sensitivity_df, x='param_val', y='kpi_val', title=f'{y_axis_kpi} vs. {x_axis_param}')
                    fig_sensitivity.update_layout(xaxis_title=x_axis_param, yaxis_title=y_axis_kpi)
                    st.plotly_chart(fig_sensitivity, use_container_width=True)
                else:
                    st.warning(f"Cannot run sensitivity analysis on a non-numeric parameter: '{x_axis_param}'.")
                    sensitivity_results.append({'param_val': val, 'kpi_val': run_res['kpis'][y_axis_kpi]})
                
                sensitivity_df = pd.DataFrame(sensitivity_results)
                fig_sensitivity = px.line(sensitivity_df, x='param_val', y='kpi_val', title=f'{y_axis_kpi} vs. {x_axis_param}')
                fig_sensitivity.update_layout(xaxis_title=x_axis_param, yaxis_title=y_axis_kpi)
                st.plotly_chart(fig_sensitivity, use_container_width=True)

    with tab4:
        st.header("Raw Simulation Patient Log")
        st.dataframe(patient_log_df)
        st.download_button(
            label="Download Log as CSV",
            data=patient_log_df.to_csv(index=False).encode('utf-8'),
            file_name='patient_log.csv',
            mime='text/csv',
        )
