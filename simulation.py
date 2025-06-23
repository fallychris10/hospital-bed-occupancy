import simpy
import numpy as np
import pandas as pd

RANDOM_SEED = 42
MINS_IN_DAY = 24 * 60

def run_hospital_simulation(params):
    """ 
    Runs a discrete-event simulation of a hospital ward based on input parameters.
    This single function is the core simulation engine for the dashboard.
    """

    class Hospital:
        def __init__(self, env, params):
            self.env = env
            self.params = params
            # The number of staff (c) are the servers
            self.staff = simpy.Resource(env, capacity=params['num_staff'])
            self.bed_capacity = params['num_beds']
            self.beds_in_use = 0
            
            # Data logs
            self.patient_log = []
            self.occupancy_log = []

        def patient(self, name):
            arrival_time = self.env.now

            # Balking: Check if total beds (K) are full
            if self.beds_in_use >= self.bed_capacity:
                self.patient_log.append({
                    'PatientID': name, 'ArrivalTime': arrival_time, 'WaitTime': np.nan,
                    'TreatmentTime': np.nan, 'TotalTime': np.nan, 'Status': 'Balked (No Beds)',
                    'TreatedBy': 'N/A'
                })
                return
            
            self.beds_in_use += 1
            self.occupancy_log.append((self.env.now, self.beds_in_use))
            
            # Request a staff member
            with self.staff.request() as req:
                yield req
                wait_time = self.env.now - arrival_time

                # Determine treatment time based on scenario
                treatment_time = self.get_treatment_time()
                
                yield self.env.timeout(treatment_time)

                # Release resources and log data
                self.beds_in_use -= 1
                self.occupancy_log.append((self.env.now, self.beds_in_use))
                
                self.patient_log.append({
                    'PatientID': name, 'ArrivalTime': arrival_time, 'WaitTime': wait_time,
                    'TreatmentTime': treatment_time, 'TotalTime': self.env.now - arrival_time,
                    'Status': 'Discharged', 'TreatedBy': 'Staff' # Simplified for this model
                })

        def get_treatment_time(self):
            scenario = self.params['scenario']
            
            if scenario == 'Baseline Model (Homogeneous Staff)':
                # Use senior staff stats as baseline
                base_service_mins = self.params['senior_service_days'] * MINS_IN_DAY
                return np.random.exponential(base_service_mins)

            elif scenario == 'Experience-Based Model (Heterogeneous Staff)':
                is_senior = np.random.rand() < (self.params['senior_staff_mix'] / 100.0)
                if is_senior:
                    service_mins = self.params['senior_service_days'] * MINS_IN_DAY
                else:
                    service_mins = self.params['junior_service_days'] * MINS_IN_DAY
                return np.random.exponential(service_mins)

            elif scenario == 'Workload-Dependent Model (Dynamic Service Rates)':
                base_service_mins = self.params['senior_service_days'] * MINS_IN_DAY
                # How many patients per staff member?
                patients_per_staff = len(self.staff.users) / self.params['num_staff']
                # If load exceeds capacity factor, performance degrades
                if patients_per_staff > self.params['workload_factor_alpha']:
                    workload_factor = 1 + (patients_per_staff - self.params['workload_factor_alpha'])
                else:
                    workload_factor = 1
                return np.random.exponential(base_service_mins * workload_factor)
            
            return 0

    def patient_source(env, hospital, params):
        patient_id = 0
        interarrival_time_mins = MINS_IN_DAY / params['arrival_rate']
        while True:
            yield env.timeout(np.random.exponential(interarrival_time_mins))
            patient_id += 1
            env.process(hospital.patient(f'P{patient_id}'))

    # --- Simulation Setup ---
    np.random.seed(RANDOM_SEED)
    env = simpy.Environment()
    hospital_ward = Hospital(env, params)
    env.process(patient_source(env, hospital_ward, params))
    env.run(until=params['simulation_duration'] * MINS_IN_DAY)

    # --- Process Results ---
    patient_log_df = pd.DataFrame(hospital_ward.patient_log)
    occupancy_df = pd.DataFrame(hospital_ward.occupancy_log, columns=['Time', 'PatientsInSystem'])
    
    # Convert minutes to days for readability in results
    for col in ['WaitTime', 'TreatmentTime', 'TotalTime']:
        if col in patient_log_df.columns:
            patient_log_df[col] /= MINS_IN_DAY

    # --- Calculate KPIs ---
    kpis = {}
    total_arrivals = len(patient_log_df)
    if total_arrivals > 0:
        kpis['Blocking Probability (%)'] = (patient_log_df['Status'].str.contains('Balked').sum() / total_arrivals) * 100
    else:
        kpis['Blocking Probability (%)'] = 0

    treated_df = patient_log_df.dropna(subset=['WaitTime'])
    if not treated_df.empty:
        kpis['Average Wait for Staff (days)'] = treated_df['WaitTime'].mean()
        kpis['Average Patient Length of Stay (days)'] = treated_df['TotalTime'].mean()
        kpis['Total Patients Served'] = len(treated_df)
    else:
        kpis['Average Wait for Staff (days)'] = 0
        kpis['Average Patient Length of Stay (days)'] = 0
        kpis['Total Patients Served'] = 0

    # Calculate average bed occupancy
    if not occupancy_df.empty:
        total_patient_minutes = 0
        last_time = 0
        for _, row in occupancy_df.iterrows():
            total_patient_minutes += row['PatientsInSystem'] * (row['Time'] - last_time)
            last_time = row['Time']
        total_minutes = params['simulation_duration'] * MINS_IN_DAY
        avg_patients = total_patient_minutes / total_minutes if total_minutes > 0 else 0
        kpis['Average Bed Occupancy (%)'] = (avg_patients / params['num_beds']) * 100 if params['num_beds'] > 0 else 0
    else:
        kpis['Average Bed Occupancy (%)'] = 0

    return {
        'kpis': kpis,
        'patient_log_df': patient_log_df,
        'occupancy_df': occupancy_df
    }
