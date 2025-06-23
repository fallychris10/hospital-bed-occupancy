import simpy
import numpy as np
import pandas as pd

# --- Input Parameters (System Definition) ---
RANDOM_SEED = 42
SIM_TIME_MINUTES = 7 * 24 * 60  # Simulation time in minutes (1 week)

# λ (Lambda): Patient arrival rate
PATIENT_ARRIVAL_RATE = 6 / (24 * 60)  # Average of 6 patients per day

# μ (Mu): Ideal service rates
AVG_TREATMENT_TIME_SENIOR_MINUTES = 2.5 * 24 * 60
AVG_TREATMENT_TIME_JUNIOR_MINUTES = 3.5 * 24 * 60

# c: Number of staff
NUM_SENIOR_STAFF = 7
NUM_JUNIOR_STAFF = 8

# K: Number of beds
NUM_BEDS = 20

# α (Alpha): Workload scaling factor
ALPHA = 1.0  # Max patients a staff member can handle effectively before performance degrades

# Data collection
patient_log = []

def patient(env, name, hospital):
    """Patient process with workload-dependent service time."""
    arrival_time = env.now
    print(f"{arrival_time:.2f}: Patient {name} arrives.")

    if hospital['beds'].count >= hospital['beds'].capacity:
        print(f"{env.now:.2f}: Patient {name} BALKS as no beds are available.")
        patient_log.append({'name': name, 'arrival_time': arrival_time, 'status': 'Balked'})
        return

    with hospital['beds'].request() as bed_req:
        yield bed_req
        bed_assign_time = env.now
        print(f"{bed_assign_time:.2f}: Patient {name} is assigned a bed.")

        staff_request_time = env.now
        senior_req = hospital['staff_senior'].request()
        junior_req = hospital['staff_junior'].request()
        result = yield senior_req | junior_req

        staff_assign_time = env.now
        
        # Determine which staff was assigned and get their properties
        if senior_req in result:
            staff_type = 'Senior'
            staff_resource = hospital['staff_senior']
            staff_req_to_release = senior_req
            mu_ideal = 1.0 / AVG_TREATMENT_TIME_SENIOR_MINUTES
            # The other request (junior_req) is automatically cancelled by simpy
        else:  # Junior staff was assigned
            staff_type = 'Junior'
            staff_resource = hospital['staff_junior']
            staff_req_to_release = junior_req
            mu_ideal = 1.0 / AVG_TREATMENT_TIME_JUNIOR_MINUTES

        print(f"{staff_assign_time:.2f}: Patient {name} is assigned a {staff_type} staff member.")

        # --- Workload Degradation Logic ---
        n = hospital['beds'].count  # Current number of patients
        s = NUM_SENIOR_STAFF + NUM_JUNIOR_STAFF  # Total staff on duty
        if n == 0: n = 1

        mu_actual = mu_ideal * min(1, ALPHA * s / n)
        treatment_duration = np.random.exponential(1.0 / mu_actual)

        # Start treatment
        yield env.timeout(treatment_duration)

        # Release the staff member so they can treat another patient
        staff_resource.release(staff_req_to_release)

        treatment_end_time = env.now
        print(f"{treatment_end_time:.2f}: Patient {name} finishes treatment (discharged by {staff_type}).")

        patient_log.append({
            'name': name, 'arrival_time': arrival_time,
            'waited_for_bed': bed_assign_time - arrival_time,
            'waited_for_staff': staff_assign_time - bed_assign_time,
            'treatment_duration': treatment_end_time - staff_assign_time,
            'staff_type': staff_type, 'status': 'Discharged'
        })

def patient_generator(env, arrival_rate, hospital):
    patient_id = 0
    while True:
        yield env.timeout(np.random.exponential(1.0 / arrival_rate))
        patient_id += 1
        env.process(patient(env, f'P{patient_id}', hospital))

# --- Simulation Setup and Run ---
print("--- Running Simulation for Objective 3: Staff Workload Impact ---")
np.random.seed(RANDOM_SEED)

env = simpy.Environment()
hospital = {
    'beds': simpy.Resource(env, capacity=NUM_BEDS),
    'staff_senior': simpy.Resource(env, capacity=NUM_SENIOR_STAFF),
    'staff_junior': simpy.Resource(env, capacity=NUM_JUNIOR_STAFF)
}

env.process(patient_generator(env, PATIENT_ARRIVAL_RATE, hospital))
env.run(until=SIM_TIME_MINUTES)

# --- Analysis and Output ---
print("\nSimulation finished.")
log_df = pd.DataFrame(patient_log)
log_df.to_csv('simulation_log_objective3.csv', index=False)
print(f"Saved simulation log to simulation_log_objective3.csv")

print("\n--- Simulation Results (Objective 3) ---")
discharged_patients = log_df[log_df['status'] == 'Discharged']
balked_patients = log_df[log_df['status'] == 'Balked']

if not discharged_patients.empty:
    avg_wait_for_bed = discharged_patients['waited_for_bed'].mean()
    avg_wait_for_staff = discharged_patients['waited_for_staff'].mean()
    print(f"Average wait time for a bed: {avg_wait_for_bed:.2f} minutes")
    print(f"Average wait time for staff: {avg_wait_for_staff:.2f} minutes")

    print("\n--- Treatment Duration by Staff Type ---")
    avg_duration_by_staff = discharged_patients.groupby('staff_type')['treatment_duration'].mean()
    print(avg_duration_by_staff)
else:
    print("No patients were discharged.")

total_patients = len(log_df)
if total_patients > 0:
    blocking_prob = len(balked_patients) / total_patients
    print(f"\nBlocking Probability (patients turned away): {blocking_prob:.2%}")

print(f"Total patients arrived: {total_patients}")
print(f"Total patients discharged: {len(discharged_patients)}")
print(f"Total patients balked (no beds): {len(balked_patients)}")
