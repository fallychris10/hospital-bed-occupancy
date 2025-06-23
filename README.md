# Hospital Bed & Staffing Simulation Dashboard

An interactive dashboard for analyzing patient flow and hospital resource management using queuing theory. This project provides a comprehensive visualization tool for understanding the relationships between staff experience, workload, and system performance metrics.

## Features

- Interactive simulation of hospital bed occupancy
- Analysis of different staffing scenarios (Baseline, Experience-Based, Workload-Dependent)
- Real-time performance metrics visualization
- Sensitivity analysis tools
- Comparative scenario analysis

## Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Data Sources

The simulation is based on an M/M/c/K queuing model with the following scenarios:
- Baseline Model (Homogeneous Staff)
- Experience-Based Model (Heterogeneous Staff)
- Workload-Dependent Model (Dynamic Service Rates)

## Usage

1. Select a simulation model from the sidebar
2. Adjust system parameters
3. Run the simulation
4. Analyze results through various visualizations and metrics

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

MIT License 