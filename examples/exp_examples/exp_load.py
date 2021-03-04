from veropt import load_experiment


experiment = load_experiment("Experiment_alpha_omega_2020_11_17_12_20_38.pkl")

experiment.run_rep()

experiment.plot_iteration()
