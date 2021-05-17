from veropt import load_optimiser
from veropt.gui import veropt_gui

# optimiser = load_optimiser("Optimiser_PredefinedTestFunction_2021_04_20_14_47_41.pkl")
optimiser = load_optimiser("Optimiser_PredefinedTestFunction_2021_03_26_17_18_58.pkl")
# optimiser = load_optimiser("Optimiser_PredefinedTestFunction_2021_04_24_18_20_17.pkl")

veropt_gui.run(optimiser)



