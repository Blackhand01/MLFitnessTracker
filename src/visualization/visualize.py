import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# seleziona solo le righe dive il valore della colonna "set" è 1.
# --------------------------------------------------------------

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])

plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
# --------------------------------------------------------------
# Adjust plot settings 
# migliora la visualizzazione dei dati 
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100
# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------