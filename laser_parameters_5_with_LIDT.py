# Fix syntax error by removing invalid triple quotes (\\""" should be just """)
# Update script to include export functionality (CSV)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# --- TITLE ---
st.title("🔬 Laser Calculator with LIDT Scaling and Export Options")

# --- INPUTS ---
st.sidebar.header("Input Parameters")
D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1)
N = st.sidebar.number_input("Number of Pulses", value=20, min_value=1)
unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"])
tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01)
tau = tau_input * (1e-6 if unit == "µs" else 1e-9)
wavelength = st.sidebar.number_input("Laser Wavelength (nm)", value=2940, min_value=100)
lock_axis = st.sidebar.checkbox("Lock X-axis scale to 1.0 s", value=False)

# --- LIDT SCALING INPUTS ---
st.sidebar.subheader("LIDT Scaling Inputs")
ref_tau = st.sidebar.number_input("Ref Pulse Duration (ns)", value=10.0)
ref_wavelength = st.sidebar.number_input("Ref Wavelength (nm)", value=1064)
ref_diameter = st.sidebar.number_input("Ref Beam Diameter (mm)", value=1.0)
ref_LIDT = st.sidebar.number_input("Ref LIDT Threshold (J/cm²)", value=5.0)

# --- THRESHOLDS ---
tissue_thresholds = {
    "None": None,
    "Liver": 2.5,
    "Skin": 4.0,
    "Muscle": 3.5,
    "Brain": 2.0,
    "Cartilage": 6.0
}
selected_tissue = st.sidebar.selectbox("Tissue Type", list(tissue_thresholds.keys()))
ref_threshold = tissue_thresholds[selected_tissue]

# --- CALCULATIONS ---
E = E_mJ / 1000
A = np.pi * (D / 20)**2
F = E / A
I_peak = E / (A * tau)
I_avg = E * f / A
P_peak = E / tau
E_total = E * N
T_exposure = N / f
T_on = N * tau
P_area_avg = E_total / (A * T_exposure)
F_per_time = F / tau
F_per_freq = F * f
ref_tau_s = ref_tau * 1e-9
scaled_LIDT = ref_LIDT * ((tau / ref_tau_s)**0.5) * (wavelength / ref_wavelength) * ((ref_diameter / D)**2)

# --- EXPORTABLE RESULTS ---
results = {
    "Spot Area (cm²)": A,
    "Fluence (J/cm²)": F,
    "Peak Irradiance (W/cm²)": I_peak,
    "Average Irradiance (W/cm²)": I_avg,
    "Peak Power (W)": P_peak,
    "Total Energy (J)": E_total,
    "Exposure Time (s)": T_exposure,
    "Laser-On Time (s)": T_on,
    "Avg Energy Density (W/cm²)": P_area_avg,
    "Fluence/Time (W·s/cm²)": F_per_time,
    "Scaled LIDT Threshold (J/cm²)": scaled_LIDT
}
df_export = pd.DataFrame(list(results.items()), columns=["Parameter", "Value"])

st.markdown("### 📤 Export Results")
selected_params = st.multiselect("Select parameters to export", options=df_export["Parameter"].tolist(), default=df_export["Parameter"].tolist())
filtered_df = df_export[df_export["Parameter"].isin(selected_params)]

csv = filtered_df.to_csv(index=False)
st.download_button("📥 Download Selected Results (CSV)", csv, "laser_results.csv", "text/csv")

# --- BASIC DISPLAY ---
st.markdown("### 📐 Calculated Parameters")
st.dataframe(df_export)

# Placeholder for future plots or additional content
