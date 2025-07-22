import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🔬 Laser Parameter Calculator")
st.markdown("Calculate and visualize key laser parameters including fluence, irradiance, peak power, and compare to tissue ablation thresholds.")

st.sidebar.header("Laser Input Parameters")

# Basic Inputs
D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1)
N = st.sidebar.number_input("Number of Shots", value=50, min_value=1)

# Pulse duration with unit selection
unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"])
tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01)
tau = tau_input * (1e-6 if unit == "µs" else 1e-9)

# Tissue Threshold Comparison
st.sidebar.markdown("---")
st.sidebar.subheader("Ablation Thresholds")
tissue_thresholds = {
    "None": None,
    "Liver": 2.5,
    "Skin": 4.0,
    "Muscle": 3.5,
    "Brain": 2.0,
    "Cartilage": 6.0
}
selected_tissue = st.sidebar.selectbox("Compare with Tissue", list(tissue_thresholds.keys()))
threshold = tissue_thresholds[selected_tissue]

# Advanced toggle
advanced = st.sidebar.checkbox("Show Advanced Parameters", value=True)

# Calculations
E = E_mJ / 1000  # J
A = np.pi * (D / 20)**2  # cm²
F = E / A  # J/cm²
I_peak = E / (A * tau)  # W/cm²
I_avg = E * f / A  # W/cm²
P_peak = E / tau  # W
E_total = E * N  # J
T_total = N / f  # s
P_area_avg = E_total / (A * T_total)  # W/cm²
F_per_time = F / tau  # W·s/cm²
F_per_freq = F * f  # W/cm²

# Results
st.subheader("📊 Calculated Parameters")

col1, col2 = st.columns(2)
with col1:
    st.metric("Spot Area", f"{A:.4f} cm²")
    st.metric("Fluence", f"{F:.2f} J/cm²")
    st.metric("Irradiance (peak)", f"{I_peak:.2e} W/cm²")
    st.metric("Irradiance (avg)", f"{I_avg:.2e} W/cm²")
    st.metric("Peak Power", f"{P_peak:.2f} W")

with col2:
    st.metric("Total Energy", f"{E_total:.2f} J")
    st.metric("Exposure Time", f"{T_total:.2f} s")
    st.metric("Energy Density (avg)", f"{P_area_avg:.2e} W/cm²")
    st.metric("Pulse Energy Density Rate", f"{F_per_time:.2e} W·s/cm²")
    st.metric("Repetition Energy Rate", f"{F_per_freq:.2e} W/cm²")

# Graph: Fluence vs threshold
st.subheader("📉 Fluence vs. Tissue Threshold")

fig, ax = plt.subplots()
ax.bar(["Your Fluence"], [F], color='green' if (threshold and F > threshold) else 'red')
if threshold:
    ax.axhline(y=threshold, color='blue', linestyle='--', label=f'{selected_tissue} Threshold ({threshold} J/cm²)')
    ax.legend()
ax.set_ylabel("Fluence (J/cm²)")
st.pyplot(fig)
