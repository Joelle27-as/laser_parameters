# Generate a clean version of the script with:
# - LIDT scaling removed
# - Wavelength input kept
# - Export option retained
# - Graphical plots (fluence threshold, timeline, thermal) restored

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- TITLE ---
st.title("🔬 Laser Calculator with Export and Visual Analysis")

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

# --- TISSUE THRESHOLD ---
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

# --- EXPORTABLE RESULTS ---
results = {
    "Laser Wavelength (nm)": wavelength,
    "Spot Area (cm²)": A,
    "Fluence (J/cm²)": F,
    "Peak Irradiance (W/cm²)": I_peak,
    "Average Irradiance (W/cm²)": I_avg,
    "Peak Power (W)": P_peak,
    "Total Energy (J)": E_total,
    "Exposure Time (s)": T_exposure,
    "Laser-On Time (s)": T_on,
    "Avg Energy Density (W/cm²)": P_area_avg,
    "Fluence/Time (W·s/cm²)": F_per_time
}
df_export = pd.DataFrame(list(results.items()), columns=["Parameter", "Value"])

st.markdown("### 📤 Export Results")
selected_params = st.multiselect("Select parameters to export", options=df_export["Parameter"].tolist(), default=df_export["Parameter"].tolist())
filtered_df = df_export[df_export["Parameter"].isin(selected_params)]

csv = filtered_df.to_csv(index=False)
st.download_button("📥 Download Selected Results (CSV)", csv, "laser_results.csv", "text/csv")

# --- DISPLAY PARAMETERS ---
st.markdown("### 📐 Calculated Parameters")
st.dataframe(df_export)

# --- COMPARISON GRAPH ---
st.markdown("### ⚖️ Fluence vs Tissue Threshold")
labels = ["Your Fluence"]
values = [F]
colors = ["green" if (ref_threshold and F > ref_threshold) else "red"]

fig_thresh, ax_thresh = plt.subplots()
ax_thresh.bar(labels, values, color=colors)
if ref_threshold:
    ax_thresh.axhline(y=ref_threshold, color='gray', linestyle='--', label=f"Tissue Threshold ({ref_threshold} J/cm²)")
ax_thresh.set_ylabel("Fluence (J/cm²)")
ax_thresh.set_title("Fluence vs Tissue Threshold")
ax_thresh.legend()
st.pyplot(fig_thresh)

# --- TIMELINE & THERMAL SIM ---
st.markdown("### 📈 Pulse Timeline & Thermal Simulation")

def simulate(N, f, tau, E, A, cooling_coef=0.05):
    interval = 1 / f
    pulse_times = np.array([i * interval for i in range(N)])
    t_res = min(tau / 10, interval / 20)
    t_max = max(pulse_times[-1] + 5 * tau, 1.0 if lock_axis else pulse_times[-1] + 5 * tau)
    t = np.arange(0, t_max, t_res)
    power = np.zeros_like(t)
    for pt in pulse_times:
        idx = (t >= pt) & (t <= pt + tau)
        power[idx] = E / tau
    heat = np.cumsum(power) * t_res / (A * np.sqrt(t + 0.001))
    cooling = np.arange(len(t)) * t_res * cooling_coef
    temperature = heat - cooling
    temperature[temperature < 0] = 0
    return t, power, temperature

t, power_profile, temperature = simulate(N, f, tau, E, A)

fig1, ax1 = plt.subplots()
ax1.plot(t, power_profile, label="Laser Power (W)")
ax1.set_xlabel("Exposure Time (s)")
ax1.set_ylabel("Power (W)")
ax1.set_title("Laser Pulse Timeline")
ax1.legend()
if lock_axis:
    ax1.set_xlim(0, 1.0)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(t, temperature, color="red", label="Simulated Temperature Rise (a.u.)")
ax2.set_xlabel("Exposure Time (s)")
ax2.set_ylabel("ΔT (a.u.)")
ax2.set_title("Thermal Buildup Over Time")
ax2.legend()
st.pyplot(fig2)
