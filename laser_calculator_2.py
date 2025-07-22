# Final version: Full-featured Streamlit app with fluence threshold comparison,
# dual time display, laser pulse timeline, and thermal buildup chart.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("🔥 Laser Burn & Fluence Calculator")
st.markdown("Simulates a 50-pulse laser burn, calculates energy parameters, compares fluence to tissue thresholds, and visualizes thermal dynamics.")

# Sidebar inputs
st.sidebar.header("Laser Input Parameters")
D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1)
N = 50  # fixed for 1 burn
unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"])
tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01)
tau = tau_input * (1e-6 if unit == "µs" else 1e-9)

# Tissue threshold input
st.sidebar.markdown("---")
st.sidebar.subheader("Tissue Fluence Thresholds")
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

# Calculations
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

# Display core metrics
st.subheader("📊 Calculated Parameters")
col1, col2 = st.columns(2)
with col1:
    st.metric("Fluence", f"{F:.2f} J/cm²")
    st.metric("Peak Irradiance", f"{I_peak:.2e} W/cm²")
    st.metric("Peak Power", f"{P_peak:.2f} W")
    st.metric("Laser-On Time", f"{T_on:.4f} s")
with col2:
    st.metric("Exposure Time", f"{T_exposure:.2f} s")
    st.metric("Avg Irradiance", f"{I_avg:.2e} W/cm²")
    st.metric("Total Energy", f"{E_total:.2f} J")
    st.metric("Energy Density (avg)", f"{P_area_avg:.2e} W/cm²")

# Fluence vs Threshold Plot
st.subheader("⚠️ Fluence vs. Tissue Threshold")
fig_thresh, ax_thresh = plt.subplots()
ax_thresh.bar(["Your Fluence"], [F], color='green' if (threshold and F > threshold) else 'red')
if threshold:
    ax_thresh.axhline(y=threshold, color='blue', linestyle='--', label=f'{selected_tissue} Threshold ({threshold} J/cm²)')
    ax_thresh.legend()
ax_thresh.set_ylabel("Fluence (J/cm²)")
st.pyplot(fig_thresh)

# Optional message
if threshold:
    if F > threshold:
        st.success(f"✅ Your fluence **exceeds** the {selected_tissue} ablation threshold ({threshold} J/cm²).")
    else:
        st.warning(f"⚠️ Your fluence is **below** the {selected_tissue} threshold ({threshold} J/cm²).")

# Timeline & Thermal Simulation
st.subheader("📈 Timeline & Simulated Thermal Rise")

def simulate(N, f, tau, E, A, cooling_coef=0.05):
    interval = 1 / f
    pulse_times = np.array([i * interval for i in range(N)])
    t_res = 0.001
    t_max = pulse_times[-1] + 0.5
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

# Plot power timeline
fig1, ax1 = plt.subplots()
ax1.plot(t, power_profile, label="Laser Power (W)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Power (W)")
ax1.set_title("Laser Pulse Timeline (50 Pulses)")
ax1.legend()

# Plot temperature
fig2, ax2 = plt.subplots()
ax2.plot(t, temperature, color="red", label="Simulated Temperature Rise (a.u.)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("ΔT (a.u.)")
ax2.set_title("Thermal Buildup Over Time")
ax2.legend()

st.pyplot(fig1)
st.pyplot(fig2)
