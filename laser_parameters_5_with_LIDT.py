# Rebuild full laser calculator app script with added LIDT scaling, wavelength input, and comparison to scaled thresholds

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- TITLE AND DESCRIPTION ---
st.title("🔬 Comprehensive Laser Calculator with LIDT Scaling")
st.markdown(\"\"\"
This tool calculates key laser-tissue interaction parameters and compares them with ablation thresholds. 
It also includes LIDT scaling for pulse duration, wavelength, and beam diameter based on ISO 21254 recommendations.
\"\"\")

# --- INPUTS ---
st.sidebar.header("🧮 Input Parameters")
D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1)
N = st.sidebar.number_input("Number of Pulses", value=20, min_value=1)
unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"])
tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01)
tau = tau_input * (1e-6 if unit == "µs" else 1e-9)
wavelength = st.sidebar.number_input("Laser Wavelength (nm)", value=2940, min_value=100)

lock_axis = st.sidebar.checkbox("Lock X-axis scale to 1.0 s", value=False)

# --- THRESHOLD SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Tissue Threshold (Unscaled)")
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

# --- BASE CALCULATIONS ---
E = E_mJ / 1000  # J
A = np.pi * (D / 20)**2  # cm²
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

# --- LIDT SCALING ---
st.sidebar.markdown("---")
st.sidebar.subheader("📏 LIDT Reference Settings")

ref_tau = st.sidebar.number_input("Ref Pulse Duration (ns)", value=10.0)
ref_wavelength = st.sidebar.number_input("Ref Wavelength (nm)", value=1064)
ref_diameter = st.sidebar.number_input("Ref Beam Diameter (mm)", value=1.0)
ref_LIDT = st.sidebar.number_input("Ref LIDT Threshold (J/cm²)", value=5.0)

# Convert durations to consistent units (seconds)
ref_tau_s = ref_tau * 1e-9

# Apply LIDT scaling law
scaled_LIDT = ref_LIDT * ((tau / ref_tau_s) ** 0.5) * (wavelength / ref_wavelength) * ((ref_diameter / D) ** 2)

# --- OUTPUT SECTION ---
st.markdown("### 📐 Step 1: Calculated Parameters")

col1, col2 = st.columns(2)
with col1:
    st.metric("Spot Area", f"{A:.4f} cm²")
    st.metric("Fluence", f"{F:.2f} J/cm²")
    st.metric("Peak Irradiance", f"{I_peak:.2e} W/cm²")
    st.metric("Avg Irradiance", f"{I_avg:.2e} W/cm²")
    st.metric("Peak Power", f"{P_peak:.2f} W")
with col2:
    st.metric("Total Energy", f"{E_total:.2f} J")
    st.metric("Exposure Time", f"{T_exposure:.2f} s")
    st.metric("Laser-On Time", f"{T_on:.6f} s")
    st.metric("Energy Density (avg)", f"{P_area_avg:.2e} W/cm²")
    st.metric("Pulse Energy Density Rate", f"{F_per_time:.2e} W·s/cm²")

# --- COMPARISON SECTION ---
st.markdown("### ⚖️ Step 2: Fluence Threshold Comparison")

labels = ["Your Fluence"]
values = [F]
colors = ["green" if F > scaled_LIDT else "red"]

fig_thresh, ax_thresh = plt.subplots()
ax_thresh.bar(labels, values, color=colors)
ax_thresh.axhline(y=scaled_LIDT, color='blue', linestyle='--', label=f"Scaled LIDT ({scaled_LIDT:.2f} J/cm²)")
if ref_threshold:
    ax_thresh.axhline(y=ref_threshold, color='gray', linestyle=':', label=f"Tissue Threshold ({ref_threshold} J/cm²)")
ax_thresh.set_ylabel("Fluence (J/cm²)")
ax_thresh.set_title("Fluence vs LIDT Thresholds")
ax_thresh.legend()
st.pyplot(fig_thresh)

if F > scaled_LIDT:
    st.success("✅ Your fluence exceeds the scaled LIDT threshold.")
else:
    st.warning("⚠️ Your fluence is below the scaled LIDT threshold.")

# --- TIMELINE & THERMAL SIM ---
st.markdown("### 📈 Step 3: Pulse Timeline & Thermal Simulation")
st.markdown(f\"\"\"
This chart simulates a train of **{N} pulses** at **{f} Hz**, each of **{tau_input} {unit}** and delivering **{E:.3f} J**.

- Pulse spacing: {1/f:.4f} s
- Exposure time: {T_exposure:.4f} s
\"\"\")

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
