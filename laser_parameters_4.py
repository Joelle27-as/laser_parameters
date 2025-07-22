# Updated Streamlit app with full explanatory text, x-axis lock toggle, and labeled time axis

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("🔬 Comprehensive Laser Parameter Calculator with Explanations")
st.markdown("This tool calculates all key laser-tissue interaction parameters and explains each section step-by-step.")

# Sidebar: User inputs
st.sidebar.header("🔧 Input Parameters")

D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1)
N = st.sidebar.number_input("Number of Pulses", value=20, min_value=1)
unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"])
tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01)
tau = tau_input * (1e-6 if unit == "µs" else 1e-9)

lock_axis = st.sidebar.checkbox("Lock X-axis scale to 1.0 s", value=False)

# Threshold comparison
st.sidebar.markdown("---")
st.sidebar.subheader("Tissue Threshold Comparison")
tissue_thresholds = {
    "None": None,
    "Liver": 2.5,
    "Skin": 4.0,
    "Muscle": 3.5,
    "Brain": 2.0,
    "Cartilage": 6.0
}
selected_tissue = st.sidebar.selectbox("Tissue Type", list(tissue_thresholds.keys()))
threshold = tissue_thresholds[selected_tissue]

# --- Calculations ---
st.markdown("### 📐 Step 1: Calculated Laser Parameters")

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

st.markdown("""
These parameters describe how energy is delivered to the tissue, and how intense the laser exposure is. 
- **Fluence** tells you how much energy hits each cm².
- **Irradiance** and **power** relate to how intense the light is.
- **Exposure time** is the full time window where the laser is active.
- **Laser-on time** is the total time the laser actually emits light.
""")

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

# --- Fluence Threshold Comparison ---
st.markdown("### ⚠️ Step 2: Fluence vs. Tissue Threshold")

fig_thresh, ax_thresh = plt.subplots()
ax_thresh.bar(["Your Fluence"], [F], color='green' if (threshold and F > threshold) else 'red')
if threshold:
    ax_thresh.axhline(y=threshold, color='blue', linestyle='--', label=f'{selected_tissue} Threshold ({threshold} J/cm²)')
    ax_thresh.legend()
ax_thresh.set_ylabel("Fluence (J/cm²)")
st.pyplot(fig_thresh)

if threshold:
    if F > threshold:
        st.success(f"✅ Fluence exceeds {selected_tissue} ablation threshold ({threshold} J/cm²).")
    else:
        st.warning(f"⚠️ Fluence is below {selected_tissue} ablation threshold ({threshold} J/cm²).")

# --- Timeline & Thermal Simulation ---
st.markdown("### 📈 Step 3: Pulse Timeline & Thermal Buildup")
st.markdown(f"""
This chart simulates a train of **{N} pulses** delivered at **{f} Hz**.  
Each pulse lasts **{tau_input} {unit}** and delivers **{E:.3f} J**.

- Pulses are spaced {1/f:.4f} s apart.
- Total exposure time is {T_exposure:.4f} s.

**Higher frequency compresses the pulse train into a shorter time.**
""")

def simulate(N, f, tau, E, A, cooling_coef=0.05):
    interval = 1 / f
    pulse_times = np.array([i * interval for i in range(N)])
    t_res = min(tau / 10, interval / 20)
    t_max = max(pulse_times[-1] + 5 * tau, 1.0 if lock_axis else 0)
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

# Power timeline
fig1, ax1 = plt.subplots()
ax1.plot(t, power_profile, label="Laser Power (W)")
ax1.set_xlabel("Exposure Time (s)")
ax1.set_ylabel("Power (W)")
ax1.set_title(f"Laser Pulse Timeline ({int(N)} Pulses)")
ax1.legend()
if lock_axis:
    ax1.set_xlim(0, 1.0)

# Thermal rise
fig2, ax2 = plt.subplots()
ax2.plot(t, temperature, color="red", label="Simulated Temperature Rise (a.u.)")
ax2.set_xlabel("Exposure Time (s)")
ax2.set_ylabel("ΔT (a.u.)")
ax2.set_title("Thermal Buildup Over Time")
ax2.legend()

st.pyplot(fig1)
st.pyplot(fig2)
