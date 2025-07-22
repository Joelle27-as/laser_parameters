# Full Streamlit app with burn timeline, dual time metrics, and thermal simulation

streamlit_code = """
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title and info
st.title("🔥 Laser Burn Calculator with Thermal Timeline")
st.markdown("Simulates a laser burn (50 pulses), calculates fluence and irradiance, and visualizes pulse timeline and thermal buildup.")

# Inputs
st.sidebar.header("Laser Input Parameters")
D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1)
N = 50  # fixed pulses per burn
unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"])
tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01)
tau = tau_input * (1e-6 if unit == "µs" else 1e-9)

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

# Metrics
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

# Simulation and visualization
st.subheader("📉 Pulse Timeline & Thermal Buildup (1 Burn, 50 Pulses)")

def simulate(N, f, tau, E, A, cooling_coef=0.05):
    interval = 1 / f
    pulse_times = np.array([i * interval for i in range(N)])
    t_res = 0.001
    t_max = pulse_times[-1] + 0.5
    t = np.arange(0, t_max, t_res)
    power = np.zeros_like(t)
    for pt in pulse_times:
        start = pt
        end = pt + tau
        idx = (t >= start) & (t <= end)
        power[idx] = E / tau
    heat = np.cumsum(power) * t_res / (A * np.sqrt(t + 0.001))
    cooling = np.arange(len(t)) * t_res * cooling_coef
    temperature = heat - cooling
    temperature[temperature < 0] = 0
    return t, power, temperature

t, power_profile, temperature = simulate(N, f, tau, E, A)

# Timeline plot
fig1, ax1 = plt.subplots()
ax1.plot(t, power_profile, label="Laser Power (W)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Power (W)")
ax1.set_title("Laser Pulse Timeline (50 Pulses)")
ax1.legend()

# Thermal plot
fig2, ax2 = plt.subplots()
ax2.plot(t, temperature, color="red", label="Simulated Temperature Rise (a.u.)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("ΔT (a.u.)")
ax2.set_title("Thermal Buildup Over Time")
ax2.legend()

st.pyplot(fig1)
st.pyplot(fig2)
