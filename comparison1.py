# Generate a new Streamlit app script for comparing two lasers side by side

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("🔬 Laser Comparison Tool")

st.markdown("Use this app to input parameters for **two lasers** and visually compare their behavior and power delivery.")

# --- Input function for laser parameters ---
def laser_inputs(label_prefix):
    st.subheader(f"{label_prefix} Parameters")
    D = st.number_input(f"{label_prefix} Spot Diameter (mm)", value=0.5, min_value=0.01, key=f"{label_prefix}_D")
    E_mJ = st.number_input(f"{label_prefix} Energy per Pulse (mJ)", value=3.0, min_value=0.0, key=f"{label_prefix}_E")
    f = st.number_input(f"{label_prefix} Pulse Frequency (Hz)", value=10, min_value=1, key=f"{label_prefix}_f")
    N = st.number_input(f"{label_prefix} Number of Pulses", value=20, min_value=1, key=f"{label_prefix}_N")
    unit = st.selectbox(f"{label_prefix} Pulse Duration Unit", ["µs", "ns"], key=f"{label_prefix}_unit")
    tau_input = st.number_input(f"{label_prefix} Pulse Duration ({unit})", value=200.0, min_value=0.01, key=f"{label_prefix}_tau")
    tau = tau_input * (1e-6 if unit == "µs" else 1e-9)
    wavelength = st.number_input(f"{label_prefix} Wavelength (nm)", value=2940, min_value=100, key=f"{label_prefix}_wl")
    return {"D": D, "E": E_mJ / 1000, "f": f, "N": N, "tau": tau, "wavelength": wavelength, "label": label_prefix}

# Input for both lasers
laser1 = laser_inputs("Laser 1")
laser2 = laser_inputs("Laser 2")

def compute_laser(params):
    A = np.pi * (params["D"] / 20)**2
    F = params["E"] / A
    I_peak = params["E"] / (A * params["tau"])
    I_avg = params["E"] * params["f"] / A
    P_peak = params["E"] / params["tau"]
    E_total = params["E"] * params["N"]
    T_exposure = params["N"] / params["f"]
    T_on = params["N"] * params["tau"]
    P_area_avg = E_total / (A * T_exposure)
    return {
        "Fluence (J/cm²)": F,
        "Peak Irradiance (W/cm²)": I_peak,
        "Average Irradiance (W/cm²)": I_avg,
        "Peak Power (W)": P_peak,
        "Total Energy (J)": E_total,
        "Exposure Time (s)": T_exposure,
        "Laser-On Time (s)": T_on,
        "Avg Power Density (W/cm²)": P_area_avg,
        "Spot Area (cm²)": A,
        "Wavelength (nm)": params["wavelength"]
    }

# Compute both lasers
res1 = compute_laser(laser1)
res2 = compute_laser(laser2)

# Display comparison table
st.markdown("### 📊 Numerical Comparison")
df_compare = pd.DataFrame({
    "Parameter": list(res1.keys()),
    "Laser 1": list(res1.values()),
    "Laser 2": [res2[k] for k in res1.keys()]
})
st.dataframe(df_compare)

# --- Plot comparisons ---
st.markdown("### 📈 Fluence & Irradiance Comparison")

fig1, ax1 = plt.subplots()
bar_width = 0.35
x_labels = list(res1.keys())[:4]
x = np.arange(len(x_labels))
y1 = [res1[k] for k in x_labels]
y2 = [res2[k] for k in x_labels]
ax1.bar(x - bar_width/2, y1, bar_width, label='Laser 1')
ax1.bar(x + bar_width/2, y2, bar_width, label='Laser 2')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=45)
ax1.set_ylabel("Value")
ax1.set_title("Fluence, Peak Irradiance, Avg Irradiance, Peak Power")
ax1.legend()
st.pyplot(fig1)

# --- Timeline pulse delivery comparison ---
st.markdown("### 🕒 Pulse Timeline Comparison")

def simulate_timeline(N, f, tau, E):
    interval = 1 / f
    pulse_times = np.array([i * interval for i in range(N)])
    t_res = min(tau / 10, interval / 20)
    t_max = pulse_times[-1] + 5 * tau
    t = np.arange(0, t_max, t_res)
    power = np.zeros_like(t)
    for pt in pulse_times:
        idx = (t >= pt) & (t <= pt + tau)
        power[idx] = E / tau
    return t, power

t1, power1 = simulate_timeline(laser1["N"], laser1["f"], laser1["tau"], laser1["E"])
t2, power2 = simulate_timeline(laser2["N"], laser2["f"], laser2["tau"], laser2["E"])

fig2, ax2 = plt.subplots()
ax2.plot(t1, power1, label="Laser 1")
ax2.plot(t2, power2, label="Laser 2")
ax2.set_xlabel("Exposure Time (s)")
ax2.set_ylabel("Power (W)")
ax2.set_title("Pulse Timeline")
ax2.legend()
st.pyplot(fig2)
