# Rebuild a clean and tested version of the laser comparison app
# with safeguards, unique keys, and no silent failures

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🔬 Dual Laser Parameter Comparison")

st.markdown("This app lets you input parameters for two lasers and compare their physical and thermal characteristics.")

# --- Function to input one laser's parameters ---
def laser_input(label, suffix):
    st.subheader(f"{label} Inputs")
    D = st.number_input(f"{label} Spot Diameter (mm)", 0.01, 100.0, 0.5, key=f"D_{suffix}")
    E_mJ = st.number_input(f"{label} Energy per Pulse (mJ)", 0.0, 10000.0, 3.0, key=f"E_{suffix}")
    f = st.number_input(f"{label} Pulse Frequency (Hz)", 1, 100000, 10, key=f"f_{suffix}")
    N = st.number_input(f"{label} Number of Pulses", 1, 100000, 20, key=f"N_{suffix}")
    duration_unit = st.selectbox(f"{label} Pulse Duration Unit", ["µs", "ns"], key=f"unit_{suffix}")
    tau_val = st.number_input(f"{label} Pulse Duration ({duration_unit})", 0.01, 100000.0, 200.0, key=f"tau_{suffix}")
    tau = tau_val * (1e-6 if duration_unit == "µs" else 1e-9)
    wl = st.number_input(f"{label} Wavelength (nm)", 100, 10000, 2940, key=f"wl_{suffix}")
    return {"D": D, "E": E_mJ / 1000, "f": f, "N": N, "tau": tau, "wl": wl, "label": label}

# Input for both lasers
laser1 = laser_input("Laser 1", "L1")
laser2 = laser_input("Laser 2", "L2")

# --- Calculation function ---
def calculate(params):
    try:
        A = np.pi * (params["D"] / 20)**2  # mm² to cm²
        F = params["E"] / A
        I_peak = params["E"] / (A * params["tau"])
        I_avg = params["E"] * params["f"] / A
        P_peak = params["E"] / params["tau"]
        E_total = params["E"] * params["N"]
        T_exp = params["N"] / params["f"]
        T_on = params["N"] * params["tau"]
        P_area_avg = E_total / (A * T_exp)
        return {
            "Wavelength (nm)": params["wl"],
            "Spot Area (cm²)": A,
            "Fluence (J/cm²)": F,
            "Peak Irradiance (W/cm²)": I_peak,
            "Avg Irradiance (W/cm²)": I_avg,
            "Peak Power (W)": P_peak,
            "Total Energy (J)": E_total,
            "Exposure Time (s)": T_exp,
            "Laser-On Time (s)": T_on,
            "Avg Power Density (W/cm²)": P_area_avg
        }
    except Exception as e:
        st.error(f"Error in calculating for {params['label']}: {str(e)}")
        return {}

# Run calculations
res1 = calculate(laser1)
res2 = calculate(laser2)

# --- Display table ---
st.markdown("### 📊 Comparison Table")
if res1 and res2:
    df = pd.DataFrame({
        "Parameter": list(res1.keys()),
        "Laser 1": list(res1.values()),
        "Laser 2": list(res2.values())
    })
    st.dataframe(df)

# --- Plot fluence, irradiance, power ---
if res1 and res2:
    st.markdown("### 📈 Key Parameter Comparison")
    keys_to_plot = ["Fluence (J/cm²)", "Peak Irradiance (W/cm²)", "Avg Irradiance (W/cm²)", "Peak Power (W)"]
    values1 = [res1[k] for k in keys_to_plot]
    values2 = [res2[k] for k in keys_to_plot]
    x = np.arange(len(keys_to_plot))

    fig, ax = plt.subplots()
    bar_width = 0.35
    ax.bar(x - bar_width/2, values1, width=bar_width, label="Laser 1")
    ax.bar(x + bar_width/2, values2, width=bar_width, label="Laser 2")
    ax.set_xticks(x)
    ax.set_xticklabels(keys_to_plot, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Fluence & Power Characteristics")
    ax.legend()
    st.pyplot(fig)

# --- Pulse timelines ---
def simulate_timeline(N, f, tau, E):
    interval = 1 / f
    pulse_times = np.arange(N) * interval
    t_res = min(tau / 10, interval / 20)
    t = np.arange(0, pulse_times[-1] + 5 * tau, t_res)
    power = np.zeros_like(t)
    for pt in pulse_times:
        idx = (t >= pt) & (t < pt + tau)
        power[idx] = E / tau
    return t, power

t1, p1 = simulate_timeline(laser1["N"], laser1["f"], laser1["tau"], laser1["E"])
t2, p2 = simulate_timeline(laser2["N"], laser2["f"], laser2["tau"], laser2["E"])

st.markdown("### 🕒 Pulse Timeline")
fig2, ax2 = plt.subplots()
ax2.plot(t1, p1, label="Laser 1")
ax2.plot(t2, p2, label="Laser 2", linestyle="--")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Power (W)")
ax2.set_title("Laser Pulse Delivery Timeline")
ax2.legend()
st.pyplot(fig2)
