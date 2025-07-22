# Generate a safe and crash-proof version of the laser comparison app with input validation and fallback handling

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🔬 Dual Laser Parameter Comparison (Safe Version)")

st.markdown("This app lets you input parameters for two lasers and compare their behavior, with safeguards for invalid input.")

# --- Safe Input Function ---
def laser_input(label, suffix):
    st.subheader(f"{label} Inputs")
    D = st.number_input(f"{label} Spot Diameter (mm)", 0.01, 100.0, 0.5, key=f"D_{suffix}")
    E_mJ = st.number_input(f"{label} Energy per Pulse (mJ)", 0.01, 10000.0, 3.0, key=f"E_{suffix}")
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

# --- Safe Calculation Function ---
def calculate(params):
    try:
        D, E, f, N, tau = params["D"], params["E"], params["f"], params["N"], params["tau"]
        A = np.pi * (D / 20)**2  # mm² to cm²
        if A <= 0 or tau <= 0 or f <= 0 or N <= 0 or E <= 0:
            raise ValueError("Invalid non-positive input detected.")
        F = E / A
        I_peak = E / (A * tau)
        I_avg = E * f / A
        P_peak = E / tau
        E_total = E * N
        T_exp = N / f
        T_on = N * tau
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
        st.error(f"❌ Error in {params['label']} calculations: {e}")
        return None

# --- Run Calculations ---
res1 = calculate(laser1)
res2 = calculate(laser2)

# --- Display Comparison ---
if res1 and res2:
    st.markdown("### 📊 Numerical Comparison")
    df = pd.DataFrame({
        "Parameter": list(res1.keys()),
        "Laser 1": list(res1.values()),
        "Laser 2": list(res2.values())
    })
    st.dataframe(df)

    # --- Plot Fluence & Irradiance ---
    st.markdown("### 📈 Fluence & Power Comparison")
    keys = ["Fluence (J/cm²)", "Peak Irradiance (W/cm²)", "Avg Irradiance (W/cm²)", "Peak Power (W)"]
    vals1 = [res1[k] for k in keys]
    vals2 = [res2[k] for k in keys]
    x = np.arange(len(keys))

    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(x - width/2, vals1, width, label="Laser 1")
    ax.bar(x + width/2, vals2, width, label="Laser 2")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Fluence & Power Comparison")
    ax.legend()
    st.pyplot(fig)

    # --- Timeline Plots ---
    st.markdown("### 🕒 Pulse Timeline")

    def simulate_timeline(N, f, tau, E):
        try:
            interval = 1 / f
            pulse_times = np.arange(N) * interval
            t_res = min(tau / 10, interval / 20)
            t = np.arange(0, pulse_times[-1] + 5 * tau, t_res)
            power = np.zeros_like(t)
            for pt in pulse_times:
                idx = (t >= pt) & (t < pt + tau)
                power[idx] = E / tau
            return t, power
        except Exception as e:
            st.error(f"Timeline simulation error: {e}")
            return np.array([0]), np.array([0])

    t1, p1 = simulate_timeline(laser1["N"], laser1["f"], laser1["tau"], laser1["E"])
    t2, p2 = simulate_timeline(laser2["N"], laser2["f"], laser2["tau"], laser2["E"])

    fig2, ax2 = plt.subplots()
    ax2.plot(t1, p1, label="Laser 1")
    ax2.plot(t2, p2, label="Laser 2", linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Power (W)")
    ax2.set_title("Pulse Delivery Timeline")
    ax2.legend()
    st.pyplot(fig2)
