# Update the laser comparison app to handle extreme mismatches in pulse durations (e.g. ns vs µs)
# Add clamped time resolution and safe simulation limits

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🔬 Dual Laser Parameter Comparison (Fixed ns/µs Timeline Bug)")

st.markdown("Compare two laser systems safely, including those with mismatched pulse durations (e.g. 6 ns vs 100 µs).")

def laser_input(label, suffix):
    st.subheader(f"{label} Inputs")
    D = st.number_input(f"{label} Spot Diameter (mm)", 0.01, 100.0, 0.5, key=f"D_{suffix}")
    E_mJ = st.number_input(f"{label} Energy per Pulse (mJ)", 0.01, 10000.0, 3.0, key=f"E_{suffix}")
    f = st.number_input(f"{label} Pulse Frequency (Hz)", 1, 100000, 10, key=f"f_{suffix}")
    N = st.number_input(f"{label} Number of Pulses", 1, 100000, 20, key=f"N_{suffix}")
    unit = st.selectbox(f"{label} Pulse Duration Unit", ["µs", "ns"], key=f"unit_{suffix}")
    tau_val = st.number_input(f"{label} Pulse Duration ({unit})", 0.01, 100000.0, 100.0, key=f"tau_{suffix}")
    tau = tau_val * (1e-6 if unit == "µs" else 1e-9)
    wl = st.number_input(f"{label} Wavelength (nm)", 100, 10000, 2940, key=f"wl_{suffix}")
    return {"D": D, "E": E_mJ / 1000, "f": f, "N": N, "tau": tau, "wl": wl, "label": label}

# Input lasers
laser1 = laser_input("Laser 1", "L1")
laser2 = laser_input("Laser 2", "L2")

# Calculation
def calculate(params):
    try:
        D, E, f, N, tau = params["D"], params["E"], params["f"], params["N"], params["tau"]
        A = np.pi * (D / 20)**2
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
        st.error(f"Calculation error for {params['label']}: {e}")
        return None

res1 = calculate(laser1)
res2 = calculate(laser2)

# Comparison table
if res1 and res2:
    st.markdown("### 📊 Comparison Table")
    df = pd.DataFrame({
        "Parameter": list(res1.keys()),
        "Laser 1": list(res1.values()),
        "Laser 2": list(res2.values())
    })
    st.dataframe(df)

    # Plot key bar chart
    st.markdown("### 📈 Fluence & Power Comparison")
    keys = ["Fluence (J/cm²)", "Peak Irradiance (W/cm²)", "Avg Irradiance (W/cm²)", "Peak Power (W)"]
    vals1 = [res1[k] for k in keys]
    vals2 = [res2[k] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, vals1, 0.4, label="Laser 1")
    ax.bar(x + 0.2, vals2, 0.4, label="Laser 2")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45)
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    # Fixed timeline simulation
    st.markdown("### 🕒 Pulse Timeline")
    def simulate(N, f, tau, E, max_points=200000):
        try:
            interval = 1 / f
            pulses = np.arange(N) * interval
            t_step = max(min(tau / 10, interval / 20), 1e-9)  # clamp t_step ≥ 1 ns
            t_end = min(pulses[-1] + 5 * tau, 1.2 * N / f)
            t = np.arange(0, t_end, t_step)
            if len(t) > max_points:
                t = np.linspace(0, t_end, max_points)
            p = np.zeros_like(t)
            for pt in pulses:
                idx = (t >= pt) & (t < pt + tau)
                p[idx] = E / tau
            return t, p
        except Exception as e:
            st.error(f"Timeline error: {e}")
            return np.array([0]), np.array([0])

    t1, p1 = simulate(laser1["N"], laser1["f"], laser1["tau"], laser1["E"])
    t2, p2 = simulate(laser2["N"], laser2["f"], laser2["tau"], laser2["E"])

    fig2, ax2 = plt.subplots()
    ax2.plot(t1, p1, label="Laser 1")
    ax2.plot(t2, p2, label="Laser 2", linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Power (W)")
    ax2.set_title("Pulse Timeline")
    ax2.legend()
    st.pyplot(fig2)
