# Update script to include:
# - log-scale y-axis for the bar chart
# - multiselect box to let user choose which parameters to display

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🔬 Dual Laser Parameter Comparison")

st.markdown("Compare two laser systems side-by-side. Timeline plot removed for better stability with nanosecond durations.")

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

laser1 = laser_input("Laser 1", "L1")
laser2 = laser_input("Laser 2", "L2")

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

if res1 and res2:
    st.markdown("### 📊 Comparison Table")
    df = pd.DataFrame({
        "Parameter": list(res1.keys()),
        "Laser 1": list(res1.values()),
        "Laser 2": list(res2.values())
    })
    st.dataframe(df)

    st.markdown("### 📈 Select Parameters to Compare (Log Scale)")
    all_keys = [
        "Fluence (J/cm²)",
        "Peak Irradiance (W/cm²)",
        "Avg Irradiance (W/cm²)",
        "Peak Power (W)",
        "Total Energy (J)",
        "Avg Power Density (W/cm²)"
    ]
    selected_keys = st.multiselect("Select parameters:", all_keys, default=all_keys[:4])

    if selected_keys:
        vals1 = [res1[k] for k in selected_keys]
        vals2 = [res2[k] for k in selected_keys]
        x = np.arange(len(selected_keys))
        fig, ax = plt.subplots()
        ax.bar(x - 0.2, vals1, 0.4, label="Laser 1")
        ax.bar(x + 0.2, vals2, 0.4, label="Laser 2")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_keys, rotation=45, ha="right")
        ax.set_ylabel("Value (Log Scale)")
        ax.set_yscale("log")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Please select at least one parameter to visualize.")
