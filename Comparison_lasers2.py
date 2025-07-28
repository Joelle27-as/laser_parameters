
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Laser Parameter Comparison", layout="wide")
st.title("🔬 Dual Laser Parameter Comparison Tool")

# -------------------------------
# 🔢 Input Parameters for Laser 1
# -------------------------------
st.sidebar.header("Laser 1 Parameters")
D1 = st.sidebar.number_input("Spot Diameter (mm)", min_value=0.01, value=0.51, key="D1")
E1_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", min_value=0.001, value=2.0, key="E1")
f1 = st.sidebar.number_input("Pulse Frequency (Hz)", min_value=1, value=20, key="f1")
N1 = st.sidebar.number_input("Number of Pulses", min_value=1, value=10, key="N1")
tau1_val = st.sidebar.number_input("Pulse Duration", min_value=0.001, value=100.0, key="tau1_val")
tau1_unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"], key="tau1_unit")
λ1 = st.sidebar.number_input("Wavelength (nm)", min_value=100, value=2940, key="λ1")

# -------------------------------
# 🔢 Input Parameters for Laser 2
# -------------------------------
st.sidebar.header("Laser 2 Parameters")
D2 = st.sidebar.number_input("Spot Diameter (mm)", min_value=0.01, value=0.81, key="D2")
E2_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", min_value=0.001, value=3.8, key="E2")
f2 = st.sidebar.number_input("Pulse Frequency (Hz)", min_value=1, value=20, key="f2")
N2 = st.sidebar.number_input("Number of Pulses", min_value=1, value=10, key="N2")
tau2_val = st.sidebar.number_input("Pulse Duration", min_value=0.001, value=6.0, key="tau2_val")
tau2_unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"], key="tau2_unit")
λ2 = st.sidebar.number_input("Wavelength (nm)", min_value=100, value=2940, key="λ2")

# -------------------------------
# 📊 Calculations
# -------------------------------
def calculate_params(D, E_mJ, f, N, tau_val, tau_unit):
    E = E_mJ / 1000  # J
    A = np.pi * (D / 10 / 2) ** 2  # cm²
    tau = tau_val * (1e-6 if tau_unit == "µs" else 1e-9)
    fluence = E / A
    I_peak = E / (A * tau)
    I_avg = E * f / A
    total_energy = E * N
    exposure_time = N / f
    return {
        "Spot Area (cm²)": A,
        "Fluence (J/cm²)": fluence,
        "Peak Irradiance (W/cm²)": I_peak,
        "Avg Irradiance (W/cm²)": I_avg,
        "Total Energy (J)": total_energy,
        "Exposure Time (s)": exposure_time,
        "Pulse Duration (s)": tau,
        "Energy (J)": E,
        "Frequency (Hz)": f
    }

res1 = calculate_params(D1, E1_mJ, f1, N1, tau1_val, tau1_unit)
res2 = calculate_params(D2, E2_mJ, f2, N2, tau2_val, tau2_unit)

# -------------------------------
# 📋 Side-by-Side Summary Table
# -------------------------------
st.markdown("### 📋 Laser Parameter Summary Table")

summary_df = pd.DataFrame({
    "Parameter": list(res1.keys()),
    "Laser 1": list(res1.values()),
    "Laser 2": [res2[k] for k in res1.keys()]
})

st.dataframe(summary_df, use_container_width=True)

csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Laser Comparison Table", csv, "laser_comparison_table.csv", "text/csv")
# -------------------------------
# 📚 Parameter Definitions
# -------------------------------
st.markdown("### 📚 Parameter Definitions")

with st.expander("Click to view definitions and formulas"):
    st.markdown(r"""
- **Fluence (J/cm²)**:  
  \[
  \text{Fluence} = \frac{E}{A}
  \]  
  Energy delivered per unit area. Determines how much energy reaches a specific surface area.

- **Peak Irradiance (W/cm²)**:  
  \[
  I_{\text{peak}} = \frac{E}{A \cdot \tau}
  \]  
  Intensity of the laser during each pulse. Critical for ablation and nonlinear effects.

- **Average Irradiance (W/cm²)**:  
  \[
  I_{\text{avg}} = \frac{E \cdot f}{A}
  \]  
  Describes how much energy is delivered per second over an area. Useful for thermal effects.

- **Total Energy (J)**:  
  \[
  E_{\text{total}} = E \cdot N
  \]  
  Total energy delivered over all pulses.

- **Exposure Time (s)**:  
  \[
  t = \frac{N}{f}
  \]  
  Total time over which all laser pulses are delivered.

- **Pulse Duration (s)**:  
  Time span of a single laser pulse. Shorter durations lead to higher peak irradiance.
    """)
# -------------------------------
# 📈 Log Comparison Plot
# -------------------------------
st.markdown("### 📈 Parameter Comparison")
params = ["Fluence (J/cm²)", "Peak Irradiance (W/cm²)", "Avg Irradiance (W/cm²)", "Total Energy (J)", "Exposure Time (s)"]
selected_params = st.multiselect("Select parameters to plot:", params, default=params)

fig, ax = plt.subplots()
index = np.arange(len(selected_params))
bar_width = 0.35

vals1 = [res1[p] for p in selected_params]
vals2 = [res2[p] for p in selected_params]

ax.bar(index, vals1, bar_width, label='Laser 1')
ax.bar(index + bar_width, vals2, bar_width, label='Laser 2')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(selected_params, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylabel("Log Scale")
ax.set_title("Comparison of Laser Parameters")
ax.legend()
st.pyplot(fig)

# -------------------------------
# 🔺 Match Peak Irradiance (Iₚₑₐₖ)
# -------------------------------
st.markdown("### 🔺 Match Peak Irradiance (Iₚₑₐₖ)")
option_peak = st.selectbox("Adjust in Laser 1:", ["Energy", "Pulse Duration", "Spot Diameter"], key="match_peak")

I_target = res2["Peak Irradiance (W/cm²)"]
A1 = res1["Spot Area (cm²)"]
tau1 = res1["Pulse Duration (s)"]
E1 = res1["Energy (J)"]

if option_peak == "Energy":
    E_new = I_target * A1 * tau1
    st.success(f"To match Iₚₑₐₖ, set Laser 1 Energy to **{E_new * 1000:.2f} mJ**")
elif option_peak == "Pulse Duration":
    tau_new = E1 / (I_target * A1)
    unit = "µs" if tau_new >= 1e-6 else "ns"
    tau_val = tau_new * 1e6 if unit == "µs" else tau_new * 1e9
    st.success(f"To match Iₚₑₐₖ, set Laser 1 Pulse Duration to **{tau_val:.2f} {unit}**")
elif option_peak == "Spot Diameter":
    A_new = E1 / (I_target * tau1)
    D_new = 2 * np.sqrt(A_new / np.pi) * 10
    st.success(f"To match Iₚₑₐₖ, set Laser 1 Spot Diameter to **{D_new:.2f} mm**")

# -------------------------------
# 🔶 Match Fluence
# -------------------------------
st.markdown("### 🔶 Match Fluence (J/cm²)")
option_fluence = st.selectbox("Adjust in Laser 1 (Fluence):", ["Energy", "Spot Diameter"], key="match_fluence")

fluence_target = res2["Fluence (J/cm²)"]

if option_fluence == "Energy":
    E_new = fluence_target * A1
    st.success(f"To match Fluence, set Laser 1 Energy to **{E_new * 1000:.2f} mJ**")
elif option_fluence == "Spot Diameter":
    A_new = E1 / fluence_target
    D_new = 2 * np.sqrt(A_new / np.pi) * 10
    st.success(f"To match Fluence, set Laser 1 Spot Diameter to **{D_new:.2f} mm**")

# -------------------------------
# 🔷 Match Average Irradiance
# -------------------------------
st.markdown("### 🔷 Match Average Irradiance (W/cm²)")
option_avg = st.selectbox("Adjust in Laser 1 (Avg Irradiance):", ["Energy", "Spot Diameter"], key="match_avg")

I_avg_target = res2["Avg Irradiance (W/cm²)"]

if option_avg == "Energy":
    E_new = (I_avg_target * A1) / f1
    st.success(f"To match Avg Irradiance, set Laser 1 Energy to **{E_new * 1000:.2f} mJ**")
elif option_avg == "Spot Diameter":
    A_new = (E1 * f1) / I_avg_target
    D_new = 2 * np.sqrt(A_new / np.pi) * 10
    st.success(f"To match Avg Irradiance, set Laser 1 Spot Diameter to **{D_new:.2f} mm**")
