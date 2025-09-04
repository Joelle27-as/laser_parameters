import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Global Sidebar (shared)
# ----------------------------
def shared_sidebar():
    st.sidebar.title("🔧 Global Settings")
    st.sidebar.info("Notes and global preferences visible in all tabs.")
    st.sidebar.text_area("Notes (applies to both tools)", key="global_notes", placeholder="Write any notes here…")
    st.sidebar.divider()
    return {}

# ----------------------------
# App 1: Laser Calculator
# ----------------------------
def app_laser_calculator():
    st.header("🔬 Laser Calculator with Export and Visual Analysis")

    # --- Tab-specific sidebar controls ---
    st.sidebar.subheader("Laser Calculator • Parameters")
    D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01, key="calc_D")
    E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0, key="calc_E")
    f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1, key="calc_f")
    N = st.sidebar.number_input("Number of Pulses", value=20, min_value=1, key="calc_N")
    unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"], key="calc_unit")
    tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01, key="calc_tau")
    tau = tau_input * (1e-6 if unit == "µs" else 1e-9)
    wavelength = st.sidebar.number_input("Laser Wavelength (nm)", value=2940, min_value=100, key="calc_wavelength")

    # --- Calculations ---
    E = E_mJ / 1000.0               # J
    A_cm2 = np.pi * (D / 20.0) ** 2  # cm²
    F = E / A_cm2                    # J/cm²
    I_peak = E / (A_cm2 * tau)       # W/cm²
    I_avg = E * f / A_cm2            # W/cm²
    P_peak = E / tau                 # W
    E_total = E * N                  # J
    T_exposure = N / f               # s
    T_on = N * tau                   # s
    P_area_avg = E_total / (A_cm2 * T_exposure)  # W/cm²
    F_per_time = F / tau             # W·s/cm²

    # --- Exportable results ---
    results = {
        "Laser Wavelength (nm)": wavelength,
        "Spot Area (cm²)": A_cm2,
        "Fluence (J/cm²)": F,
        "Peak Irradiance (W/cm²)": I_peak,
        "Average Irradiance (W/cm²)": I_avg,
        "Peak Power (W)": P_peak,
        "Total Energy (J)": E_total,
        "Exposure Time (s)": T_exposure,
        "Laser-On Time (s)": T_on,
        "Avg Energy Density (W/cm²)": P_area_avg,
        "Fluence/Time (W·s/cm²)": F_per_time
    }
    df_export = pd.DataFrame(list(results.items()), columns=["Parameter", "Value"])

    st.markdown("### 📤 Export Results")
    selected_params = st.multiselect(
        "Select parameters to export",
        options=df_export["Parameter"].tolist(),
        default=df_export["Parameter"].tolist(),
        key="calc_export"
    )
    filtered_df = df_export[df_export["Parameter"].isin(selected_params)]
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download Selected Results (CSV)", csv, "laser_results.csv", "text/csv")

    st.markdown("### 📐 Calculated Parameters")
    st.dataframe(df_export, use_container_width=True)

    # --- Fluence vs simple tissue threshold ---
    st.markdown("### ⚖️ Fluence vs Tissue Threshold")
    tissue_thresholds = {"None": None, "Liver": 2.5, "Skin": 4.0, "Muscle": 3.5, "Brain": 2.0, "Cartilage": 6.0}
    selected_tissue = st.selectbox("Tissue Type", list(tissue_thresholds.keys()), key="calc_tissue_top")
    ref_threshold = tissue_thresholds[selected_tissue]
    labels = ["Your Fluence"]
    values = [F]
    colors = ["green" if (ref_threshold and F > ref_threshold) else "red"]

    fig_thresh, ax_thresh = plt.subplots()
    ax_thresh.bar(labels, values, color=colors)
    if ref_threshold:
        ax_thresh.axhline(y=ref_threshold, color='gray', linestyle='--', label=f"Tissue Threshold ({ref_threshold} J/cm²)")
    ax_thresh.set_ylabel("Fluence (J/cm²)")
    ax_thresh.set_title("Fluence vs Tissue Threshold")
    ax_thresh.legend()
    st.pyplot(fig_thresh)

# ----------------------------
# App 2: Dual Laser Comparison
# ----------------------------
def app_laser_comparison():
    st.header("🔬 Dual Laser Parameter Comparison Tool")

    # Tab-specific sidebar controls
    st.sidebar.subheader("Laser Comparison • Laser 1")
    D1 = st.sidebar.number_input("Spot Diameter (mm)", min_value=0.01, value=0.51, key="cmp_D1")
    E1_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", min_value=0.001, value=2.0, key="cmp_E1")
    f1 = st.sidebar.number_input("Pulse Frequency (Hz)", min_value=1, value=20, key="cmp_f1")
    N1 = st.sidebar.number_input("Number of Pulses", min_value=1, value=10, key="cmp_N1")
    tau1_val = st.sidebar.number_input("Pulse Duration", min_value=0.001, value=100.0, key="cmp_tau1_val")
    tau1_unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"], key="cmp_tau1_unit")
    _λ1 = st.sidebar.number_input("Wavelength (nm)", min_value=100, value=2940, key="cmp_λ1")

    st.sidebar.subheader("Laser Comparison • Laser 2")
    D2 = st.sidebar.number_input("Spot Diameter (mm)", min_value=0.01, value=0.81, key="cmp_D2")
    E2_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", min_value=0.001, value=3.8, key="cmp_E2")
    f2 = st.sidebar.number_input("Pulse Frequency (Hz)", min_value=1, value=20, key="cmp_f2")
    N2 = st.sidebar.number_input("Number of Pulses", min_value=1, value=10, key="cmp_N2")
    tau2_val = st.sidebar.number_input("Pulse Duration", min_value=0.001, value=6.0, key="cmp_tau2_val")
    tau2_unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"], key="cmp_tau2_unit")
    _λ2 = st.sidebar.number_input("Wavelength (nm)", min_value=100, value=2940, key="cmp_λ2")

    def calculate_params(D, E_mJ, f, N, tau_val, tau_unit):
        E = E_mJ / 1000.0  # J
        A = np.pi * (D / 10.0 / 2.0) ** 2  # cm²
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

    # Summary table
    st.markdown("### 📋 Laser Parameter Summary Table")
    summary_df = pd.DataFrame({
        "Parameter": list(res1.keys()),
        "Laser 1": list(res1.values()),
        "Laser 2": [res2[k] for k in res1.keys()]
    })
    st.dataframe(summary_df, use_container_width=True)
    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Laser Comparison Table", csv, "laser_comparison_table.csv", "text/csv", key="cmp_download")

    # Comparison plot
    st.markdown("### 📈 Parameter Comparison")
    params = ["Fluence (J/cm²)", "Peak Irradiance (W/cm²)", "Avg Irradiance (W/cm²)", "Total Energy (J)", "Exposure Time (s)"]
    selected_params = st.multiselect("Select parameters to plot:", params, default=params, key="cmp_plot")
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

# ----------------------------
# Main
# ----------------------------
st.set_page_config(page_title="Laser Tools", layout="wide")
st.title("💡 Laser Tools")

_ = shared_sidebar()

tab1, tab2 = st.tabs(["Laser Calculator", "Laser Comparison"])
with tab1:
    app_laser_calculator()
with tab2:
    app_laser_comparison()
