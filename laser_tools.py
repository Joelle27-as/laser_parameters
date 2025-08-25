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
# Physical thermal model
# ----------------------------
def simulate_physical(
    N, f, tau, E, A_m2,
    rho=1000.0, c=4200.0, k=0.5, eta=0.7, T_amb=37.0, tau_cool=0.5,
    lock_axis=False
):
    """
    Semi-infinite solid (surface) temperature with pulsed surface heat flux.
    q(t) = eta * P(t) / A  [W/m²]
    ΔT(0,t) = (1/(k * sqrt(pi*alpha))) * ∫ q(t') / sqrt(t - t') dt'   (convolution with 1/sqrt(t))
    Then multiplied by an exponential cooling envelope ~ exp(-t/tau_cool) for Newtonian-like cooling.
    Returns t [s], P(t) [W], T(t) [°C] (absolute).
    """
    alpha = k / (rho * c)  # m²/s
    interval = 1.0 / f
    pulse_times = np.arange(N) * interval
    # time resolution: resolve pulse and interpulse regions
    t_res = min(tau / 10.0, interval / 20.0)
    # total time: end of last pulse + some tail (or 1 s if locked)
    t_max = pulse_times[-1] + 10.0 * tau
    if lock_axis:
        t_max = max(t_max, 1.0)
    t = np.arange(0.0, t_max, t_res)

    # Rectangular power profile (W)
    P = np.zeros_like(t)
    P_pulse = E / tau  # W
    for pt in pulse_times:
        idx = (t >= pt) & (t < pt + tau)
        P[idx] = P_pulse

    # Surface heat flux q(t) [W/m²]
    q = eta * P / A_m2

    # Convolution with kernel 1/sqrt(Δt)
    # kernel[0] would be 1/sqrt(0) -> start from Δt = t_res
    kernel = 1.0 / np.sqrt(np.arange(1, len(t) + 1, dtype=float) * t_res)
    conv = np.convolve(q, kernel, mode='full')[:len(t)] * t_res  # numeric integral
    dT = conv / (k * np.sqrt(np.pi * alpha))  # °C (K)

    # Newtonian-like cooling envelope (simple, tunable)
    if tau_cool > 0:
        env = np.exp(-t / tau_cool)
        dT *= env

    T_abs = T_amb + dT
    return t, P, T_abs, dT

# ----------------------------
# App 1: Laser Calculator
# ----------------------------
def app_laser_calculator():
    st.header("🔬 Laser Calculator with Export and Visual Analysis")

    # --- Tab-specific sidebar controls (under shared section) ---
    st.sidebar.subheader("Laser Calculator • Parameters")
    D = st.sidebar.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01, key="calc_D")
    E_mJ = st.sidebar.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0, key="calc_E")
    f = st.sidebar.number_input("Pulse Frequency (Hz)", value=10, min_value=1, key="calc_f")
    N = st.sidebar.number_input("Number of Pulses", value=20, min_value=1, key="calc_N")
    unit = st.sidebar.selectbox("Pulse Duration Unit", ["µs", "ns"], key="calc_unit")
    tau_input = st.sidebar.number_input(f"Pulse Duration ({unit})", value=200.0, min_value=0.01, key="calc_tau")
    tau = tau_input * (1e-6 if unit == "µs" else 1e-9)
    wavelength = st.sidebar.number_input("Laser Wavelength (nm)", value=2940, min_value=100, key="calc_wavelength")
    lock_axis = st.sidebar.checkbox("Lock X-axis scale to 1.0 s", value=False, key="calc_lock_axis")

    st.sidebar.subheader("Material & Absorption")
    rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0, key="mat_rho")
    c = st.sidebar.number_input("Specific heat c (J/kg·K)", value=4200.0, min_value=1.0, key="mat_c")
    k = st.sidebar.number_input("Conductivity k (W/m·K)", value=0.5, min_value=0.001, key="mat_k")
    eta = st.sidebar.slider("Absorbed fraction η", 0.0, 1.0, 0.7, key="mat_eta")
    T_amb = st.sidebar.number_input("Ambient / Baseline (°C)", value=37.0, key="mat_Tamb")
    tau_cool = st.sidebar.number_input("Cooling time constant τ_c (s)", value=0.5, min_value=0.0, key="mat_tauc")

    # --- Calculations (geometric & energy) ---
    E = E_mJ / 1000.0               # J
    A_cm2 = np.pi * (D / 20.0) ** 2  # cm²
    A_m2 = A_cm2 * 1e-4              # m²
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

    # --- Fluence vs simple tissue threshold (optional visual) ---
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

    # --- Timeline & Thermal (PHYSICAL MODEL) ---
    st.markdown("### 📈 Pulse Timeline & Thermal (physical model)")

    t, P_t, T_abs, dT = simulate_physical(
        N=N, f=f, tau=tau, E=E, A_m2=A_m2,
        rho=rho, c=c, k=k, eta=eta, T_amb=T_amb, tau_cool=tau_cool,
        lock_axis=lock_axis
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Max ΔT (°C)", f"{np.max(dT):.2f}")
    c2.metric("Max T (°C)", f"{np.max(T_abs):.2f}")
    c3.metric("Laser-On Time (s)", f"{T_on:.4f}")

    # Power timeline
    fig1, ax1 = plt.subplots()
    ax1.plot(t, P_t, label="Laser Power (W)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Power (W)")
    ax1.set_title("Laser Pulse Timeline")
    ax1.legend()
    if lock_axis:
        ax1.set_xlim(0, 1.0)
    st.pyplot(fig1)

    # Temperature over time (absolute)
    fig2, ax2 = plt.subplots()
    ax2.plot(t, T_abs, label="Surface Temperature (°C)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("°C")
    ax2.set_title("Thermal Buildup Over Time (surface)")
    ax2.legend()
    st.pyplot(fig2)

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

    # Matching helpers
    st.markdown("### 🔺 Match Peak Irradiance (Iₚₑₐₖ)")
    option_peak = st.selectbox("Adjust in Laser 1:", ["Energy", "Pulse Duration", "Spot Diameter"], key="cmp_match_peak")
    I_target = res2["Peak Irradiance (W/cm²)"]
    A1, tau1, E1 = res1["Spot Area (cm²)"], res1["Pulse Duration (s)"], res1["Energy (J)"]
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

    st.markdown("### 🔶 Match Fluence (J/cm²)")
    option_fluence = st.selectbox("Adjust in Laser 1 (Fluence):", ["Energy", "Spot Diameter"], key="cmp_match_fluence")
    fluence_target = res2["Fluence (J/cm²)"]
    if option_fluence == "Energy":
        E_new = fluence_target * A1
        st.success(f"To match Fluence, set Laser 1 Energy to **{E_new * 1000:.2f} mJ**")
    elif option_fluence == "Spot Diameter":
        A_new = E1 / fluence_target
        D_new = 2 * np.sqrt(A_new / np.pi) * 10
        st.success(f"To match Fluence, set Laser 1 Spot Diameter to **{D_new:.2f} mm**")

    st.markdown("### 🔷 Match Average Irradiance (W/cm²)")
    option_avg = st.selectbox("Adjust in Laser 1 (Avg Irradiance):", ["Energy", "Spot Diameter"], key="cmp_match_avg")
    I_avg_target = res2["Avg Irradiance (W/cm²)"]
    if option_avg == "Energy":
        E_new = (I_avg_target * A1) / f1
        st.success(f"To match Avg Irradiance, set Laser 1 Energy to **{E_new * 1000:.2f} mJ**")
    elif option_avg == "Spot Diameter":
        A_new = (E1 * f1) / I_avg_target
        D_new = 2 * np.sqrt(A_new / np.pi) * 10
        st.success(f"To match Avg Irradiance, set Laser 1 Spot Diameter to **{D_new:.2f} mm**")

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
