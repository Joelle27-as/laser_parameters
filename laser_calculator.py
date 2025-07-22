import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Laser Parameter Calculator")

D = st.number_input("Spot Diameter (mm)", value=0.5, min_value=0.01)
E_mJ = st.number_input("Energy per Pulse (mJ)", value=3.0, min_value=0.0)
f = st.number_input("Frequency (Hz)", value=10, min_value=0)
N = st.number_input("Number of Shots", value=50, min_value=1)
tau_us = st.number_input("Pulse Duration (µs)", value=200.0, min_value=0.1)
threshold = st.number_input("Fluence Threshold (J/cm²)", value=2.5, min_value=0.0)

E = E_mJ / 1000
A = np.pi * (D / 20)**2
F = E / A
tau = tau_us * 1e-6
I = E / (A * tau)
E_total = E * N
P_peak = E / tau

st.subheader("Calculated Parameters")
st.write(f"**Spot Area:** {A:.4f} cm²")
st.write(f"**Fluence:** {F:.2f} J/cm²")
st.write(f"**Irradiance:** {I:.2e} W/cm²")
st.write(f"**Peak Power:** {P_peak:.2f} W")
st.write(f"**Total Deposited Energy:** {E_total:.2f} J")

st.subheader("Fluence vs. Threshold")
fig, ax = plt.subplots()
ax.bar(["Your Fluence"], [F], color='green' if F > threshold else 'red')
ax.axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold} J/cm²)')
ax.legend()
ax.set_ylabel("Fluence (J/cm²)")
st.pyplot(fig)
