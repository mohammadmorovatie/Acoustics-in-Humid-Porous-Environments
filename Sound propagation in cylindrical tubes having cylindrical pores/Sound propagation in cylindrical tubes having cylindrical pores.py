import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from pyfluids import InputHumidAir
from humid_air_extended import HumidAirExtended

# -------------------------
# 1. Ambient Conditions and Geometry
# -------------------------

T_c = 20                   # Temperature (°C)
P_0 = 1.0132e5             # Pressure (Pa)
RH = 50                    # Relative Humidity (%)
R = 0.25e-3                 # Radius of the tube (m)
d = 0.065                  # Thickness of the sample (m)

# Frequency range
f = np.linspace(80,3400,500)
omega = 2 * np.pi * f     # Angular frequency (rad/s)

# -------------------------
# 2. Humid Air Properties
# -------------------------

humid_air = HumidAirExtended().with_state(
    InputHumidAir.pressure(P_0),
    InputHumidAir.temperature(T_c),
    InputHumidAir.relative_humidity(RH)
)

rho_0 = humid_air.density
eta = humid_air.dynamic_viscosity
gamma = humid_air.gamma
Pr = humid_air.prandtl
P0 = humid_air.pressure
c_0 = np.sqrt(gamma * P0 / rho_0)
Z_0 = rho_0 * c_0

# -------------------------
# 3. Effective Density and Bulk Modulus
# -------------------------

s = np.sqrt((omega * rho_0 * R**2) / eta)
rho_eff = rho_0 / (1 - (2 / (s * np.sqrt(-1j))) * (jv(1, s * np.sqrt(-1j)) / jv(0, s * np.sqrt(-1j))))
K_eff = (gamma * P0) / (1 + (gamma - 1) * (2 / (np.sqrt(Pr) * s * np.sqrt(-1j))) * (jv(1, np.sqrt(Pr) * s * np.sqrt(-1j)) / jv(0, np.sqrt(Pr) * s * np.sqrt(-1j))))

# Normalized values (for plotting)
rho_norm = rho_eff / rho_0
K_norm = K_eff / P0

# -------------------------
# 4. Complex Wave Parameters
# -------------------------

Z_c = np.sqrt(rho_eff * K_eff)
k_c = omega * np.sqrt(rho_eff / K_eff)

# -------------------------
# 5. Transfer Matrix Method
# -------------------------

T11 = np.cos(k_c * d)
T12 = 1j * Z_c * np.sin(k_c * d)
T21 = (1j * np.sin(k_c * d)) / Z_c
T22 = np.cos(k_c * d)

R = (T11 - Z_0 * T21) / (T11 + Z_0 * T21)
alpha = 1 - np.abs(R)**2

# -------------------------
# 6. Plotting (High-Resolution, No Titles, Frequency Only)
# -------------------------

def plot_complex_quantity(x, y, xlabel, ylabel_real, ylabel_imag):
    plt.figure(figsize=(7, 8), dpi=200)
    
    plt.subplot(211)
    plt.plot(x, np.real(y), color='blue', linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel_real)
    plt.grid(True)

    plt.subplot(212)
    plt.plot(x, np.imag(y), color='red', linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel_imag)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plot normalized effective density
plot_complex_quantity(f, rho_norm, 'Frequency (Hz)', 'Re(ρ̃ / ρ₀)', 'Im(ρ̃ / ρ₀)')

# Plot normalized bulk modulus
plot_complex_quantity(f, K_norm, 'Frequency (Hz)', 'Re(K̃ / P₀)', 'Im(K̃ / P₀)')

# Plot reflection coefficient
plot_complex_quantity(f, R, 'Frequency (Hz)', 'Re(R)', 'Im(R)')

# Plot absorption coefficient
plt.figure(figsize=(7, 4), dpi=200)
plt.plot(f, alpha, color='blue', linewidth=1.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Absorption Coefficient α')
plt.grid(True)
plt.tight_layout()
plt.show()
