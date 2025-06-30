import numpy as np
import matplotlib.pyplot as plt
from pyfluids import InputHumidAir
from humid_air_extended import HumidAirExtended  # Custom extended class

# -----------------------------------------------------------------------------
# Initialize humid air at standard atmospheric conditions
# -----------------------------------------------------------------------------
humid_air = HumidAirExtended().with_state(
    InputHumidAir.pressure(101325),         # Pressure in Pa
    InputHumidAir.temperature(20),          # Temperature in °C
    InputHumidAir.relative_humidity(50),    # Relative Humidity in %
)

# -----------------------------------------------------------------------------
# Extract physical properties from the humid air model
# -----------------------------------------------------------------------------
rho_0 = humid_air.density                              # Density [kg/m^3]
cp = humid_air.specific_heat                           # Specific heat at constant pressure [J/kg·K]
cv = humid_air.specific_heat_const_volume              # Specific heat at constant volume [J/kg·K]
gamma = humid_air.gamma                                # Cp / Cv
eta = humid_air.dynamic_viscosity                      # Dynamic viscosity [Pa·s]
k = humid_air.thermal_conductivity                     # Thermal conductivity [W/m·K]
N_Pr = humid_air.prandtl                               # Prandtl number

# -----------------------------------------------------------------------------
# Define frequency range (Hz) and compute angular frequency (rad/s)
# -----------------------------------------------------------------------------
frequency_hz = np.linspace(80, 3400, 500)               # Frequency in Hz
omega = 2 * np.pi * frequency_hz                       # Convert to angular frequency [rad/s]

# -----------------------------------------------------------------------------
# Compute skin depths (in meters), then convert to millimeters
# -----------------------------------------------------------------------------
delta_m = np.sqrt((2 * eta) / (omega * rho_0))                   # Viscous skin depth
delta_prime_m = np.sqrt((2 * eta) / (omega * N_Pr * rho_0))      # Thermal skin depth

delta_mm = delta_m * 1000
delta_prime_mm = delta_prime_m * 1000

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(frequency_hz, delta_mm, label=r'Viscous Skin Depth $\delta$', lw=2)
plt.plot(frequency_hz, delta_prime_mm, label=r"Thermal Skin Depth $\delta'$", lw=2, linestyle='--')

plt.xlabel(r'Frequency $f$ (Hz)')
plt.ylabel(r'Skin Depth (mm)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save high-resolution image
plt.savefig("skin_depth_vs_frequency.png", dpi=300, bbox_inches='tight')

# Show plot
plt.show()
