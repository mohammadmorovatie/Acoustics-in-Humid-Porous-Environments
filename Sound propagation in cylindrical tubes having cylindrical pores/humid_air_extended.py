from __future__ import annotations

from pyfluids import HumidAir


class HumidAirExtended(HumidAir):
    """An example of how to add new properties to the HumidAir class."""

    def __init__(self):
        super().__init__()
        self.__specific_heat_const_volume: float | None = None
        self.__water_mole_fraction: float | None = None
        self.__gamma: float | None = None

    @property
    def specific_heat_const_volume(self) -> float:
        """Mass specific constant volume specific heat [J/kg/K]."""
        if self.__specific_heat_const_volume is None:
            self.__specific_heat_const_volume = self._keyed_output("CVha")
        return self.__specific_heat_const_volume
    
    @property
    def gamma(self) -> float:
        """Heat capacity ratio."""
        if self.__gamma is None:
            self.__gamma = self.specific_heat/self.specific_heat_const_volume
        return self.__gamma
    
    @property
    def water_mole_fraction(self) -> float:
        """Water mole fraction."""
        if self.__water_mole_fraction is None:
            self.__water_mole_fraction = self._keyed_output("psi_w")
        return self.__water_mole_fraction
    
    @property
    def air_molar_mass(self) -> float:
        """Molar mass of dry air [kg/mol]."""
        return 28.959e-3

    @property
    def water_molar_mass(self) -> float:
        """Molar mass of water vapor [kg/mol]."""
        return 18.015268e-3
    
    @property
    def molar_mass(self) -> float:
        """Molar mass of humid air [kg/mol]."""
        return self.air_molar_mass*(1-self.water_mole_fraction) + self.water_molar_mass*self.water_mole_fraction

    @property
    def barodiffusion(self) -> float:
        """Barodiffusion coefficient or dry air in water vapor."""
        return (self.water_molar_mass - self.air_molar_mass)/self.molar_mass

    @property
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity of humid air [Pa·s]."""
        return self._keyed_output("mu")

    @property
    def thermal_conductivity(self) -> float:
         """Thermal conductivity of humid air [W/m/K]."""
         return self._keyed_output("k")

    @property
    def prandtl(self) -> float:
        """Prandtl number (dimensionless)."""
        return self.dynamic_viscosity * self.specific_heat / self.thermal_conductivity

      

    def factory(self) -> HumidAirExtended:
        return HumidAirExtended()

    def reset(self):
        super().reset()
        self.__specific_heat_const_volume = None
        self.__water_mole_fraction = None
        self.__gamma = None



# class TestHumidAirExtended:
#     humid_air = HumidAirExtended().with_state(
#         InputHumidAir.pressure(101325),
#         InputHumidAir.temperature(20),
#         InputHumidAir.relative_humidity(50),
#     )

#     def test_specific_heat_const_volume_humid_air_in_standard_conditions_returns_722(
#         self,
#     ):
#         assert self.humid_air.specific_heat_const_volume == 722.68718970632506
