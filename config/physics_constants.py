from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PhysicalConstants:
    """
    Dataclass holding fundamental physical constants in SI units for electrodiffusion simulations.
    These values are used in calculations involving thermodynamics (e.g., kT for energy scales),
    electrostatics (e.g., Coulomb potentials), and diffusion (e.g., Debye screening).
    Constants are immutable by default, but instances can be created with overrides if needed.
    Derived properties provide commonly used combinations like β (1/kT) for Boltzmann factors.
    """

    # Boltzmann constant: Relates temperature to energy (J/K); key for kT in PMF and diffusion coeffs.
    KB: float = 1.380649e-23  # Exact value from SI definition

    # Elementary charge: Charge of a proton (C); used in ion charges (e.g., q = 1*e for monovalent ions).
    E: float = 1.602176e-19  # Exact value from SI

    # Vacuum permittivity: Dielectric constant of free space (F/m); base for Coulomb's law.
    EPS0: float = 8.854188e-12  # Approximate value

    # Relative permittivity of water: Dimensionless factor for solvent dielectric (typically 78-80 at room temp).
    EPS_R: float = 80.0  # Standard value for aqueous solutions in biophysical modeling

    # Temperature: System temperature (K); affects thermal fluctuations and energy scales.
    T: float = 300.0  # Room temperature approximation

    # Avogadro's number: Particles per mole; used in molar concentrations and gas constant.
    NA: float = 6.022140e23  # Approximate value

    @property
    def BETA(self) -> float:
        """
        Inverse temperature β = 1 / (k_B * T), in J^{-1}.
        This is frequently used in Boltzmann distributions (e.g., exp(-β ΔE)) for probabilities
        and in drift terms of Langevin equations to scale forces.
        """
        return 1.0 / (self.KB * self.T)

    @property
    def RT(self) -> float:
        """
        Gas constant times temperature (R * T), in J/mol.
        Equivalent to N_A * k_B * T; useful for free energy calculations in chemical potentials
        or Nernst equations for ion gradients.
        """
        return self.KB * self.NA * self.T

    def debye_length(self, ionic_strength: float, temperature: Optional[float] = None) -> float:
        """
        Computes the Debye screening length κ^{-1} in meters.
        This length scale determines the range of electrostatic interactions in electrolytes,
        crucial for screened Coulomb potentials in ion channel models.

        Args:
            ionic_strength (float): Effective ionic strength I = ½ ∑ c_i z_i² (mol/m³ or M).
            temperature (Optional[float]): Override temperature (default: self.T).

        Returns:
            float: Debye length.

        Note: Formula: κ^{-1} = sqrt(ε_0 ε_r kT / (2 N_A e² I)); clamps to prevent NaN if I=0.
        """
        T = temperature or self.T
        kappa_sq = 2 * self.E**2 * self.NA * ionic_strength / (self.EPS0 * self.EPS_R * self.KB * T)
        kappa_sq = np.maximum(kappa_sq, 1e-10)  # Clamp to avoid division by zero or negative sqrt
        return 1.0 / np.sqrt(kappa_sq)

    def einstein_mobility(self, diffusion_coeff: float, temperature: Optional[float] = None) -> float:
        """
        Computes ionic mobility μ = D / (kT) from Einstein relation, in m²/(V s).
        Mobility relates drift velocity to electric field, used in Nernst-Planck fluxes.

        Args:
            diffusion_coeff (float): Diffusion coefficient D (m²/s).
            temperature (Optional[float]): Override T (default: self.T).

        Returns:
            float: Mobility μ.

        Note: Assumes monovalent ion (scale by z*e if needed); not in original but added for completeness.
        """
        T = temperature or self.T
        return diffusion_coeff / (self.KB * T)
