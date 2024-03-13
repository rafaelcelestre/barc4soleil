#!/bin/python

"""
This module provides a collection of functions for performing calculations related to 
undulators used in synchrotron radiation sources.
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '12/MAR/2024'
__changed__ = '12/MAR/2024'

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

#***********************************************************************************
# auxiliary functions
#***********************************************************************************

def energy_wavelength(value: float, unity: str) -> float:
    """
    Converts energy to wavelength and vice versa.
    
    Parameters:
        value (float): The value of either energy or wavelength.
        unity (str): The unit of 'value'. Can be 'eV', 'meV', 'keV', 'm', 'nm', or 'A'. Case sensitive. 
        
    Returns:
        float: Converted value in meters if the input is energy, or in eV if the input is wavelength.
        
    Raises:
        ValueError: If an invalid unit is provided.
    """
    factor = 1.0
    
    # Determine the scaling factor based on the input unit
    if unity.endswith('eV') or unity.endswith('meV') or unity.endswith('keV'):
        prefix = unity[:-2]
        if prefix == "m":
            factor = 1e-3
        elif prefix == "k":
            factor = 1e3
    elif unity.endswith('m'):
        prefix = unity[:-1]
        if prefix == "n":
            factor = 1e-9
    elif unity.endswith('A'):
        factor = 1e-10
    else:
        raise ValueError("Invalid unit provided: {}".format(unity))

    return physical_constants["Planck constant"][0] * \
           physical_constants["speed of light in vacuum"][0] / \
           physical_constants["atomic unit of charge"][0] / \
           (value * factor)


def get_gamma(E: float) -> float:
    """
    Calculate the Lorentz factor (γ) based on the energy of electrons in GeV.

    Parameters:
        E (float): Energy of electrons in GeV.

    Returns:
        float: Lorentz factor (γ).
    """
    return E * 1e9 / (physical_constants["electron mass"][0] * physical_constants["speed of light in vacuum"][0] ** 2) * physical_constants["atomic unit of charge"][0]

#***********************************************************************************
# undulator parameters
#***********************************************************************************

def get_K_from_B(B: float, period: float) -> float:
    """
    Calculate the undulator parameter K from the magnetic field B and the undulator period lambda_u.

    Parameters:
    B (float): Magnetic field in Tesla.
    period (float): Undulator period in meters.

    Returns:
    float: The undulator parameter K.
    """
    K = physical_constants["atomic unit of charge"][0] * B * period / (2 * np.pi * physical_constants["electron mass"][0] * physical_constants["speed of light in vacuum"][0])
    return K


def get_B_from_K(K: float, period: float) -> float:
    """
    Calculate the undulator magnetic field in Tesla from the undulator parameter K and the undulator period lambda_u.

    Parameters:
    K (float): The undulator parameter K.
    period (float): Undulator period in meters.

    Returns:
    float: Magnetic field in Tesla.
    """
    B = K * 2 * np.pi * physical_constants["electron mass"][0] * physical_constants["speed of light in vacuum"][0]/(physical_constants["atomic unit of charge"][0] * period)
    return B

#***********************************************************************************
# field-gap relationship
#***********************************************************************************

def fit_gap_field_relation(gap_table: List[float], B_table: List[float], 
                           u_period: float) -> Tuple[float, float, float]:
    """
    Fit parameters coeff0, coeff1, and coeff2 for an undulator from the given tables:

    B0 = c0 * exp[c1(gap/u_period) + c2(gap/u_period)]

    Parameters:
        gap_table (List[float]): List of gap sizes in meters.
        B_table (List[float]): List of magnetic field values in Tesla corresponding to the gap sizes.
        u_period (float): Undulator period in meters.

    Returns:
        Tuple[float, float, float]: Fitted parameters (coeff0, coeff1, coeff2).
    """
    def _model(gp, c0, c1, c2):
        return c0 * np.exp(c1*gp + c2*gp**2)

    def _fit_parameters(gap, u_period, B):
        gp = gap / u_period
        popt, pcov = curve_fit(_model, gp, B, p0=(1, 1, 1)) 
        return popt

    popt = _fit_parameters(np.asarray(gap_table), u_period, np.asarray(B_table))
    coeff0_fit, coeff1_fit, coeff2_fit = popt

    print("Fitted parameters:")
    print("coeff0:", coeff0_fit)
    print("coeff1:", coeff1_fit)
    print("coeff2:", coeff2_fit)

    return coeff0_fit, coeff1_fit, coeff2_fit


def get_B_from_gap(gap: float, u_period: float, coeff0: float, coeff1: float, 
                   coeff2: float) -> Union[float, None]:
    """
    Calculate the magnetic field B from the given parameters:
       B0 = c0 * exp[c1(gap/u_period) + c2(gap/u_period)]

    Parameters:
        gap (float): Gap size in meters.
        u_period (float): Undulator period in meters.
        coeff0 (float): Coefficient 0.
        coeff1 (float): Coefficient 1.
        coeff2 (float): Coefficient 2.

    Returns:
        float: Calculated magnetic field B.
        None: If gap or period is non-positive, returns None.
    """

    gp = np.asarray(gap) / u_period
    B = coeff0 * np.exp(coeff1 * gp + coeff2 * gp**2)
    
    return B

#***********************************************************************************
# undulator emission
#***********************************************************************************

def get_emission_energy(u_period: float, K: float, ring_e: float, n: int = 1, theta: float = 0) -> float:
    """
    Calculate the energy of an undulator emission in a storage ring.

    Parameters:
        u_period (float): Undulator period in meters.
        K (float): Undulator parameter.
        ring_e (float): Energy of electrons in GeV.
        n (int, optional): Harmonic number (default is 1).
        theta (float, optional): Observation angle in radians (default is 0).

    Returns:
        float: Emission energy in electron volts.
    """
    gamma = get_gamma(ring_e)
    emission_wavelength = u_period * (1 + (K ** 2) / 2 + (gamma * theta) ** 2) / (2 * n * gamma ** 2)

    return energy_wavelength(emission_wavelength, "m")


def find_emission_harmonic():
    pass


#***********************************************************************************
# power calculation
#***********************************************************************************

def total_power(ring_e: float, ring_curr: float, und_per: float, und_n_per: int,
              B: Optional[float] = None, K: Optional[float] = None,
              verbose: bool = False) -> float:
    """ 
    Calculate the total power emitted by a planar undulator in kilowatts (kW) based on Eq. 56 
    from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
    [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

    :param ring_e: Ring energy in gigaelectronvolts (GeV).
    :param ring_curr: Ring current in amperes (A).
    :param und_per: Undulator period in meters (m).
    :param und_n_per: Number of periods.
    :param B: Magnetic field in tesla (T). If not provided, it will be calculated based on K.
    :param K: Deflection parameter. Required if B is not provided.
    :param verbose: Whether to print intermediate calculation results. Defaults to False.
    
    :return: Total power emitted by the undulator in kilowatts (kW).
    """

    if B is None:
        if K is None:
            raise TypeError("Please, provide either B or K for the undulator")
        else:
            B = K/(0.934*und_per/1E-2)
            if verbose:
                print(">>> B = %.5f [T]"%B)

    return 0.63*(ring_e**2)*(B**2)*ring_curr*und_per*und_n_per

def partial_power():
   pass