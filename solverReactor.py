import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from dataclasses import dataclass, field
import scipy.linalg
from typing import ClassVar, Dict, List
from numba import njit, jit
from numba.experimental import jitclass


UNIVERSALGASCONSTANT = 8.31446261815324 #J/molK
T_REF   = 273.15
P_REF   = 101325.0 #Pa
PI      = 3.141592653589793

@dataclass
class Specie:
    counter: ClassVar[int] = 0

    name      : str
    molarMass : float
    heatCapacityModel   : str   = "const" 
    enthalpyFormation   : float = 0.0
    entropyFormation    : float = 0.0
    heatCapacityValue   : float = 900
    heatCapacityCoefficients : list[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = Specie.counter
        type(self).counter += 1
        
        if self.heatCapacityModel == "polynomial":
            self.cp_poly = np.polynomial.Polynomial(self.heatCapacityCoefficients)
            self.H_poly = self.cp_poly.integ()  
    def heatCapacity(self, T: np.ndarray):
        T = np.asarray(T)
        if self.heatCapacityModel == "const":
            return np.full_like(T, self.heatCapacityValue)
        else:
            return self.cp_poly(T) * UNIVERSALGASCONSTANT

    def enthalpy(self, T):
        T = np.asarray(T)
        if self.heatCapacityModel == "const":
            sensible_enthalpy = self.heatCapacityValue * (T - T_REF)
        else:
            sensible_enthalpy = (self.H_poly(T) - self.H_poly(T_REF)) * UNIVERSALGASCONSTANT
            
        return self.enthalpyFormation + sensible_enthalpy
    
@dataclass
class Reaction:
    name: str
    stochiometricCoefficients   : np.ndarray    = field(default_factory=lambda: np.array([]))
    speciesExponent             : np.ndarray    = field(default_factory=lambda: np.array([])) 
    reversedSpecieExponent      : np.ndarray    = field(default_factory=lambda: np.array([])) 
    isReversible                : bool          = True
    ahrreniusPreExponent        : float         = 1.0
    ahrreniusActivationEnergy   : float         = 0.0  # J/mol
    species                     : List          = field(default_factory=list, repr=False)
    entropyChange               : float         = field(init=False)   # J/mol/K
    enthalpyChange              : float         = field(init=False)   # J/mol
    molarMasses                 : np.ndarray    = field(init=False)

    def __post_init__(self):
        if not (len(self.stochiometricCoefficients) == len(self.speciesExponent) == len(self.reversedSpecieExponent)):
            raise ValueError("Stoichiometric coefficients and exponents require the same length")
        
        self.enthalpyChange = 0.0
        self.entropyChange  = 0.0
        molar_masses_list = []

        for nu, specie in zip(self.stochiometricCoefficients, self.species):
            self.enthalpyChange += nu * specie.enthalpyFormation     # J/mol
            self.entropyChange  += nu * specie.entropyFormation      # J/mol/K
            molar_masses_list.append(specie.molarMass)
            
        self.molarMasses = np.array(molar_masses_list)

    def forwardRateConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        arg = -self.ahrreniusActivationEnergy / (UNIVERSALGASCONSTANT * T)
        return self.ahrreniusPreExponent * np.exp(arg)

    def equilibriumConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        delta_G = self.enthalpyChange - T * self.entropyChange

        arg = -delta_G / (UNIVERSALGASCONSTANT * T)
        K_p = np.exp(arg)
        return np.clip(K_p, 1e-100, 1e100)

    def backwardRateConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if not self.isReversible:
            return np.zeros_like(T)
        
        k_f = self.forwardRateConstant(T)
        K_p = self.equilibriumConstant(T)
        
        delta_nu = np.sum(self.stochiometricCoefficients)
        K_c = K_p * ((P_REF / (UNIVERSALGASCONSTANT * T)) ** delta_nu)

        K_c = np.clip(K_c, 1e-50, np.inf)
        k_b = k_f / K_c
        return k_b

    def enthalpyReactionChange(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        delta_H_rxn = np.zeros_like(T)

        for nu_i, sp in zip(self.stochiometricCoefficients, self.species):
            delta_H_rxn += nu_i * sp.enthalpy(T)

        return delta_H_rxn
    
    def reactionHeatSourceDerivative(self, T: np.ndarray, rates: tuple) -> np.ndarray:
        T = np.maximum(np.asarray(T, dtype=float), 1e-3)
        rateForward, rateBackward = rates
        
        delta_cp = np.zeros_like(T)
        for nu_i, sp in zip(self.stochiometricCoefficients, self.species):
            delta_cp += nu_i * sp.heatCapacity(T)
            
        term_den = UNIVERSALGASCONSTANT * (T ** 2)
        dRf_dT = rateForward * (self.ahrreniusActivationEnergy / term_den) 
        
        delta_H_rxn = self.enthalpyReactionChange(T)
        
        if self.isReversible:
            dRb_dT = rateBackward * ((self.ahrreniusActivationEnergy - delta_H_rxn) / term_den)
        else:
            dRb_dT = np.zeros_like(dRf_dT)
        dQ_dT = - (dRf_dT - dRb_dT) * delta_H_rxn - (rateForward - rateBackward) * delta_cp
        
        return dQ_dT
    
    def reactionRate(self, T: np.ndarray, concentrations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T, dtype=float)
        C = np.asarray(concentrations, dtype=float)

        C_safe = np.maximum(C, 1e-20) 
        alpha  = np.asarray(self.speciesExponent, dtype=float)[:, np.newaxis]
        massActionForward = np.prod(C_safe ** alpha, axis=0)
        
        rateForward = self.forwardRateConstant(T) * massActionForward

        if not self.isReversible:
            return rateForward, np.zeros_like(rateForward)

        beta = np.asarray(self.reversedSpecieExponent, dtype=float)[:, np.newaxis]
        massActionBackward = np.prod(C_safe ** beta, axis=0)

        rateBackward = self.backwardRateConstant(T) * massActionBackward

        return rateForward, rateBackward

    def reactionMassSource(self, rates: tuple) -> np.ndarray:
        rateForward, rateBackward = rates
        net_rate = rateForward - rateBackward
        massSources = (net_rate[np.newaxis, :] * self.stochiometricCoefficients[:, np.newaxis]) * self.molarMasses[:, np.newaxis]
        return massSources

    def reactionHeatSource(self, T: np.ndarray, rates: tuple):
        rateForward, rateBackward = rates
        delta_H_rxn = self.enthalpyReactionChange(T)
        return (rateForward - rateBackward) * delta_H_rxn
    
    def rateDerivativeConcentration(self, T: np.ndarray, C: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T, dtype=float)
        C = np.asarray(C, dtype=float)

        alpha = np.asarray(self.speciesExponent, dtype=float)[:, np.newaxis]  
        C_safe = np.maximum(C, 1e-20)
        massActionForward = np.prod(C_safe ** alpha, axis=0) 
        k_f = self.forwardRateConstant(T)
        
        alpha_j = alpha[j, 0]
        Cj_safe = np.maximum(C[j, :], 1e-20)
        
        dM_f_dCj = alpha_j * massActionForward / Cj_safe
        dRf_dCj  = k_f * dM_f_dCj

        if not self.isReversible:
            return dRf_dCj, np.zeros_like(dRf_dCj)

        beta = np.asarray(self.reversedSpecieExponent, dtype=float)[:, np.newaxis]
        massActionBackward = np.prod(C_safe ** beta, axis=0)
        k_b = self.backwardRateConstant(T)

        beta_j = beta[j, 0]
        dM_b_dCj = beta_j * massActionBackward / Cj_safe
        dRb_dCj  = k_b * dM_b_dCj
        
        return dRf_dCj, dRb_dCj
    
    
@dataclass
class Mixture:
    densityModel    : str           = "ideal-incompressible-gas"   # or "const"
    densityValue    : float         = 1.2225                       # used explicitly if densityModel == "const"
    species         : List          = field(default_factory=list, repr=False)
    molarMasses     : np.ndarray    = field(init=False)

    def __post_init__(self):
        self.molarMasses = np.asarray([sp.molarMass for sp in self.species], dtype=float)

    def equivalentMolarMass(self, speciesFractions: np.ndarray) -> np.ndarray:
        """
        speciesFractions: (n_species, n_cells) mass fractions Y_i
        returns: (n_cells,) mixture molar mass [kg/mol]
        """
        y           = np.asarray(speciesFractions, dtype=float)
        M_i         = self.molarMasses[:, np.newaxis]
        
        inv_M_mix   = np.sum(y / M_i, axis=0)
        inv_M_mix   = np.maximum(inv_M_mix, 1e-16)
        
        M_mix       = 1.0 / inv_M_mix
        return M_mix

    def idealGasDensity(self, T: np.ndarray, speciesFractions: np.ndarray) -> np.ndarray:
        """
        T: (n_cells,) [K]
        speciesFractions: (n_species, n_cells) mass fractions Y_i
        returns: rho: (n_cells,) [kg/m^3]
        """
        T       = np.asarray(T, dtype=float)
        M_mix   = self.equivalentMolarMass(speciesFractions)      # [kg/mol]
        rho     = P_REF * M_mix / (UNIVERSALGASCONSTANT * T)      # Ideal gas law
        return rho
    
    def mixtureHeatCapacity(self, T: np.ndarray, speciesFractions: np.ndarray) -> np.ndarray:
        """
        Calculates the mixture specific heat capacity in J/(kg K)
        """
        cp_mass_array = np.stack([
            sp.heatCapacity(T) / sp.molarMass 
            for sp in self.species
        ]) 
        
        return np.sum(speciesFractions * cp_mass_array, axis=0)
@dataclass
class domainSetup:
    diameter: float
    # Removed massFlowRate since it's calculated from inlet velocity
    inletMassFractions : np.ndarray = field(default_factory=lambda: np.array([])) 

@dataclass 
class Inlet:
    position    : int = 0
    velocity    : float = 50.0
    temperature : float = 700.0
    speciesMassFractions : List[float] = field(default_factory=list)

    def inletValues(self, mixture: Mixture, domain: domainSetup):
        Y_in    = np.asarray(self.speciesMassFractions, dtype=float)      # (n_species,)
        Y_in_2d = Y_in.reshape(-1, 1)                                     # (n_species, 1)
        
        T_array = np.array([float(self.temperature)], dtype=float)        # (1,)

        density = mixture.idealGasDensity(T=T_array,
                                        speciesFractions=Y_in_2d)         # rho: (1,)

        area          = (domain.diameter ** 2) * PI / 4.0
        massFlowrate  = density[0] * area * self.velocity                 # Float
        temperatureBC = float(self.temperature)                           # Float
        specieFrac    = Y_in                                              # 1D Array

        return massFlowrate, temperatureBC, specieFrac
# @dataclass
# class BoundaryConditions:
#     counter: int = field(init=False, default=0, repr=False)

#     def addBoundary(self):
#         self.id = type(self).counter
#         type(self).counter += 1
@dataclass
class Zone:
    counter : ClassVar[int] = 0
    length  : float         = 0.005
    type    : str           = "null"

    def __post_init__(self):
        type(self).counter  += 1
        self.id             = type(self).counter
        
        self.heatSource      : bool  = False
        self.massSource      : bool  = False
        self.heatSourceValue : float = 0.0

    def zoneAssign(self, heating: bool = False, reaction: bool = True):
        self.heatSource = heating
        self.massSource = reaction

    def zoneAssignHeating(self, heatValue: float):
        """
        Assign the total heat power [W] for this zone (Case B)
        or the volumetric heat generation [W/m^3] (Case A).
        """
        self.heatSourceValue = heatValue
        self.heatSource = True


class Mesh:
    def __init__(self, domain: domainSetup, zoneList: List[Zone], sizing: float = 0.005):
        self.sizing     = sizing
        self.domain     = domain
        self.meshZones  = zoneList

    def meshCreate(self):
        zones = [z for z in self.meshZones if z.length > 0.0]
        if not zones:
            self.cell_centers    = np.array([], dtype=float)
            self.cell_sizes      = np.array([], dtype=float)
            self.cell_volumes    = np.array([], dtype=float)
            self.cell_zone_id    = np.array([], dtype=int)
            self.cell_zone_type  = np.array([], dtype=object)
            self.cell_heat_flag  = np.array([], dtype=bool)
            self.cell_mass_flag  = np.array([], dtype=bool)
            self.cell_heat_value = np.array([], dtype=float)
            self.n_cells         = 0
            self.length          = 0.0
            return
        Lz         = np.array([z.length for z in zones], dtype=float)   # (Nz,)
        zone_ids   = np.array([z.id for z in zones], dtype=int)     # (Nz,)
        zone_types = np.array([z.type for z in zones], dtype=object)  # (Nz,)
        heat_flags = np.array([z.heatSource for z in zones], dtype=bool)    # (Nz,)
        mass_flags = np.array([z.massSource for z in zones], dtype=bool)    # (Nz,)
        heat_vals  = np.array([z.heatSourceValue for z in zones], dtype=float)   # (Nz,)

        # number of cells and spacing per zone
        n_cells_zone = np.maximum(1, np.rint(Lz / self.sizing).astype(int))  # (Nz,)
        dz_zone      = Lz / n_cells_zone                                    # (Nz,)

        # total cells
        n_total = int(np.sum(n_cells_zone))

        self.cell_centers    = np.empty(n_total, dtype=float)
        self.cell_sizes      = np.empty(n_total, dtype=float)
        self.cell_volumes    = np.empty(n_total, dtype=float)
        self.cell_zone_id    = np.empty(n_total, dtype=int)
        self.cell_zone_type  = np.empty(n_total, dtype=object)
        self.cell_heat_flag  = np.empty(n_total, dtype=bool)
        self.cell_mass_flag  = np.empty(n_total, dtype=bool)
        self.cell_heat_value = np.empty(n_total, dtype=float)

        z_starts = np.concatenate(([0.0], np.cumsum(Lz[:-1])))      # (Nz,)
        cell_start_idx = np.concatenate(([0], np.cumsum(n_cells_zone[:-1])))  # (Nz,)

        area = (self.domain.diameter**2) * PI / 4.0

        for i, z in enumerate(zones):
            nc    = n_cells_zone[i]
            dz_i  = dz_zone[i]
            z0    = z_starts[i]
            start = cell_start_idx[i]
            end   = start + nc
            idx   = slice(start, end)

            k = np.arange(nc, dtype=float)
            self.cell_centers[idx] = z0 + (k + 0.5) * dz_i
            self.cell_sizes[idx]   = dz_i
            self.cell_volumes[idx] = dz_i * area

            self.cell_zone_id[idx]    = zone_ids[i]
            self.cell_zone_type[idx]  = zone_types[i]
            self.cell_heat_flag[idx]  = heat_flags[i]
            self.cell_mass_flag[idx]  = mass_flags[i]
            self.cell_heat_value[idx] = heat_vals[i] 

        self.n_cells = n_total
        self.length  = float(z_starts[-1] + Lz[-1])

class scalarField:
    def __init__(self, variable: str, field_type: str = "specie"):
        self.variable   = variable
        self.field_type = field_type
        self.cellField  = None  
        
    def fieldInitialize(self, mesh):
        self.cellField  = np.zeros_like(mesh.cell_centers)

class solver:
    def __init__(self, mesh, mixture, reaction, specieFields: List, inlet):
        self.mesh       = mesh
        self.mixture    = mixture
        self.reaction   = reaction
        self.inlet      = inlet

        self.massFlux   = inlet.inletValues(mixture=mixture, domain=self.mesh.domain)[0]
        self.specieFields = np.vstack([f.cellField for f in specieFields])

        self.temperatureField = scalarField("temperature", "temperature")
        self.temperatureField.fieldInitialize(mesh)
        self.temperatureField.cellField[:] = self.inlet.temperature

        self.velocityField = scalarField("velocity", "velocity")
        self.velocityField.fieldInitialize(mesh)
        self.velocityField.cellField[:] = self.inlet.velocity
        
        self.density = self.mixture.idealGasDensity(
            T=self.temperatureField.cellField, 
            speciesFractions=self.specieFields
        )

        self.heatSourcesDerivative = np.zeros_like(self.temperatureField.cellField)
        self.massSources = np.zeros_like(self.specieFields)                 # (n_species, n_cells)
        self.massSourcesDerivative = np.zeros_like(self.specieFields)
        self.heatSources = np.zeros_like(self.temperatureField.cellField)   # (n_cells,)
        self.heatResidual = np.zeros_like(self.temperatureField.cellField)
        self.specieResidual = np.zeros_like(self.specieFields)

        self.reactionRates = np.zeros((2, mesh.n_cells), dtype=float)

    def concentrationArray(self):
        rho = np.asarray(self.density, dtype=float)              # (n_cells,)
        Y   = np.asarray(self.specieFields, dtype=float)         # (n_species, n_cells)
        M   = np.asarray(self.mixture.molarMasses, dtype=float)  # (n_species,)

        if Y.ndim != 2:
            raise ValueError(f"specieFields must be 2D, got shape {Y.shape}")
        if Y.shape[0] != M.shape[0]:
            raise ValueError(f"Species count mismatch: Y has {Y.shape[0]}, M has {M.shape[0]}")
        if Y.shape[1] != rho.shape[0]:
            raise ValueError(f"Cell count mismatch: Y has {Y.shape[1]}, rho has {rho.shape[0]}")

        C = rho[np.newaxis, :] * Y / M[:, np.newaxis]
        return C

    def update_density(self):
        self.density = self.mixture.idealGasDensity(
            T=self.temperatureField.cellField,
            speciesFractions=self.specieFields
        )
    def sourcesEvaluation(self):
        self.heatSources.fill(0.0)
        self.heatSourcesDerivative.fill(0.0)
        self.massSources.fill(0.0)
        self.massSourcesDerivative.fill(0.0)

        C   = self.concentrationArray()      # (n_species, n_cells)
        rho = self.density                   # (n_cells,)
        Y   = self.specieFields              # (n_species, n_cells)
        T   = self.temperatureField.cellField # Extract array once

        reactionMask = (self.mesh.cell_mass_flag == True)
        heatMask     = (self.mesh.cell_heat_flag == True)

        self.heatSources[heatMask] = np.asarray(self.mesh.cell_heat_value)[heatMask]
        rateForward, rateBackward = self.reaction.reactionRate(T, C)
        self.reactionRates[0, :]  = rateForward
        self.reactionRates[1, :]  = rateBackward
        massSources_all = self.reaction.reactionMassSource((rateForward, rateBackward))
        self.massSources[:, reactionMask] = massSources_all[:, reactionMask]

        M_species = self.reaction.molarMasses
        stoich    = self.reaction.stochiometricCoefficients
        for k in range(self.specieFields.shape[0]):
            dRf_dCk, dRb_dCk = self.reaction.rateDerivativeConcentration(T, C, j=k)
            dOmega_dC = stoich[k] * M_species[k] * (dRf_dCk - dRb_dCk)  

            dCk_dYk  = rho / M_species[k]
            dOmega_dYk = dOmega_dC * dCk_dYk
            self.massSourcesDerivative[k, reactionMask] = dOmega_dYk[reactionMask]

        heatReactionSource = self.reaction.reactionHeatSource(T, (rateForward, rateBackward))
        self.heatSources[reactionMask] += heatReactionSource[reactionMask]
        dQ_dT = self.reaction.reactionHeatSourceDerivative(T, (rateForward, rateBackward))
        self.heatSourcesDerivative[reactionMask] = dQ_dT[reactionMask]

    
    def matrixSpecieEquationAssembly(self, specieIndex: int):
        n = self.mesh.n_cells
        F = float(self.massFlux)
        V = self.mesh.cell_volumes


        A = (-1 * np.diag(np.ones(n-1), -1) + np.identity(n)) * F

        Y_old   = self.specieFields[specieIndex, :]
        S       = self.massSources[specieIndex, :]
        dS      = self.massSourcesDerivative[specieIndex, :]  

        zero_tol = 1e-10
        if np.max(Y_old) < zero_tol:
            dS = np.zeros(n)

        dS_implicit = np.minimum(dS, 0.0)
        A -= np.diag(dS_implicit * V)

        S_U = S - dS_implicit * Y_old  
        
        b = np.zeros(n, dtype=float)
        b[1:] = S_U[1:] * V[1:]
        A[0, 0] = F - dS_implicit[0] * V[0] 
        b[0]    = F * self.inlet.speciesMassFractions[specieIndex] + S_U[0] * V[0]

        return A, b
    
    def matrixTemperatureEquationAssembly(self):
        n       = self.mesh.n_cells
        F       = float(self.massFlux)
        
        T_old   = self.temperatureField.cellField 
        cp_mix  = self.mixture.mixtureHeatCapacity(T_old, self.specieFields)  
        Q       = self.heatSources                      
        dQ_dT   = self.heatSourcesDerivative       
        V       = self.mesh.cell_volumes               

        A = np.zeros((n, n), dtype=float)
        A += np.diag(-F * cp_mix)
        A += np.diag(F * cp_mix[1:], k=-1)

        dQ_dT = np.clip(dQ_dT, -1e10, 1e10)

        dQ_dT_implicit = np.minimum(dQ_dT, 0.0)
        A   += np.diag(dQ_dT_implicit * V)
        
        # FIX 2: Positive Q for heat generation
        Q_U = Q - dQ_dT_implicit * T_old 
        
        b = np.zeros(n, dtype=float)

        b[1:]   = Q_U[1:] * V[1:]
        A[0, 0] = F * cp_mix[0] - dQ_dT_implicit[0] * V[0]
        b[0]    = F * cp_mix[0] * self.inlet.temperature + Q_U[0] * V[0]

        return A, b
    
    def specieScalarEquation(self):
        F = float(self.massFlux)
        Y = self.specieFields
        S = self.massSources
        V = self.mesh.cell_volumes
        
        res = np.zeros_like(self.specieResidual)
        Y_in = np.asarray(self.inlet.speciesMassFractions, dtype=float)
        res[:, 0] = F * Y[:, 0] - F * Y_in - S[:, 0] * V[0]
        
        res[:, 1:] = F * Y[:, 1:] - F * Y[:, :-1] - S[:, 1:] * V[1:]
        
        self.specieResidual = res
        
    def heatEquation(self):
        F = float(self.massFlux)
        T = self.temperatureField.cellField
        Q = self.heatSources
        V = self.mesh.cell_volumes
        
        cp_mix = self.mixture.mixtureHeatCapacity(T, self.specieFields)
        res = np.zeros_like(self.heatResidual)
        cp_in = cp_mix[0]
        res[0] = F * cp_mix[0] * T[0] - F * cp_in * self.inlet.temperature - Q[0] * V[0]
        
        res[1:] = F * cp_mix[1:] * T[1:] - F * cp_mix[:-1] * T[:-1] - Q[1:] * V[1:]
        
        self.heatResidual = res

    def initializeCase(self):
        inletMassFlowrate, inletTemperature, inletSpecies = self.inlet.inletValues(self.mixture, self.mesh.domain)

        inletSpecies2d = np.asarray(inletSpecies, dtype=float).reshape(-1, 1)
        self.specieFields[:, :] = inletSpecies2d                               

        self.temperatureField.cellField[:] = float(inletTemperature)
        self.update_density()

    def steadyState(self, max_iter: int,
                    relaxationFactorSpecie: float,
                    relaxationFactorTemperature: float,
                    convergenceCriteria: float):

        self.update_density()
        
        area = (self.mesh.domain.diameter ** 2) * np.pi / 4.0
        self.velocityField.cellField[:] = self.massFlux / (self.density * area)
        
        self.sourcesEvaluation()

        for it in range(max_iter):
            Y_old = self.specieFields.copy()
            T_old = self.temperatureField.cellField.copy()

            for i in range(self.specieFields.shape[0]):
                A_Y, b_Y = self.matrixSpecieEquationAssembly(i)
                Y_star = scipy.linalg.solve(A_Y, b_Y)
                Y_star = np.clip(Y_star, 0.0, 1.0)

                self.specieFields[i, :] = (
                    Y_old[i, :] + relaxationFactorSpecie * (Y_star - Y_old[i, :])
                )

            total_mass = np.sum(self.specieFields, axis=0)
            total_mass = np.maximum(total_mass, 1e-16)
            self.specieFields /= total_mass
            self.update_density()
            self.velocityField.cellField[:] = self.massFlux / (self.density * area)

            self.sourcesEvaluation()
            A_T, b_T = self.matrixTemperatureEquationAssembly()
            T_star = scipy.linalg.solve(A_T, b_T)
            T_star = np.clip(T_star, 200.0, 2000.0)
            
            self.temperatureField.cellField[:] = (
                T_old + relaxationFactorTemperature * (T_star - T_old)
            )

            dY = np.max(np.abs(self.specieFields - Y_old))
            dT = np.max(np.abs(self.temperatureField.cellField - T_old))
            scaleddY = dY / max(self.massFlux, 1e-10)
            scaleddT = dT / max(np.mean(self.temperatureField.cellField), 1.0)

            if it % 10 == 0 or it == max_iter - 1:
                print(f"Iter {it:04d} | Max scaled dY: {scaleddY:.2e} | Max scaled dT: {scaleddT:.2e}")
                print(f"          | Outlet Temp: {self.temperatureField.cellField[-1]:.2f} K")
                print(f"          | Outlet Y: {self.specieFields[:, -1]}")

            if max(dY, dT) < convergenceCriteria:
                print(f"\nConverged successfully after {it} iterations!")
                break
        results = self.specieFields[:, -1]
        return results
            

class ReactorPlotter:
    def __init__(self, solver):
        """
        solver: object holding mesh, species, temperatureField, specieFields
        """
        self.solver = solver

    def get_axis(self):
        """
        Return axial coordinate. Adjust if you use a different name.
        """
        mesh = self.solver.mesh
        if hasattr(mesh, "cell_centers"):
            return mesh.cell_centers
        else:
            lengths = mesh.cell_lengths
            z_edges = np.concatenate(([0.0], np.cumsum(lengths)))
            return 0.5 * (z_edges[:-1] + z_edges[1:])

    def plot_temperature(self, show=True, ax=None):
        z = self.get_axis()
        T = self.solver.temperatureField.cellField

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(z, T, color="red", label="Temperature")
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Temperature [K]")
        ax.grid(True)
        ax.legend()
        
        if show and ax is None:
            plt.tight_layout()
            plt.show()

    def plot_species(self, show=True, ax=None):
        z = self.get_axis()
        Y = self.solver.specieFields

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        for i, sp in enumerate(self.solver.reaction.species):
            ax.plot(z, Y[i, :], label=sp.name)

        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Mass fraction [-]")
        ax.grid(True)
        ax.legend()
        
        if show and ax is None:
            plt.tight_layout()
            plt.show()

    def plot_all(self):
        """
        Convenience method: two subplots in one figure.
        """
        z = self.get_axis()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        ax1.plot(z, self.solver.temperatureField.cellField, color="red")
        ax1.set_xlabel("z [m]")
        ax1.set_ylabel("T [K]")
        ax1.grid(True)

        for i, sp in enumerate(self.solver.reaction.species):
            ax2.plot(z, self.solver.specieFields[i, :], label=sp.name)
        ax2.set_xlabel("z [m]")
        ax2.set_ylabel("Y_i [-]")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":

    UNIVERSALGASCONSTANT = 8.31446261815324
    cp_o2_coeffs = [3.53]
    cp_so2_coeffs = [3.53]
    cp_so3_coeffs = [3.53]
    cp_n2_coeffs = [3.53]

    SO2 = Specie(
        name="so2",
        molarMass= 0.64,
        heatCapacityModel="polynomial",
        enthalpyFormation=-296861.220,
        entropyFormation=248.100,   
        heatCapacityCoefficients=cp_so2_coeffs
    )

    O2 = Specie(
        name="o2",
        molarMass=.32,
        heatCapacityModel="polynomial",
        enthalpyFormation= 0.0,
        entropyFormation=205.000,    
        heatCapacityCoefficients=cp_o2_coeffs
    )

    N2 = Specie(
        name="n2",
        molarMass=.28,
        heatCapacityModel="polynomial",
        enthalpyFormation= 0.0,
        entropyFormation=191.500,    
        heatCapacityCoefficients=cp_n2_coeffs
    )

    SO3 = Specie(
        name="so3",
        molarMass=.80,
        heatCapacityModel="polynomial",
        enthalpyFormation=-395782.300,
        entropyFormation=256.600,    
        heatCapacityCoefficients=cp_so3_coeffs
    )
    species = [SO2, O2, SO3, N2]

    reaction = Reaction(
        name="so2conversion",
        stochiometricCoefficients=np.array([-1.0, -0.5, 1.0, 0.0]),
        speciesExponent=np.array([1.0, 0.5, 0.0, 0.0]),
        reversedSpecieExponent=np.array([0.0, 0.0, 1.0, 0.0]),
        isReversible=True,
        ahrreniusPreExponent= 10e10,
        ahrreniusActivationEnergy= 165000.,
        species=species
    )

    z0 = Zone(length=2.0, type="reaction")
    z0.zoneAssign(heating=False, reaction=True)
    z0.zoneAssignHeating(0.0)

    z1 = Zone(length=1.0, type="heating")
    z1.zoneAssign(heating=True, reaction=False)
    z1.zoneAssignHeating(5000.0)

    z2 = Zone(length=2.0, type="reaction")
    z2.zoneAssign(heating=False, reaction=True)
    z2.zoneAssignHeating(0.0)

    Y_so2 = 0.11
    Y_so3 = 1e-6
    Y_o2 = 0.21 * (1 - Y_so2)
    Y_n2 = 1 - Y_so2 - Y_o2 - Y_so3
    
    domain = domainSetup(
        diameter=2.5,
        inletMassFractions=np.array([Y_so2, Y_o2, Y_so3, Y_n2])
    )
    mesh = Mesh(domain=domain, zoneList=[z0, z1, z2], sizing=0.01)
    mesh.meshCreate()

    Yso2_field = scalarField("Y_so2")
    Yo2_field = scalarField("Y_o2")
    Yso3_field = scalarField("Y_so3")
    Yn2_field = scalarField("Y_n2")

    Yso2_field.fieldInitialize(mesh)
    Yo2_field.fieldInitialize(mesh)
    Yso3_field.fieldInitialize(mesh)
    Yn2_field.fieldInitialize(mesh)
    specieFields = [Yso2_field, Yo2_field, Yso3_field, Yn2_field]
    
    inlet = Inlet(0, 1, 690, [Y_so2, Y_o2, Y_so3, Y_n2])
    mixture = Mixture(
        densityModel="ideal-incompressible-gas",
        densityValue=0.457,
        species=species
    )

    sol = solver(mesh=mesh, mixture=mixture, reaction=reaction, specieFields=specieFields, inlet=inlet)
    sol.initializeCase()

    sol.steadyState(max_iter=350, relaxationFactorSpecie=0.1, relaxationFactorTemperature=0.1, convergenceCriteria=1e-5)
    
    plotter = ReactorPlotter(sol)
    plotter.plot_all()

    z = sol.mesh.cell_centers
    plt.figure(figsize=(10, 4))
    plt.plot(z, sol.heatSources, label="Q_rxn + Q_ext", color='purple')
    plt.axvspan(0.0, 1.0, color="red", alpha=0.1, label="Rxn zone 1")
    plt.axvspan(1.0, 2.0, color="blue", alpha=0.1, label="Cooling")
    plt.axvspan(2.0, 3.0, color="red", alpha=0.1, label="Rxn zone 2")
    plt.xlabel("Axial position z [m]")
    plt.ylabel("Heat Source [W/m³]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()