import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

UNIVERSALGASCONSTANT = 8.31446261815324 #J/molK
T_REF   = 273.15
P_REF   = 101325.0 #Pa
PI      = 3.141592653589793
from pathlib import Path
import yaml

@dataclass
class Specie:
   
    counter: ClassVar[int] = 0
    name: str
    molarMass: float
    heatCapacityModel: str = "const"
    enthalpyFormation: float = 0.0
    entropyFormation: float = 0.0
    heatCapacityValue: float = 900.0
    heatCapacityCoefficients: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.id = Specie.counter
        type(self).counter += 1
        if self.heatCapacityModel == "polynomial":
            self.cp_poly = np.polynomial.Polynomial(self.heatCapacityCoefficients)
            self.H_poly = self.cp_poly.integ()

    @classmethod
    def from_dict(cls, name: str, cfg: dict) -> "Specie":
        data = dict(cfg)
        if "heatCapacityCoefficients" in data and data["heatCapacityCoefficients"] is None:
            data["heatCapacityCoefficients"] = []
        return cls(name=name, **data)
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
    @classmethod
    def from_dict(cls, name: str, cfg: dict, species: list[Specie]) -> "Reaction":
        data = dict(cfg)

        data["stochiometricCoefficients"] = np.array(
            data["stochiometricCoefficients"], dtype=float
        )
        data["speciesExponent"] = np.array(
            data["speciesExponent"], dtype=float
        )
        data["reversedSpecieExponent"] = np.array(
            data["reversedSpecieExponent"], dtype=float
        )

        data["isReversible"] = bool(data.get("isReversible", True))
        data["ahrreniusPreExponent"] = float(data.get("ahrreniusPreExponent", 1.0))
        data["ahrreniusActivationEnergy"] = float(data.get("ahrreniusActivationEnergy", 0.0))

        return cls(name=name, species=species, **data)
    
    def _equilibriumMask(self,
                        rateForward: np.ndarray,
                        rateBackward: np.ndarray,
                        rel_tol: float = 1e-2,
                        abs_tol: float = 1e-4) -> np.ndarray:
        rateForward = np.asarray(rateForward, dtype=float)
        rateBackward = np.asarray(rateBackward, dtype=float)

        net_rate = rateForward - rateBackward
        scale = np.maximum(np.maximum(np.abs(rateForward), np.abs(rateBackward)), 1.0)

        return np.abs(net_rate) <= (rel_tol * scale + abs_tol)


    def forwardRateConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.maximum(np.asarray(T, dtype=float), 1e-12)
        arg = -self.ahrreniusActivationEnergy / (UNIVERSALGASCONSTANT * T)
        return self.ahrreniusPreExponent * np.exp(arg)


    def equilibriumConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.maximum(np.asarray(T, dtype=float), 1e-12)
        delta_G = self.enthalpyChange - T * self.entropyChange

        arg = -delta_G / (UNIVERSALGASCONSTANT * T)
        K_p = np.exp(arg)
        return np.clip(K_p, 1e-100, 1e100)


    def backwardRateConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.maximum(np.asarray(T, dtype=float), 1e-12)
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


    def reactionRate(self, T: np.ndarray, concentrations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T, dtype=float)
        C = np.asarray(concentrations, dtype=float)

        C_safe = np.maximum(C, 1e-20)
        alpha = np.asarray(self.speciesExponent, dtype=float)[:, np.newaxis]
        massActionForward = np.prod(C_safe ** alpha, axis=0)

        rateForward = self.forwardRateConstant(T) * massActionForward

        if not self.isReversible:
            return rateForward, np.zeros_like(rateForward)

        beta = np.asarray(self.reversedSpecieExponent, dtype=float)[:, np.newaxis]
        massActionBackward = np.prod(C_safe ** beta, axis=0)

        rateBackward = self.backwardRateConstant(T) * massActionBackward

        return rateForward, rateBackward


    def reactionMassSource(self,
                        rates: tuple,
                        rel_tol: float = 1e-5,
                        abs_tol: float = 1e-12) -> np.ndarray:
        rateForward, rateBackward = rates
        rateForward = np.asarray(rateForward, dtype=float)
        rateBackward = np.asarray(rateBackward, dtype=float)

        net_rate = rateForward - rateBackward
        near_eq = self._equilibriumMask(rateForward, rateBackward, rel_tol=rel_tol, abs_tol=abs_tol)

        net_rate = net_rate.copy()
        net_rate[near_eq] = 0.0

        massSources = (
            net_rate[np.newaxis, :]
            * self.stochiometricCoefficients[:, np.newaxis]
            * self.molarMasses[:, np.newaxis]
        )
        return massSources


    def reactionHeatSource(self,
                        T: np.ndarray,
                        rates: tuple,
                        rel_tol: float = 1e-5,
                        abs_tol: float = 1e-12):
        rateForward, rateBackward = rates
        rateForward = np.asarray(rateForward, dtype=float)
        rateBackward = np.asarray(rateBackward, dtype=float)

        delta_H_rxn = self.enthalpyReactionChange(T)
        net_rate = rateForward - rateBackward
        near_eq = self._equilibriumMask(rateForward, rateBackward, rel_tol=rel_tol, abs_tol=abs_tol)

        q = -net_rate * delta_H_rxn
        q = np.asarray(q, dtype=float)
        q[near_eq] = 0.0
        return q


    def reactionHeatSourceDerivative(self,
                                    T: np.ndarray,
                                    rates: tuple,
                                    rel_tol: float = 1e-4,
                                    abs_tol: float = 1e-9) -> np.ndarray:
        T = np.maximum(np.asarray(T, dtype=float), 1e-3)
        rateForward, rateBackward = rates
        rateForward = np.asarray(rateForward, dtype=float)
        rateBackward = np.asarray(rateBackward, dtype=float)

        near_eq = self._equilibriumMask(rateForward, rateBackward, rel_tol=rel_tol, abs_tol=abs_tol)

        delta_cp = np.zeros_like(T)
        for nu_i, sp in zip(self.stochiometricCoefficients, self.species):
            delta_cp += nu_i * sp.heatCapacity(T)

        delta_H_rxn = self.enthalpyReactionChange(T)
        delta_nu = float(np.sum(self.stochiometricCoefficients))

        R = UNIVERSALGASCONSTANT
        term_den = R * T**2

        dRf_dT = rateForward * (self.ahrreniusActivationEnergy / term_den)

        if self.isReversible:
            dRb_dT = rateBackward * (
                (self.ahrreniusActivationEnergy - delta_H_rxn) / term_den
                + delta_nu / T
            )
        else:
            dRb_dT = np.zeros_like(dRf_dT)

        net_rate = rateForward - rateBackward
        dQ_dT = -(dRf_dT - dRb_dT) * delta_H_rxn - net_rate * delta_cp

        dQ_dT = np.asarray(dQ_dT, dtype=float)
        dQ_dT[near_eq] = 0.0

        return dQ_dT


    def rateDerivativeConcentration(self,
                                    T: np.ndarray,
                                    C: np.ndarray,
                                    j: int) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T, dtype=float)
        C = np.asarray(C, dtype=float)

        alpha = np.asarray(self.speciesExponent, dtype=float)[:, np.newaxis]
        C_safe = np.maximum(C, 1e-20)
        massActionForward = np.prod(C_safe ** alpha, axis=0)
        k_f = self.forwardRateConstant(T)

        alpha_j = alpha[j, 0]
        Cj_safe = np.maximum(C[j, :], 1e-20)

        dM_f_dCj = alpha_j * massActionForward / Cj_safe
        dRf_dCj = k_f * dM_f_dCj

        if not self.isReversible:
            return dRf_dCj, np.zeros_like(dRf_dCj)

        beta = np.asarray(self.reversedSpecieExponent, dtype=float)[:, np.newaxis]
        massActionBackward = np.prod(C_safe ** beta, axis=0)
        k_b = self.backwardRateConstant(T)

        beta_j = beta[j, 0]
        dM_b_dCj = beta_j * massActionBackward / Cj_safe
        dRb_dCj = k_b * dM_b_dCj

        return dRf_dCj, dRb_dCj
        
@dataclass
class Mixture:

    densityModel    : str           = "ideal-incompressible-gas"   # or "const"
    densityValue    : float         = 1.2225                       # used explicitly if densityModel == "const"
    species         : List          = field(default_factory=list, repr=False)
    molarMasses     : np.ndarray    = field(init=False)

    @classmethod
    def from_dict(cls, cfg: dict, species: list[Specie]) -> "Mixture":
        density_value = cfg.get("densityValue", 1.2225)
        if density_value is None:
            density_value = 1.2225
        return cls(
            densityModel=cfg.get("densityModel", "ideal-incompressible-gas"),
            densityValue=float(density_value),
            species=species,
        )
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
        if mixture.densityModel == "ideal-incompressible-gas":
            density = mixture.idealGasDensity(T=T_array,
                                            speciesFractions=Y_in_2d)         # rho: (1,)
        else:
            density = mixture.densityValue

        area          = (domain.diameter ** 2) * PI / 4.0
        massFlowrate  = density[0] * area * self.velocity                 # Float
        temperatureBC = float(self.temperature)                           # Float
        specieFrac    = Y_in                                              # 1D Array

        return massFlowrate, temperatureBC, specieFrac
    
    @classmethod
    def from_dict(cls, cfg: dict, species: list[Specie]) -> "Inlet":
        order = [sp.name for sp in species]
        specie_map = cfg["specie"]
        y = np.array([float(specie_map[name]) for name in order], dtype=float)

        total = np.sum(y)
        if total <= 0.0:
            raise ValueError("Inlet species mass fractions must sum to > 0")
        y /= total

        return cls(
            position=int(cfg.get("position", 0)),
            velocity=float(cfg["velocity"]),
            temperature=float(cfg["temperature"]),
            speciesMassFractions=y.tolist(),
        )

@dataclass
class Outlet:


    position: int = -1
    temperature: float = 0.0
    speciesMassFractions: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    density: float = 0.0
    velocity: float = 0.0
    massFlowrate: float = 0.0
    concentrations: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @classmethod
    def fromSolver(cls, slv: "solver", position: int = -1) -> "Outlet":
        """
        Build Outlet object from a converged solver instance.

        Parameters
        ----------
        slv : solver
            Solver object containing final field values.
        position : int, optional
            Cell index from which outlet data are extracted.
            Default is -1 (last cell).

        Returns
        -------
        Outlet
            Populated outlet state.
        """
        T = float(slv.temperatureField.cellField[position])
        Y = np.asarray(slv.specieFields[:, position], dtype=float).copy()
        rho = float(slv.density[position])
        u = float(slv.velocityField.cellField[position])
        mdot = float(slv.massFlux)

        C = rho * Y / np.asarray(slv.mixture.molarMasses, dtype=float)

        return cls(
            position=position,
            temperature=T,
            speciesMassFractions=Y,
            density=rho,
            velocity=u,
            massFlowrate=mdot,
            concentrations=C,
        )

    def asDict(self, species=None) -> dict:
        """
        Return outlet data as a plain dictionary.

        Parameters
        ----------
        species : list[Specie], optional
            Species list used to label mass fractions and concentrations.

        Returns
        -------
        dict
            Dictionary representation of outlet data.
        """
        data = {
            "position": self.position,
            "temperature": self.temperature,
            "density": self.density,
            "velocity": self.velocity,
            "massFlowrate": self.massFlowrate,
            "speciesMassFractions": self.speciesMassFractions.copy(),
            "concentrations": self.concentrations.copy(),
        }

        if species is not None:
            data["speciesMassFractions"] = {
                sp.name: float(self.speciesMassFractions[i])
                for i, sp in enumerate(species)
            }
            data["concentrations"] = {
                sp.name: float(self.concentrations[i])
                for i, sp in enumerate(species)
            }

        return data

    def specieIndex(self, name: str, species: list) -> int:
        """
        Return index of species with given name.
        """
        for i, sp in enumerate(species):
            if sp.name.lower() == name.lower():
                return i
        raise ValueError(f"Species '{name}' not found in provided species list.")

    def massFraction(self, name: str, species: list) -> float:
        """
        Return outlet mass fraction of the requested species.
        """
        i = self.specieIndex(name, species)
        return float(self.speciesMassFractions[i])

    def concentration(self, name: str, species: list) -> float:
        """
        Return outlet molar concentration of the requested species [mol/m^3].
        """
        i = self.specieIndex(name, species)
        return float(self.concentrations[i])


@dataclass
class Zone:

    counter: ClassVar[int] = 0
    length: float = 0.005
    zoneType: str = "null"

    def __post_init__(self):
        type(self).counter += 1
        self.id = type(self).counter
        self.heatSource = False
        self.massSource = False
        self.heatSourceValue = 0.0

    @classmethod
    def from_dict(cls, name: str, cfg: dict) -> "Zone":
        zone = cls(
            length=float(cfg["length"]),
            zoneType=cfg.get("zoneType", name),
        )

        zone.heatSource = bool(cfg.get("heatSource", False))
        zone.massSource = bool(cfg.get("massSource", True))

        heat_value = cfg.get("heatSourceValue", 0.0)
        if heat_value is None:
            heat_value = 0.0
        zone.heatSourceValue = float(heat_value)

        return zone

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
        zone_types = np.array([z.zoneType for z in zones], dtype=object)  # (Nz,)
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

    @classmethod
    def from_dict(cls, cfg: dict, domain, zones: list["Zone"]) -> "Mesh":
        mesh_block = cfg.get("mesh", {})
        sizing = float(mesh_block.get("sizing", 0.005))

        mesh = cls(
            domain=domain,
            zoneList=zones,
            sizing=sizing,
        )
        mesh.meshCreate()
        return mesh

class scalarField:

    def __init__(self, variable: str, field_type: str = "specie"):
        self.variable   = variable
        self.field_type = field_type
        self.cellField  = None  
        
    def fieldInitialize(self, mesh):
        self.cellField  = np.zeros_like(mesh.cell_centers)

class solver:
 

    def __init__(self, mesh, mixture, reactions: List["Reaction"], specieFields: List, inlet):
        self.mesh = mesh
        self.mixture = mixture

        if isinstance(reactions, Reaction):
            reactions = [reactions]
        self.reactions = reactions
        self.inlet = inlet
        self.outlet = Outlet()
        self.massFlux = inlet.inletValues(mixture=mixture, domain=self.mesh.domain)[0]
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
        self.massSources = np.zeros_like(self.specieFields)
        self.massSourcesDerivative = np.zeros_like(self.specieFields)
        self.heatSources = np.zeros_like(self.temperatureField.cellField)
        self.heatResidual = np.zeros_like(self.temperatureField.cellField)
        self.specieResidual = np.zeros_like(self.specieFields)

        self.reactionRates = np.zeros((len(self.reactions), 2, mesh.n_cells), dtype=float)

    def concentrationArray(self):
        rho = np.asarray(self.density, dtype=float)
        Y = np.asarray(self.specieFields, dtype=float)
        M = np.asarray(self.mixture.molarMasses, dtype=float)

        if Y.ndim != 2:
            raise ValueError(f"specieFields must be 2D, got shape {Y.shape}")
        if Y.shape[0] != M.shape[0]:
            raise ValueError(f"Species count mismatch: Y has {Y.shape[0]}, M has {M.shape[0]}")
        if Y.shape[1] != rho.shape[0]:
            raise ValueError(f"Cell count mismatch: Y has {Y.shape[1]}, rho has {rho.shape[0]}")

        return rho[np.newaxis, :] * Y / M[:, np.newaxis]

    def update_density(self):
        self.density = self.mixture.idealGasDensity(
            T=self.temperatureField.cellField,
            speciesFractions=self.specieFields
        )

    def sourcesEvaluation(
        self,
        underRelaxationFactorHeatSource=0.05,
        underRelaxationFactorMassSource=0.15,
        eq_rel_tol=1e-8,
        eq_abs_tol=1e-20,
    ):
        mass_old = self.massSources.copy()
        massd_old = self.massSourcesDerivative.copy()

        heat_rxn_old = getattr(self, "heatReactionSources", np.zeros_like(self.heatSources))
        heatd_rxn_old = getattr(self, "heatReactionSourcesDerivative", np.zeros_like(self.heatSourcesDerivative))

        self.heatSources.fill(0.0)
        self.heatSourcesDerivative.fill(0.0)
        self.massSources.fill(0.0)
        self.massSourcesDerivative.fill(0.0)

        if not hasattr(self, "heatReactionSources"):
            self.heatReactionSources = np.zeros_like(self.heatSources)
        if not hasattr(self, "heatReactionSourcesDerivative"):
            self.heatReactionSourcesDerivative = np.zeros_like(self.heatSourcesDerivative)

        self.heatReactionSources.fill(0.0)
        self.heatReactionSourcesDerivative.fill(0.0)

        C = self.concentrationArray()
        rho = self.density
        T = self.temperatureField.cellField

        reactionMask = self.mesh.cell_mass_flag
        heatMask = self.mesh.cell_heat_flag

        self.heatSources[heatMask] += np.asarray(self.mesh.cell_heat_value, dtype=float)[heatMask]

        n_species = self.specieFields.shape[0]

        for r_idx, rxn in enumerate(self.reactions):
            rateForward, rateBackward = rxn.reactionRate(T, C)
            self.reactionRates[r_idx, 0, :] = rateForward
            self.reactionRates[r_idx, 1, :] = rateBackward

            net_rate = rateForward - rateBackward
            rate_scale = np.maximum(np.maximum(np.abs(rateForward), np.abs(rateBackward)), 1.0)
            near_eq = np.abs(net_rate) <= (eq_rel_tol * rate_scale + eq_abs_tol)
            near_eq = near_eq & reactionMask

            massSources_r = (
                net_rate[np.newaxis, :]
                * rxn.stochiometricCoefficients[:, np.newaxis]
                * rxn.molarMasses[:, np.newaxis]
            )
            massSources_r[:, near_eq] = 0.0
            self.massSources[:, reactionMask] += massSources_r[:, reactionMask]

            M_species = rxn.molarMasses
            stoich = rxn.stochiometricCoefficients

            for k in range(n_species):
                dRf_dCk, dRb_dCk = rxn.rateDerivativeConcentration(T, C, j=k)
                dOmega_dC = stoich[k] * M_species[k] * (dRf_dCk - dRb_dCk)
                dOmega_dC[near_eq] = 0.0

                dCk_dYk = rho / M_species[k]
                dOmega_dYk = dOmega_dC * dCk_dYk
                self.massSourcesDerivative[k, reactionMask] += dOmega_dYk[reactionMask]

            deltaHrxn = rxn.enthalpyReactionChange(T)
            heatReactionSource_r = -net_rate * deltaHrxn
            dQ_dT_r = rxn.reactionHeatSourceDerivative(T, (rateForward, rateBackward))

            heatReactionSource_r[near_eq] = 0.0
            dQ_dT_r[near_eq] = 0.0

            self.heatReactionSources[reactionMask] += heatReactionSource_r[reactionMask]
            self.heatReactionSourcesDerivative[reactionMask] += dQ_dT_r[reactionMask]

        self.heatReactionSources = (
            (1.0 - underRelaxationFactorHeatSource) * heat_rxn_old
            + underRelaxationFactorHeatSource * self.heatReactionSources
        )
        self.heatReactionSourcesDerivative = (
            (1.0 - underRelaxationFactorHeatSource) * heatd_rxn_old
            + underRelaxationFactorHeatSource * self.heatReactionSourcesDerivative
        )

        self.heatSources += self.heatReactionSources
        self.heatSourcesDerivative[:] = self.heatReactionSourcesDerivative

        self.massSources = (
            (1.0 - underRelaxationFactorMassSource) * mass_old
            + underRelaxationFactorMassSource * self.massSources
        )
        self.massSourcesDerivative = (
            (1.0 - underRelaxationFactorMassSource) * massd_old
            + underRelaxationFactorMassSource * self.massSourcesDerivative
        )

    def matrixSpecieEquationAssembly(self, specieIndex: int):
        n = self.mesh.n_cells
        F = float(self.massFlux)
        V = self.mesh.cell_volumes

        Y_old = self.specieFields[specieIndex, :]
        S = self.massSources[specieIndex, :]
        dS = self.massSourcesDerivative[specieIndex, :]

        zero_tol = 1e-10
        if np.max(Y_old) < zero_tol:
            dS = np.zeros(n)

        dS_implicit = np.minimum(dS, 0.0)
        S_U = S - dS_implicit * Y_old

        main = np.full(n, F, dtype=float) - dS_implicit * V
        lower = np.full(n - 1, -F, dtype=float)

        b = np.zeros(n, dtype=float)
        b[1:] = S_U[1:] * V[1:]
        b[0] = F * self.inlet.speciesMassFractions[specieIndex] + S_U[0] * V[0]

        main[0] = F - dS_implicit[0] * V[0]

        A = diags(
            diagonals=[lower, main],
            offsets=[-1, 0],
            shape=(n, n),
            format="csr",
            dtype=float,
        )

        return A, b
    
    def matrixTemperatureEquationAssembly(self):
        n = self.mesh.n_cells
        F = float(self.massFlux)

        T_old = np.asarray(self.temperatureField.cellField, dtype=float)
        cp_mix = np.asarray(self.mixture.mixtureHeatCapacity(T_old, self.specieFields), dtype=float)
        Q = np.asarray(self.heatSources, dtype=float)
        dQ_dT = np.asarray(self.heatSourcesDerivative, dtype=float)
        V = np.asarray(self.mesh.cell_volumes, dtype=float)

        Sp = np.minimum(dQ_dT, 0.0)
        Su = Q - Sp * T_old

        main = F * cp_mix - Sp * V
        lower = -F * cp_mix[:-1]

        b = np.zeros(n, dtype=float)
        b[1:] = Su[1:] * V[1:]
        b[0] = F * cp_mix[0] * self.inlet.temperature + Su[0] * V[0]

        A = diags(
            diagonals=[lower, main],
            offsets=[-1, 0],
            shape=(n, n),
            format="csr",
            dtype=float,
        )

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

    def steadyState(self, maxiter: int,
                    relaxationFactorSpecie: float = 0.4,
                    relaxationFactorTemperature: float = 0.4,
                    convergenceCriteria: float = 1e-6,
                    temperatureClipLow: float = 200,
                    temperatureClipHigh : float = 2000):

        self.update_density()
        
        area = (self.mesh.domain.diameter ** 2) * np.pi / 4.0
        self.velocityField.cellField[:] = self.massFlux / (self.density * area)
        
        self.sourcesEvaluation()

        for it in range(maxiter):
            Y_old = self.specieFields.copy()
            T_old = self.temperatureField.cellField.copy()

            for i in range(self.specieFields.shape[0]):
                A_Y, b_Y = self.matrixSpecieEquationAssembly(i)
                Y_star = spsolve(A_Y, b_Y)
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
            T_star = spsolve(A_T, b_T)
            T_star = np.clip(T_star, temperatureClipLow, temperatureClipHigh)
            
            self.temperatureField.cellField[:] = (
                T_old + relaxationFactorTemperature * (T_star - T_old)
            )

            dY = np.max(np.abs(self.specieFields - Y_old))
            dT = np.max(np.abs(self.temperatureField.cellField - T_old))
            scaleddY = dY / max(self.massFlux, 1e-10)
            scaleddT = dT / max(np.mean(self.temperatureField.cellField), 1.0)

            if it % 10 == 0 or it == maxiter - 1:
                print(f"Iter {it:04d} | Max scaled dY: {scaleddY:.2e} | Max scaled dT: {scaleddT:.2e}")
                print(f"          | Outlet Temp: {self.temperatureField.cellField[-1]:.2f} K")
                print(f"          | Outlet Y: {self.specieFields[:, -1]}")

            if max(dY, dT) < convergenceCriteria:
                print(f"\nConverged successfully after {it} iterations!")
                break

        self.outlet = Outlet.fromSolver(self)
        return self.outlet
    
def build_reactor_from_context(case_ctx):
    chemistry = case_ctx["chemistry"]
    mesh_cfg = case_ctx["mesh"]
    inlet_cfg = case_ctx["inlet"]

    Specie.counter = 0
    Zone.counter = 0

    species = [
        Specie.from_dict(name, cfg)
        for name, cfg in chemistry["species"].items()
    ]

    reaction_items = list(chemistry["reactions"].items())
    if len(reaction_items) < 1:
        raise ValueError("At least one reaction must be defined")

    reactions = [
        Reaction.from_dict(name, cfg, species)
        for name, cfg in reaction_items
    ]

    # Guard: every reaction must reference the same species ordering as
    # the mixture, otherwise summed sources will silently misalign.
    species_names = [sp.name for sp in species]
    for rxn in reactions:
        rxn_names = [sp.name for sp in rxn.species]
        if rxn_names != species_names:
            raise ValueError(
                f"Reaction '{rxn.name}' species ordering {rxn_names} "
                f"does not match mixture species ordering {species_names}"
            )

    mixture = Mixture.from_dict(chemistry["mixture"], species)

    domain = domainSetup(diameter=float(inlet_cfg["diameter"]))

    zones = [
        Zone.from_dict(name, cfg)
        for name, cfg in mesh_cfg["zones"].items()
    ]

    mesh = Mesh.from_dict(mesh_cfg, domain, zones)
    inlet = Inlet.from_dict(inlet_cfg, species)

    specie_fields = []
    for sp in species:
        fld = scalarField(f"Y_{sp.name}", "specie")
        fld.fieldInitialize(mesh)
        specie_fields.append(fld)

    slv = solver(
        mesh=mesh,
        mixture=mixture,
        reactions=reactions,
        specieFields=specie_fields,
        inlet=inlet,
    )

    return slv, species

class ReactorPlotter:

    def __init__(self, solver):
        self.solver = solver

    def get_axis(self):
        mesh = self.solver.mesh
        if hasattr(mesh, "cell_centers"):
            return mesh.cell_centers
        lengths = mesh.cell_lengths
        z_edges = np.concatenate(([0.0], np.cumsum(lengths)))
        return 0.5 * (z_edges[:-1] + z_edges[1:])

    def save_temperature(self, path, dpi=200):
        z = self.get_axis()
        T = self.solver.temperatureField.cellField

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z, T, color="red", lw=2, label="Temperature")
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Temperature [K]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _species_list(self):
        # Species now live on the mixture, not on a single reaction object,
        # since the solver supports multiple simultaneous reactions.
        if hasattr(self.solver, "mixture") and hasattr(self.solver.mixture, "species"):
            return self.solver.mixture.species
        if hasattr(self.solver, "reactions") and self.solver.reactions:
            return self.solver.reactions[0].species
        if hasattr(self.solver, "reaction"):
            return self.solver.reaction.species
        raise AttributeError("Could not determine species list from solver")

    def save_species(self, path, dpi=200):
        z = self.get_axis()
        Y = self.solver.specieFields
        species = self._species_list()

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, sp in enumerate(species):
            ax.plot(z, Y[i, :], lw=2, label=sp.name)
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Mass fraction [-]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_species_subset(self, path, names, dpi=200):
        z = self.get_axis()
        Y = self.solver.specieFields
        species = self._species_list()
        name_set = {n.lower() for n in names}

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, sp in enumerate(species):
            if sp.name.lower() in name_set:
                ax.plot(z, Y[i, :], lw=2, label=sp.name)
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Mass fraction [-]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_all(self, path, dpi=200):
        z = self.get_axis()
        species = self._species_list()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        ax1.plot(z, self.solver.temperatureField.cellField, color="red", lw=2)
        ax1.set_xlabel("z [m]")
        ax1.set_ylabel("T [K]")
        ax1.grid(True, alpha=0.3)

        for i, sp in enumerate(species):
            ax2.plot(z, self.solver.specieFields[i, :], lw=2, label=sp.name)
        ax2.set_xlabel("z [m]")
        ax2.set_ylabel("Y_i [-]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_reaction_rates(self, path, dpi=200):
        """
        Plot forward/backward rate of each reaction vs axial position.
        Requires solver.reactionRates with shape (n_reactions, 2, n_cells)
        and solver.reactions: List[Reaction].
        """
        z = self.get_axis()
        rates = getattr(self.solver, "reactionRates", None)
        reactions = getattr(self.solver, "reactions", None)
        if rates is None or reactions is None:
            raise AttributeError(
                "solver.reactionRates / solver.reactions not available; "
                "multi-reaction rate tracking is required for this plot"
            )

        fig, ax = plt.subplots(figsize=(8, 4))
        for r_idx, rxn in enumerate(reactions):
            rf = rates[r_idx, 0, :]
            rb = rates[r_idx, 1, :]
            ax.plot(z, rf, lw=2, label=f"{rxn.name} (fwd)")
            if rxn.isReversible:
                ax.plot(z, rb, lw=2, ls="--", label=f"{rxn.name} (bwd)")
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Rate [mol/(m^3 s)]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_heat_source(self, path, dpi=200):
        z = self.get_axis()
        q = getattr(self.solver, "heatReactionSources", None)
        if q is None:
            q = self.solver.heatSources

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z, q, color="orange", lw=2, label="Reaction heat source")
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Heat source [W/m^3]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_concentrations(self, path, dpi=200):
        z = self.get_axis()
        species = self._species_list()
        C = self.solver.concentrationArray()

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, sp in enumerate(species):
            ax.plot(z, C[i, :], lw=2, label=sp.name)
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel("Concentration [mol/m^3]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)