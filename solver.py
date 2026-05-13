import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from dataclasses import dataclass, field
import scipy.linalg
from typing import ClassVar, Dict, List


PI = 3.141592653589793
UNIVERSALGASCONSTANT = 8314.46261815324 #J/kmolK
T_REF = 273.15
P_REF = 101325.0 #Pa

@dataclass
class Specie:
    counter: ClassVar[int] = 0

    name : str
    molarMass : float
    heatCapacityModel : str = "const" #""polynomial" also available"
    enthalpyFormation : float = 0.0
    entropyFormation : float = 0.0
    heatCapacityValue : float = 900
    heatCapacityCoefficients : list[float] = field(default_factory= lambda : [])
    
    def __post_init__(self):
        self.field = scalarField(self.name, "specie")
        self.id = Specie.counter
        type(self).counter += 1
        if self.heatCapacityModel == "polynomial":
            self.heatCapacityValue = np.poly1d(self.heatCapacityCoefficients) #np polynomial object, function of some value (in this case T)
    def enthalpy(self, T):
        T = np.asarray(T, dtype=float)
        if self.heatCapacityModel == "const":

            cp_const = self.heatCapacityValue
            return cp_const * (T - T_REF)

        else:  # polynomial cp(T)
            cp_poly = self.heatCapacityValue  # np.poly1d
            H_poly = cp_poly.integ()          # antiderivative polynomial
            return H_poly(T) - H_poly(T_REF)

@dataclass
class Reaction:
    name: str
    stochiometricCoefficients: np.ndarray = field(default_factory=list) #Must be lenght of all the species
    # speciesID: np.ndarray = field(default_factory=list)
    speciesExponent: np.ndarray = field(default_factory=list) #Must be lenght of all the species
    reversedSpecieExponent : np.ndarray = field(default_factory=list) #Must be lenght of all the species
    isReversible: bool = True
    ahrreniusPreExponent : float = 1.0
    ahrreniusActivationEnergy: float = 0.0  # J/mol
    species: List[Specie] = field(default_factory=dict, repr=False) #Just simply pass list of all the species
    entropyChange: float = field(init=False)   # J/mol/K
    enthalpyChange: float = field(init=False)  # J/mol
    molarMasses : List[float] = field(init=False)

    def __post_init__(self):
        # consistency check
        if not (len(self.stochiometricCoefficients) == len(self.speciesExponent) and len(self.stochiometricCoefficients) == len(self.reversedSpecieExponent)):
            raise ValueError("stochiometricCoefficients and exponents require this same lenght")
        self.enthalpyChange = 0.0
        self.entropyChange = 0.0
        self.molarMasses = []

        for nu, specie in zip(self.stochiometricCoefficients, self.species):
            self.enthalpyChange += nu * specie.enthalpyFormation   # J/mol
            self.entropyChange += nu * specie.entropyFormation     # J/mol/K
            self.molarMasses.append(specie.molarMass)
        self.molarMasses = np.asanyarray(self.molarMasses)

    def forwardRateConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T)
        """Arrhenius forward rate constant k_f(T)."""
        return self.ahrreniusPreExponent * np.exp(-self.ahrreniusActivationEnergy / (UNIVERSALGASCONSTANT * T))

    def equilibriumConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T)
        """Equilibrium constant from deltaG = deltaH - TdeltaS"""
        delta_G = self.enthalpyChange - T * self.entropyChange  # J/mol
        return np.exp(-delta_G / (UNIVERSALGASCONSTANT * T))

    def backwardRateConstant(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T)
        """Backward rate constant from k_b = k_f / K_eq."""
        if not self.isReversible:
            return np.zeros_like(T)
        k_f = self.forwardRateConstant(T)
        K_eq = self.equilibriumConstant(T)
        return k_f / K_eq
    
    def enthalpyReactionChange(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        Hr_T = np.zeros_like(T)

        for nu_i, sp in zip(self.stochiometricCoefficients, self.species):
            dH_i = sp.enthalpy(T)                # H(T) - H(T_ref)
            H_i_T = sp.enthalpyFormation + dH_i  # absolute H_i(T)
            Hr_T += nu_i * H_i_T

        return Hr_T
    
    def reactionRate(self, T: np.ndarray, concentrations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T, dtype=float)                      # (n_cells,)
        C = np.asarray(concentrations, dtype=float)         # (n_species, n_cells)

        alpha = np.asarray(self.speciesExponent, dtype=float)[:, np.newaxis]
        massActionForward = np.prod(C ** alpha, axis=0)     # (n_cells,)

        k_f = self.forwardRateConstant(T)                   # (n_cells,)
        rateForward = k_f * massActionForward               # (n_cells,)

        if not self.isReversible:
            rateBackward = np.zeros_like(rateForward)
            return rateForward, rateBackward

        beta = np.asarray(self.reversedSpecieExponent, dtype=float)[:, np.newaxis]
        massActionBackward = np.prod(C ** beta, axis=0)
        k_b = self.backwardRateConstant(T)
        rateBackward = k_b * massActionBackward

        return rateForward, rateBackward

    def reactionMassSource(self, rates : tuple) -> np.ndarray:
        rateForward, rateBackward = rates
        massSources = ((rateForward - rateBackward)[np.newaxis, :] * self.stochiometricCoefficients[:, np.newaxis]) * self.molarMasses[:, np.newaxis]
        return massSources

    def reactionHeatSource(self, T : np.ndarray, rates : tuple):
        rateForward, rateBackward = rates
        enthalpy = self.enthalpyReactionChange(T= T)
        return (rateForward - rateBackward) * enthalpy
    
@dataclass
@dataclass
class Mixture:
    densityModel: str = "ideal-incompressible-gas"   # or "const"
    densityValue: float = 1.2225                     # used only if densityModel == "const"
    species: List[Specie] = field(default_factory=list, repr=False)
    molarMasses: np.ndarray = field(init=False)

    def __post_init__(self):
        self.molarMasses = np.asarray([sp.molarMass for sp in self.species], dtype=float)

    def equivalentMolarMass(self, speciesFractions: np.ndarray) -> np.ndarray:
        """
        speciesFractions: (n_species, n_cells) mass fractions Y_i
        returns: (n_cells,) mixture molar mass [kg/mol]
        """
        # 1 / M_mix = Σ (Y_i / M_i)
        Y = np.asarray(speciesFractions, dtype=float)           # (n_species, n_cells)
        M_i = self.molarMasses[:, np.newaxis]                   # (n_species, 1)
        inv_M_mix = np.sum(Y / M_i, axis=0)                     # (n_cells,)
        M_mix = 1.0 / inv_M_mix                                 # (n_cells,)
        return M_mix

    def idealGasDensity(self, T: np.ndarray, speciesFractions: np.ndarray) -> np.ndarray:
        """
        T: (n_cells,) [K]
        speciesFractions: (n_species, n_cells) mass fractions Y_i
        returns: rho: (n_cells,) [kg/m^3]
        """
        T = np.asarray(T, dtype=float)
        M_mix = self.equivalentMolarMass(speciesFractions)      # (n_cells,)
        rho = P_REF * M_mix / (UNIVERSALGASCONSTANT * T)        # p M / (R T)
        return rho

@dataclass
class domainSetup:
    diameter: float
    massFlowRate : float
    inletMassFractions : np.ndarray = field(default_factory=list) 

@dataclass 
class Inlet:
    position: int = 0
    velocity: float = 50.0
    temperature : float = 700
    speciesMassFractions: List[float] = field(default_factory=list)

    def inletValues(self):
        """Returns list of np.arrays with with variables:
        massFlowrate, temperature, speciesMassFractions"""
        temperatureBC = np.array([self.temperature])
        velocityBC = np.array([self.velocity])
        specieFlowrateBC = np.array([self.speciesMassFractions])
        return velocityBC, temperatureBC, specieFlowrateBC
    
# @dataclass
# class BoundaryConditions:
#     counter: int = field(init=False, default=0, repr=False)

#     def addBoundary(self):
#         self.id = type(self).counter
#         type(self).counter += 1

@dataclass
class Zone:
    counter: ClassVar[int] = 0

    lenght: float = 0.005
    type: str = "null"   # "cooling", "reaction", "null"

    # heatSource : bool = False
    # massSource : bool = False

    # heatSource: float  # W/m3
    # massSource: List[str] = field(default_factory=list)

    def __post_init__(self):
        type(self).counter += 1
        self.id = type(self).counter

    def zoneAssign(self, heating : bool = False, reaction : bool = True):
        self.heatSource = heating
        self.massSource = reaction

    def zoneAssignHeating(self, heatValue : float):
        self.heatSourceValue = heatValue



class Mesh:
    def __init__(self, domain: domainSetup, zoneList: List[Zone], sizing: float = 0.005):
        self.sizing = sizing
        self.domain = domain
        self.meshZones = zoneList

    def meshCreate(self):
        # filter out zero-length zones
        zones = [z for z in self.meshZones if z.lenght > 0.0]
        if not zones:
            self.cell_centers    = np.array([], dtype=float)
            self.cell_sizes      = np.array([], dtype=float)
            self.cell_volumes    = np.array([], dtype=float)
            self.cell_zone_id    = np.array([], dtype=int)
            self.cell_zone_type  = np.array([], dtype=object)
            self.cell_heat_flag  = np.array([], dtype=bool)
            self.cell_mass_flag  = np.array([], dtype=bool)
            self.cell_heat_value = np.array([], dtype=float)
            self.n_cells = 0
            self.lenght  = 0.0
            return
        Lz         = np.array([z.lenght          for z in zones], dtype=float)   # (Nz,)
        zone_ids   = np.array([z.id              for z in zones], dtype=int)     # (Nz,)
        zone_types = np.array([z.type            for z in zones], dtype=object)  # (Nz,)
        heat_flags = np.array([z.heatSource      for z in zones], dtype=bool)    # (Nz,)
        mass_flags = np.array([z.massSource      for z in zones], dtype=bool)    # (Nz,)
        heat_vals  = np.array([z.heatSourceValue for z in zones], dtype=float)   # (Nz,)

        # number of cells and spacing per zone
        n_cells_zone = np.maximum(1, np.rint(Lz / self.sizing).astype(int))  # (Nz,)
        dz_zone      = Lz / n_cells_zone                                    # (Nz,)

        # total cells
        n_total = int(np.sum(n_cells_zone))

        # allocate per-cell arrays
        self.cell_centers    = np.empty(n_total, dtype=float)
        self.cell_sizes      = np.empty(n_total, dtype=float)
        self.cell_volumes    = np.empty(n_total, dtype=float)
        self.cell_zone_id    = np.empty(n_total, dtype=int)
        self.cell_zone_type  = np.empty(n_total, dtype=object)
        self.cell_heat_flag  = np.empty(n_total, dtype=bool)
        self.cell_mass_flag  = np.empty(n_total, dtype=bool)
        self.cell_heat_value = np.empty(n_total, dtype=float)

        # compute cumulative lengths and cell indices per zone
        z_starts = np.concatenate(([0.0], np.cumsum(Lz[:-1])))      # (Nz,)
        cell_start_idx = np.concatenate(([0], np.cumsum(n_cells_zone[:-1])))  # (Nz,)

        area = (self.domain.diameter**2) * PI / 4.0

        # loop only over zones, but each zone block is vectorized
        for i, z in enumerate(zones):
            nc  = n_cells_zone[i]
            dz_i = dz_zone[i]
            z0  = z_starts[i]
            start = cell_start_idx[i]
            end   = start + nc
            idx   = slice(start, end)

            # local indices 0..nc-1
            k = np.arange(nc, dtype=float)

            # cell centers and sizes in this zone
            self.cell_centers[idx] = z0 + (k + 0.5) * dz_i
            self.cell_sizes[idx]   = dz_i
            self.cell_volumes[idx] = dz_i * area

            # repeated zone-level properties to cells
            self.cell_zone_id[idx]   = zone_ids[i]
            self.cell_zone_type[idx] = zone_types[i]
            self.cell_heat_flag[idx] = heat_flags[i]
            self.cell_mass_flag[idx] = mass_flags[i]
            self.cell_heat_value[idx] = heat_vals[i]

        self.n_cells = n_total
        self.lenght  = float(z_starts[-1] + Lz[-1])
        
class scalarField:
    def __init__(self, variable: str, type: str = "specie"):
        self.variable = variable
        self.type = type
        
    def fieldInitialize(self, mesh : Mesh):
        self.cellField = np.zeros((mesh.n_cells))
        self.volumetricSources = np.zeros((mesh.n_cells))

class solver:
    def __init__(self, mesh: Mesh, mixture: Mixture, reaction: Reaction, specieFields: List[scalarField]):
        self.mesh = mesh
        self.mixture = mixture
        self.reaction = reaction
        self.density = np.full(mesh.n_cells, self.mixture.densityValue, dtype=float)
        self.massFlux = np.full(mesh.n_cells, self.mesh.domain.massFlowRate, dtype=float)
        # stack species fields into array (n_species, n_cells)
        # assuming each field.cellField is shape (1, n_cells)
        self.specieFields = np.vstack([f.cellField for f in specieFields])  # (n_species, n_cells)
        

        self.temperatureField = scalarField("temperature", "temperature").fieldInitialize(mesh= mesh)
        self.velocityField = scalarField("velocity", "velocity").fieldInitialize(mesh= mesh)
        
        self.massSources = np.zeros_like(self.specieFields)                 # (n_species, n_cells)
        self.heatSources = np.zeros_like(self.temperatureField)             # (n_cells,)

        self.reactionRates = np.zeros((2, mesh.n_cells), dtype=float)       # row 0: forward, row 1: backward

    def concentrationArray(self) -> np.ndarray:
        """Return concentrations C_i [mol/m^3] for all species and cells."""
        rho = self.density[np.newaxis, :]                               # scalar
        M = self.reaction.molarMasses                                 # (n_species,)
        # shape: (n_species, n_cells)
        C = rho * self.specieFields / M[:, np.newaxis]
        return C
    
    def update_density(self):
        # rebuild rho from current T and Y
        self.density = self.mixture.idealGasDensity(
            T=self.temperatureField,
            speciesFractions=self.specieFields
        )

    def sourcesEvaluation(self):
        C = self.concentrationArray()  # (n_species, n_cells)

        reactionMask = (self.mesh.cell_mass_flag == True)   # (n_cells,)
        heatMask     = (self.mesh.cell_heat_flag == True)   # (n_cells,)

        self.heatSources[heatMask] = np.asarray(self.mesh.cell_heat_value)[heatMask]

        rateForward, rateBackward = self.reaction.reactionRate(self.temperatureField, C)  # each (n_cells,)
        self.reactionRates[0, :] = rateForward
        self.reactionRates[1, :] = rateBackward

        massSources_all = self.reaction.reactionMassSource((rateForward, rateBackward))   # (n_species, n_cells)
        self.massSources[:, reactionMask] = massSources_all[:, reactionMask]

        heatReactionSource = self.reaction.reactionHeatSource(self.temperatureField, (rateForward, rateBackward))  # (n_cells,)
        self.heatSources[reactionMask] += heatReactionSource[reactionMask]
        print("self.massSources shape:", self.massSources.shape)
        print("massSources_all shape:", massSources_all.shape)
        print("reactionMask shape:", reactionMask.shape)

    def specieScalarEquation(self):
        
        
        
if __name__ == "__main__":
    # Polynomial cp: cp(T) = a*T^2 + b*T + c  [J/mol/K]
    # Just pick simple coefficients for A and B
    cp_A_coeffs = [1e-3, 0.1, 25.0]   # cp_A(T)
    cp_B_coeffs = [0.5e-3, 0.05, 20.0]  # cp_B(T)

    A = Specie(
        name="A",
        molarMass=0.028,
        heatCapacityModel="polynomial",
        enthalpyFormation=0.0,
        entropyFormation=0.0,
        heatCapacityCoefficients=cp_A_coeffs
    )

    B = Specie(
        name="B",
        molarMass=0.028,
        heatCapacityModel="polynomial",
        enthalpyFormation=-5.0e4,   # lower enthalpy → exothermic A->B
        entropyFormation=0.0,
        heatCapacityCoefficients=cp_B_coeffs
    )

    species = [A, B]

    # Reaction A -> B, irreversible
    reaction = Reaction(
        name="A_to_B",
        stochiometricCoefficients=np.array([-1.0, 1.0]),
        speciesExponent=np.array([1.0, 0.0]),
        reversedSpecieExponent=np.array([0.0, 0.0]),
        isReversible=False,
        ahrreniusPreExponent=1.0e2,
        ahrreniusActivationEnergy=80000.0,
        species=species
    )

    # Mesh: one reaction zone 0.5 m, 5 cells
    z = Zone(lenght=0.5, type="reaction")
    z.zoneAssign(heating=False, reaction=True)
    z.zoneAssignHeating(0.0)

    domain = domainSetup(
        diameter=0.1,
        massFlowRate=1.0,
        inletMassFractions=np.array([1.0, 0.0])
    )

    mesh = Mesh(domain=domain, zoneList=[z], sizing=0.1)
    mesh.meshCreate()

    print("n_cells:", mesh.n_cells)
    print("cell centers:", mesh.cell_centers)
    print("mass flags:", mesh.cell_mass_flag)

    # Species fields
    YA_field = scalarField("YA")
    YB_field = scalarField("YB")
    YA_field.fieldInitialize(mesh)
    YB_field.fieldInitialize(mesh)
    YA_field.cellField[...] = 1.0
    YB_field.cellField[...] = 0.0

    specieFields = [YA_field, YB_field]

    mixture = Mixture(densityModel="const", densityValue=1.0)

    sol = solver(mesh=mesh, mixture=mixture, reaction=reaction, specieFields=specieFields)
    sol.temperatureField[:] = 800.0  # K

    sol.sourcesEvaluation()

    print("ΔH_r(T):", reaction.enthalpyReactionChange(sol.temperatureField))
    print("Mass sources shape:", sol.massSources.shape)
    print("Heat sources shape:", sol.heatSources.shape)
    print("Mass source A (kg/m^3/s):", sol.massSources[0, :])
    print("Mass source B (kg/m^3/s):", sol.massSources[1, :])
    print("Heat source (J/m^3/s):", sol.heatSources)