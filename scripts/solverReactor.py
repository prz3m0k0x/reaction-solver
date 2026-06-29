import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from dataclasses import dataclass, field
import scipy.linalg
from typing import ClassVar, Dict, List
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

UNIVERSALGASCONSTANT = 8.31446261815324 #J/molK
T_REF   = 273.15
P_REF   = 101325.0 #Pa
PI      = 3.141592653589793

@dataclass
class Specie:
    """
    Represents a chemical species in a CFD/reaction simulation.

    Each species is automatically assigned a unique integer ID upon instantiation
    via a class-level counter. Thermodynamic properties (heat capacity and enthalpy)
    can be modeled as either a constant value or a polynomial function of temperature.

    Class Variables
    ---------------
    counter : int
        Class-level counter used to assign unique IDs to each species instance.

    Parameters
    ----------
    name : str
        Name of the chemical species (e.g., 'H2O', 'O2', 'N2').
    molarMass : float
        Molar mass of the species [kg/mol].
    heatCapacityModel : str, optional
        Model used for heat capacity calculation. Accepted values:
            - ``'const'``      : Constant heat capacity (default).
            - ``'polynomial'`` : NASA-style polynomial fit, evaluated as
                                 cp(T) = poly(T) * R, where R is the universal gas constant.
        Default is ``'const'``.
    enthalpyFormation : float, optional
        Standard enthalpy of formation at reference temperature T_REF [J/mol].
        Default is ``0.0``.
    entropyFormation : float, optional
        Standard entropy of formation at reference temperature T_REF [J/(mol·K)].
        Default is ``0.0``.
    heatCapacityValue : float, optional
        Constant heat capacity value [J/(kg·K)], used when ``heatCapacityModel='const'``.
        Default is ``900``.
    heatCapacityCoefficients : list[float], optional
        Polynomial coefficients for heat capacity [dimensionless, NASA form],
        used when ``heatCapacityModel='polynomial'``. Coefficients are passed to
        ``numpy.polynomial.Polynomial`` in ascending order (i.e., [a0, a1, a2, ...]).
        Default is an empty list.

    Attributes
    ----------
    id : int
        Unique integer identifier assigned automatically at instantiation.
    cp_poly : numpy.polynomial.Polynomial
        Polynomial object for cp(T) evaluation. Set only when
        ``heatCapacityModel='polynomial'``.
    H_poly : numpy.polynomial.Polynomial
        Antiderivative of ``cp_poly``, used for enthalpy integration.
        Set only when ``heatCapacityModel='polynomial'``.

    Methods
    -------
    heatCapacity(T)
        Compute heat capacity at temperature(s) T.

        Parameters
        ----------
        T : array-like
            Temperature(s) [K].

        Returns
        -------
        numpy.ndarray
            Heat capacity [J/(kg·K)] for ``'const'`` model, or
            [J/(mol·K)] for ``'polynomial'`` model (scaled by R).

    enthalpy(T)
        Compute total enthalpy (formation + sensible) at temperature(s) T.

        Parameters
        ----------
        T : array-like
            Temperature(s) [K].

        Returns
        -------
        numpy.ndarray
            Total enthalpy [J/kg] for ``'const'`` model, or
            [J/mol] for ``'polynomial'`` model.

        Notes
        -----
        Sensible enthalpy is computed relative to the global reference
        temperature ``T_REF``. For the polynomial model:

        .. math::

            H_{sensible}(T) = R \int_{T_{ref}}^{T} c_p(T') \, dT'

    Examples
    --------
    Constant heat capacity species:

    >>> aluminum = Specie(name='Al', molarMass=0.027, heatCapacityValue=900)
    >>> aluminum.heatCapacity(np.array([300, 500, 700]))
    array([900., 900., 900.])

    Polynomial heat capacity species (e.g., O2 with NASA coefficients):

    >>> o2 = Specie(
    ...     name='O2',
    ...     molarMass=0.032,
    ...     heatCapacityModel='polynomial',
    ...     enthalpyFormation=0.0,
    ...     heatCapacityCoefficients=[3.5, 0.0, -1e-5, 0.0, 2e-9]
    ... )
    >>> o2.heatCapacity(np.array([300.0, 1000.0]))
    """
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
    
    """Represents a single elementary or global chemical reaction in a simulation.

        This class encodes the full kinetic description of a reaction — stoichiometry,
        Arrhenius rate parameters, species ordering, and thermodynamic state functions —
        and exposes vectorized methods for computing forward/backward rates, mass sources,
        and heat release over spatially distributed fields (NumPy arrays).

        Reaction rates follow the Arrhenius law with optional reversibility via the
        equilibrium constant derived from Gibbs free energy:

        .. math::

            k_f(T) = A \exp\!\left(-\frac{E_a}{R T}\right)

        .. math::

            K_p(T) = \exp\!\left(-\frac{\Delta G}{R T}\right), \quad
            \Delta G = \Delta H_{rxn} - T \Delta S_{rxn}

        .. math::

            k_b(T) = \frac{k_f(T)}{K_c(T)}, \quad
            K_c = K_p \left(\frac{P_{ref}}{R T}\right)^{\Delta\nu}

        Parameters
        ----------
        name : str
            Human-readable identifier for the reaction (e.g., ``'H2 + 0.5 O2 -> H2O'``).
        stochiometricCoefficients : np.ndarray, optional
            Array of stoichiometric coefficients ``nu_i`` for each species, shape ``(n,)``.
            Reactants carry negative values; products carry positive values.
            Default is an empty array.
        speciesExponent : np.ndarray, optional
            Forward reaction rate exponents ``alpha_i`` for each species in the mass-action
            expression, shape ``(n,)``. Must satisfy ``len == len(stochiometricCoefficients)``.
            Default is an empty array.
        reversedSpecieExponent : np.ndarray, optional
            Backward reaction rate exponents ``beta_i`` for each species in the reverse
            mass-action expression, shape ``(n,)``. Must satisfy the same length constraint.
            Default is an empty array.
        isReversible : bool, optional
            If ``True``, the backward rate is computed from the equilibrium constant.
            If ``False``, the backward rate is zero. Default is ``True``.
        ahrreniusPreExponent : float, optional
            Arrhenius pre-exponential factor ``A`` [units depend on reaction order,
            typically mol·m⁻³·s⁻¹]. Default is ``1.0``.
        ahrreniusActivationEnergy : float, optional
            Arrhenius activation energy ``E_a`` [J/mol]. Default is ``0.0``.
        species : list[Specie], optional
            Ordered list of :class:`Specie` instances participating in the reaction.
            Must correspond 1-to-1 with the coefficient/exponent arrays.
            Excluded from ``repr``. Default is an empty list.

        Attributes
        ----------
        enthalpyChange : float
            Standard enthalpy change of reaction at reference temperature ``T_REF``
            [J/mol], computed as ``sum(nu_i * H°_f,i)``. Set in ``__post_init__``.
        entropyChange : float
            Standard entropy change of reaction at reference temperature ``T_REF``
            [J/(mol·K)], computed as ``sum(nu_i * S°_i)``. Set in ``__post_init__``.
        molarMasses : np.ndarray
            Array of molar masses [kg/mol] for each species in the reaction order,
            shape ``(n,)``. Set in ``__post_init__``.

        Raises
        ------
        ValueError
            If ``stochiometricCoefficients``, ``speciesExponent``, and
            ``reversedSpecieExponent`` do not all have the same length.

        Methods
        -------
        forwardRateConstant(T)
            Compute the Arrhenius forward rate constant k_f(T).

            Parameters
            ----------
            T : array-like
                Temperature field [K]. Values are clipped to a minimum of 1e-12.

            Returns
            -------
            np.ndarray
                Forward rate constant [units consistent with ``ahrreniusPreExponent``].

        equilibriumConstant(T)
            Compute the pressure-based equilibrium constant K_p(T) from Gibbs energy.

            Parameters
            ----------
            T : array-like
                Temperature field [K].

            Returns
            -------
            np.ndarray
                K_p [-], clipped to ``[1e-100, 1e100]`` for numerical stability.

        backwardRateConstant(T)
            Compute the backward rate constant k_b(T) = k_f / K_c.

            Converts K_p to the concentration-based K_c using the net mole change
            ``delta_nu = sum(nu_i)`` and the reference pressure ``P_REF``:

            .. math::

                K_c = K_p \left(\frac{P_{ref}}{R T}\right)^{\Delta\nu}

            Parameters
            ----------
            T : array-like
                Temperature field [K].

            Returns
            -------
            np.ndarray
                Backward rate constant. Returns a zero array if ``isReversible=False``.

        enthalpyReactionChange(T)
            Compute the temperature-dependent reaction enthalpy
            ``delta_H_rxn(T) = sum(nu_i * H_i(T))`` using species sensible + formation
            enthalpies from :meth:`Specie.enthalpy`.

            Parameters
            ----------
            T : array-like
                Temperature field [K].

            Returns
            -------
            np.ndarray
                Reaction enthalpy [J/mol] at each grid point, shape ``(N,)``.

        reactionRate(T, concentrations)
            Compute forward and backward reaction rates via mass-action kinetics.

            Forward rate:  ``r_f = k_f(T) * prod(C_i ** alpha_i)``

            Backward rate: ``r_b = k_b(T) * prod(C_i ** beta_i)``

            Concentrations are clipped to 1e-20 before applying power laws.

            Parameters
            ----------
            T : array-like
                Temperature field [K], shape ``(N,)``.
            concentrations : array-like
                Species concentration field [mol/m³], shape ``(n, N)``.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                ``(rateForward, rateBackward)``, each shape ``(N,)`` [mol/(m³·s)].
                ``rateBackward`` is all zeros if ``isReversible=False``.

        reactionMassSource(rates, rel_tol=1e-5, abs_tol=1e-12)
            Compute the per-species mass production/consumption rate [kg/(m³·s)].

            Near-equilibrium cells are zeroed out to prevent stiff source contributions.

            .. math::

                \dot{m}_i = \nu_i \cdot M_i \cdot (r_f - r_b)

            Parameters
            ----------
            rates : tuple[np.ndarray, np.ndarray]
                ``(rateForward, rateBackward)`` from :meth:`reactionRate`.
            rel_tol : float, optional
                Relative equilibrium tolerance. Default is ``1e-5``.
            abs_tol : float, optional
                Absolute equilibrium tolerance. Default is ``1e-12``.

            Returns
            -------
            np.ndarray
                Mass source array, shape ``(n, N)`` [kg/(m³·s)].
                Positive → net production; negative → net consumption.

        reactionHeatSource(T, rates, rel_tol=1e-5, abs_tol=1e-12)
            Compute the volumetric heat release rate [W/m³].

            .. math::

                \dot{q} = -(r_f - r_b) \cdot \Delta H_{rxn}(T)

            Positive values indicate exothermic release; negative values indicate
            endothermic absorption.

            Parameters
            ----------
            T : array-like
                Temperature field [K].
            rates : tuple[np.ndarray, np.ndarray]
                ``(rateForward, rateBackward)`` from :meth:`reactionRate`.
            rel_tol : float, optional
                Default is ``1e-5``.
            abs_tol : float, optional
                Default is ``1e-12``.

            Returns
            -------
            np.ndarray
                Volumetric heat source [W/m³], shape ``(N,)``.

        reactionHeatSourceDerivative(T, rates, rel_tol=1e-4, abs_tol=1e-9)
            Compute the analytical Jacobian dQ/dT [W/(m³·K)] for implicit
            energy equation linearization.

            Accounts for both Arrhenius temperature sensitivity and the Kirchhoff
            correction from the temperature-dependent reaction enthalpy:

            .. math::

                \frac{dQ}{dT} = -(\dot{r}_f' - \dot{r}_b') \Delta H_{rxn}
                                - (r_f - r_b) \Delta c_p

            where primes denote d/dT and
            ``delta_cp = sum(nu_i * cp_i(T))``.

            Parameters
            ----------
            T : array-like
                Temperature field [K], clipped to a minimum of 1e-3.
            rates : tuple[np.ndarray, np.ndarray]
                ``(rateForward, rateBackward)`` from :meth:`reactionRate`.
            rel_tol : float, optional
                Default is ``1e-4``.
            abs_tol : float, optional
                Default is ``1e-9``.

            Returns
            -------
            np.ndarray
                dQ/dT [W/(m³·K)], shape ``(N,)``. Near-equilibrium cells are zero.

        rateDerivativeConcentration(T, C, j)
            Compute the analytical Jacobian of forward and backward rates with respect
            to the concentration of species ``j``. Used to build the species Jacobian
            for implicit coupled solvers.

            .. math::

                \frac{\partial r_f}{\partial C_j} = k_f \cdot \alpha_j
                    \cdot \frac{\prod C_i^{\alpha_i}}{C_j}

            Parameters
            ----------
            T : array-like
                Temperature field [K].
            C : array-like
                Species concentration field [mol/m³], shape ``(n, N)``.
            j : int
                Species index with respect to which the derivative is taken.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                ``(dRf_dCj, dRb_dCj)``, each shape ``(N,)`` [m³/(mol·s)].
                ``dRb_dCj`` is zero if ``isReversible=False``.

        Notes
        -----
        **Equilibrium masking** — :meth:`_equilibriumMask` identifies cells where the
        net rate is negligible relative to the scale of the individual directional rates:

        .. math::

            |r_f - r_b| \leq \mathrm{rel\_tol} \cdot \max(|r_f|, |r_b|, 1) + \mathrm{abs\_tol}

        This prevents near-cancellation of large opposite-sign terms from generating
        spurious stiff source contributions during implicit solver iterations.

        **Concentration clipping** — all concentrations are clipped to 1e-20 before
        power-law evaluation to avoid singularities for non-integer exponents at zero
        concentration.

        **K_p → K_c conversion** — assumes ideal gas behaviour. The factor
        ``(P_REF / (R*T)) ** delta_nu`` converts the pressure-based equilibrium
        constant to a concentration-based one. ``K_c`` is additionally clipped to
        ``[1e-50, inf]`` before inverting to obtain ``k_b``.

        Examples
        --------
        Irreversible reaction A → B with constant-cp species:

        >>> A = Specie(name='A', molarMass=0.018, enthalpyFormation=-2.42e5)
        >>> B = Specie(name='B', molarMass=0.018, enthalpyFormation=-2.86e5)
        >>> rxn = Reaction(
        ...     name='A -> B',
        ...     stochiometricCoefficients=np.array([-1.0, 1.0]),
        ...     speciesExponent=np.array([1.0, 0.0]),
        ...     reversedSpecieExponent=np.array([0.0, 1.0]),
        ...     isReversible=False,
        ...     ahrreniusPreExponent=1e8,
        ...     ahrreniusActivationEnergy=50e3,
        ...     species=[A, B],
        ... )
        >>> T = np.array([500.0, 1000.0, 1500.0])
        >>> C = np.array([[1.0, 0.5, 0.2],   # [A]
        ...               [0.0, 0.1, 0.3]])   # [B]
        >>> rf, rb = rxn.reactionRate(T, C)
        >>> q = rxn.reactionHeatSource(T, (rf, rb))
        >>> mdot = rxn.reactionMassSource((rf, rb))  # shape (2, 3)
    """

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

    """
    Represents a multi-species gas mixture and provides thermophysical property
    evaluations over distributed field arrays.

    The mixture composition is described by mass fractions ``Y_i`` on a per-cell
    basis, enabling vectorized evaluation of density, molar mass, and heat capacity
    across the entire computational domain in a single NumPy call.

    Parameters
    ----------
    densityModel : str, optional
        Model used for density evaluation. Accepted values:
            - ``'ideal-incompressible-gas'`` : Density from the ideal gas law at
              reference pressure ``P_REF`` (default).
            - ``'const'``                    : Constant density equal to ``densityValue``.
        Default is ``'ideal-incompressible-gas'``.
    densityValue : float, optional
        Constant density [kg/m³], used only when ``densityModel='const'``.
        Default is ``1.2225``.
    species : list[Specie], optional
        Ordered list of :class:`Specie` instances forming the mixture.
        Order must be consistent with all ``speciesFractions`` arrays passed
        to the methods of this class. Excluded from ``repr``. Default is an empty list.

    Attributes
    ----------
    molarMasses : np.ndarray
        Array of species molar masses [kg/mol], shape ``(n_species,)``,
        assembled from ``species`` in ``__post_init__``.

    Methods
    -------
    equivalentMolarMass(speciesFractions)
        Compute the mixture molar mass from species mass fractions using the
        harmonic mean:

        .. math::

            M_{mix} = \left(\sum_i \frac{Y_i}{M_i}\right)^{-1}

        The inverse sum is clipped to ``1e-16`` to prevent division by zero
        in cells with near-zero or unphysical mass fractions.

        Parameters
        ----------
        speciesFractions : array-like
            Mass fractions ``Y_i``, shape ``(n_species, n_cells)``.
            Rows must be ordered consistently with ``self.species``.

        Returns
        -------
        np.ndarray
            Mixture molar mass [kg/mol], shape ``(n_cells,)``.

    idealGasDensity(T, speciesFractions)
        Compute the mixture density from the ideal gas law at reference pressure:

        .. math::

            \rho = \frac{P_{ref} \cdot M_{mix}(Y_i)}{R \cdot T}

        Parameters
        ----------
        T : array-like
            Temperature field [K], shape ``(n_cells,)``.
        speciesFractions : array-like
            Mass fractions ``Y_i``, shape ``(n_species, n_cells)``.

        Returns
        -------
        np.ndarray
            Mixture density [kg/m³], shape ``(n_cells,)``.

    mixtureHeatCapacity(T, speciesFractions)
        Compute the mass-weighted mixture specific heat capacity [J/(kg·K)]:

        .. math::

            c_{p,mix} = \sum_i Y_i \cdot \frac{c_{p,i}(T)}{M_i}

        Each species heat capacity is evaluated via :meth:`Specie.heatCapacity`
        and normalized by molar mass to convert from [J/(mol·K)] to [J/(kg·K)].

        Parameters
        ----------
        T : array-like
            Temperature field [K], shape ``(n_cells,)``.
        speciesFractions : array-like
            Mass fractions ``Y_i``, shape ``(n_species, n_cells)``.

        Returns
        -------
        np.ndarray
            Mixture heat capacity [J/(kg·K)], shape ``(n_cells,)``.

    Notes
    -----
    The ``'ideal-incompressible-gas'`` model computes density at constant
    reference pressure ``P_REF`` — this is suitable for low-Mach-number
    reacting flow solvers where pressure variations are small relative to the
    thermodynamic pressure. It is *not* appropriate for compressible or
    high-Mach-number flows.

    The ``densityModel`` field is stored but its branching logic (i.e., switching
    between ``'const'`` and ``'ideal-incompressible-gas'``) must be implemented
    at the solver level. No internal dispatch is currently performed by this class.

    Examples
    --------
    >>> H2O = Specie(name='H2O', molarMass=0.018)
    >>> N2  = Specie(name='N2',  molarMass=0.028)
    >>> mix = Mixture(species=[H2O, N2])
    >>> Y   = np.array([[0.1, 0.2],   # Y_H2O at two cells
    ...                 [0.9, 0.8]])   # Y_N2
    >>> T   = np.array([800.0, 1000.0])
    >>> mix.equivalentMolarMass(Y)
    array([0.02561..., 0.02432...])
    >>> mix.idealGasDensity(T, Y)
    array([0.384..., 0.295...])
    """
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
    """
    Holds geometric parameters describing the computational domain.

    Currently encodes a single cylindrical (pipe/reactor) domain defined by its
    diameter. Inlet mass fractions can optionally be stored here as a reference
    composition for domain initialization.

    Parameters
    ----------
    diameter : float
        Inner diameter of the domain [m]. Used to compute the cross-sectional
        area for inlet mass flow rate calculations.
    inletMassFractions : np.ndarray, optional
        Reference inlet mass fractions ``Y_i`` [-], shape ``(n_species,)``.
        Default is an empty array.

    Notes
    -----
    The cross-sectional area of a circular duct is computed as:

    .. math::

        A = \frac{\pi d^2}{4}

    This class is intended as a lightweight configuration container.
    No validation of ``inletMassFractions`` consistency with the active
    :class:`Mixture` is performed here.
    """
    diameter: float
    inletMassFractions : np.ndarray = field(default_factory=lambda: np.array([])) 

@dataclass
class Inlet:
    """
    Defines the boundary condition at the domain inlet.

    Computes the primitive inlet variables — mass flow rate, temperature,
    and species composition — in the form expected by the solver's boundary
    condition assembly, given a :class:`Mixture` and a :class:`domainSetup`.

    Parameters
    ----------
    position : int, optional
        Cell or face index identifying the inlet location in the mesh.
        Default is ``0``.
    velocity : float, optional
        Bulk inlet velocity [m/s]. Default is ``50.0``.
    temperature : float, optional
        Inlet temperature [K]. Default is ``700.0``.
    speciesMassFractions : list[float], optional
        Ordered list of species mass fractions ``Y_i`` [-] at the inlet.
        Must be consistent with the species ordering of the :class:`Mixture`
        passed to :meth:`inletValues`. Default is an empty list.

    Methods
    -------
    inletValues(mixture, domain)
        Compute the inlet boundary conditions from the prescribed primitive variables.

        The inlet density is evaluated from the ideal gas law via
        :meth:`Mixture.idealGasDensity`. The mass flow rate follows from:

        .. math::

            \dot{m} = \rho \cdot A \cdot u, \quad A = \frac{\pi d^2}{4}

        Parameters
        ----------
        mixture : Mixture
            The active mixture object used to evaluate inlet density.
        domain : domainSetup
            Domain geometry providing the duct diameter for area calculation.

        Returns
        -------
        massFlowrate : float
            Inlet mass flow rate [kg/s].
        temperatureBC : float
            Inlet temperature boundary condition [K].
        specieFrac : np.ndarray
            Inlet species mass fractions [-], shape ``(n_species,)``.

    Notes
    -----
    The density used for mass flow rate calculation is evaluated at the single
    inlet temperature and composition, treating the inlet as a uniform plug-flow
    boundary. Spatial non-uniformities (e.g., radial profiles) are not supported
    by this class.

    Examples
    --------
    >>> mix = Mixture(species=[H2O, N2])
    >>> domain = domainSetup(diameter=0.05)
    >>> inlet = Inlet(
    ...     velocity=30.0,
    ...     temperature=800.0,
    ...     speciesMassFractions=[0.15, 0.85],
    ... )
    >>> mdot, T_bc, Y_bc = inlet.inletValues(mix, domain)
    """
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

@dataclass
class Zone:
    """
    Represents a single spatial zone (control volume segment) in the
    computational domain.

    Each zone is automatically assigned a unique integer ID upon instantiation
    via a class-level counter. Zones can act as sources of heat and/or species
    mass, controlled through the assignment methods.

    Class Variables
    ---------------
    counter : int
        Class-level counter used to assign unique IDs to each zone instance.

    Parameters
    ----------
    length : float, optional
        Physical length of the zone along the axial domain direction [m].
        Default is ``0.005``.
    zoneType : str, optional
        Zone type label (e.g., ``'fluid'``, ``'wall'``, ``'reaction'``).
        Used for zone classification at the solver level. Default is ``'null'``.

    Attributes
    ----------
    id : int
        Unique integer identifier, assigned automatically in ``__post_init__``.
    heatSource : bool
        Flag indicating whether this zone contributes a heat source term
        to the energy equation. Set via :meth:`zoneAssign` or
        :meth:`zoneAssignHeating`. Default is ``False``.
    massSource : bool
        Flag indicating whether this zone contributes species mass source
        terms (i.e., chemical reactions are active). Set via :meth:`zoneAssign`.
        Default is ``False``.
    heatSourceValue : float
        Prescribed heat input for this zone [W] (total power) or [W/m³]
        (volumetric heat generation rate), depending on the calling context.
        Set via :meth:`zoneAssignHeating`. Default is ``0.0``.

    Methods
    -------
    zoneAssign(heating=False, reaction=True)
        Configure whether this zone has active heat and/or mass source terms.

        Parameters
        ----------
        heating : bool, optional
            If ``True``, enables the heat source flag (``heatSource=True``).
            Default is ``False``.
        reaction : bool, optional
            If ``True``, enables the mass source flag (``massSource=True``),
            activating chemical reaction source terms in this zone.
            Default is ``True``.

    zoneAssignHeating(heatValue)
        Assign a prescribed heat source magnitude to this zone and enable
        the heat source flag.

        Two interpretations depending on the solver configuration:

        - **Case A** — volumetric heat generation rate [W/m³], applied
          uniformly across the zone volume.
        - **Case B** — total heat power [W], integrated over the zone
          and distributed by the solver.

        Parameters
        ----------
        heatValue : float
            Heat source magnitude [W] or [W/m³]. Sets ``heatSourceValue``
            and activates ``heatSource=True``.

    Notes
    -----
    The ``type`` attribute shadows the Python built-in ``type``. Within
    ``__post_init__``, ``type(self).counter`` uses the built-in explicitly
    to access the class — this works correctly but requires care if ``type``
    is ever used in subclass logic.

    Zone IDs start from ``1`` (the counter is incremented before assignment).
    To reset the counter between simulation setups (e.g., in unit tests),
    call ``Zone.counter = 0`` manually before instantiating new zones.

    Examples
    --------
    A reaction-active zone with external heating:

    >>> z = Zone(length=0.01, type='fluid')
    >>> z.zoneAssign(heating=False, reaction=True)
    >>> z.massSource
    True

    A purely heated zone with no reaction (electric heater segment):

    >>> z_heat = Zone(length=0.005, type='heater')
    >>> z_heat.zoneAssignHeating(500.0)   # 500 W total power
    >>> z_heat.heatSource, z_heat.heatSourceValue
    (True, 500.0)
    """
    counter : ClassVar[int] = 0
    length  : float         = 0.005
    zoneType    : str           = "null"

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
    """
    Constructs a one-dimensional finite volume mesh from an ordered list of
    :class:`Zone` objects.

    Each zone is discretized into uniform cells of approximately ``sizing``
    length. The number of cells per zone is rounded to the nearest integer
    (minimum 1), and the actual cell size is adjusted to exactly fill the
    zone length. All cell-level arrays are contiguous and zone-ordered,
    making them directly addressable by the solver's field arrays.

    Parameters
    ----------
    domain : domainSetup
        Domain geometry object providing the duct diameter, used to compute
        the cross-sectional area for cell volume calculation.
    zoneList : list[Zone]
        Ordered list of :class:`Zone` instances defining the spatial
        decomposition of the domain. Zones with ``length <= 0.0`` are
        silently excluded during :meth:`meshCreate`.
    sizing : float, optional
        Target cell size [m] used to determine the number of cells per zone:

        .. math::

            n_{cells,z} = \max\!\left(1,\ \mathrm{round}\!\left(
            \frac{L_z}{\Delta x_{target}}\right)\right)

        The actual cell size is ``dz = L_z / n_cells_z``, which may differ
        slightly from ``sizing`` due to rounding. Default is ``0.005``.

    Attributes
    ----------
    sizing : float
        Target cell size [m] as provided at construction.
    domain : domainSetup
        Reference to the domain geometry object.
    meshZones : list[Zone]
        The full zone list as provided (including zero-length zones).
    cell_centers : np.ndarray
        Axial position of each cell centre [m], shape ``(n_cells,)``.
        Computed as ``z_start + (k + 0.5) * dz`` for cell index ``k``
        within a zone starting at ``z_start``.
    cell_sizes : np.ndarray
        Axial cell width ``dz`` [m], shape ``(n_cells,)``. Uniform within
        each zone, but may differ between zones.
    cell_volumes : np.ndarray
        Cell volume [m³], shape ``(n_cells,)``. Computed as:

        .. math::

            V_{cell} = \Delta z \cdot A, \quad A = \frac{\pi d^2}{4}

    cell_zone_id : np.ndarray of int
        Zone ID (from :attr:`Zone.id`) for each cell, shape ``(n_cells,)``.
    cell_zone_type : np.ndarray of object
        Zone type string (from :attr:`Zone.zoneType`) for each cell,
        shape ``(n_cells,)``.
    cell_heat_flag : np.ndarray of bool
        Per-cell flag indicating whether a heat source term is active,
        shape ``(n_cells,)``. Mirrors :attr:`Zone.heatSource`.
    cell_mass_flag : np.ndarray of bool
        Per-cell flag indicating whether reaction mass source terms are
        active, shape ``(n_cells,)``. Mirrors :attr:`Zone.massSource`.
    cell_heat_value : np.ndarray of float
        Prescribed heat source magnitude [W] or [W/m³] for each cell,
        shape ``(n_cells,)``. Mirrors :attr:`Zone.heatSourceValue`.
    n_cells : int
        Total number of cells in the mesh. Set to ``0`` if all zones have
        zero length.
    length : float
        Total axial length of the mesh [m], equal to the sum of all
        active zone lengths.

    Methods
    -------
    meshCreate()
        Discretize all active zones and populate the cell-level arrays.

        Zones with ``length <= 0.0`` are excluded. If no active zones
        remain, all arrays are set to empty and ``n_cells = 0``.

        The discretization proceeds zone by zone in order:

        1. Compute ``n_cells_z = max(1, round(L_z / sizing))`` per zone.
        2. Compute the exact cell size ``dz = L_z / n_cells_z``.
        3. Compute cell centres as ``z0 + (k + 0.5) * dz`` for
           ``k = 0, ..., n_cells_z - 1``, where ``z0`` is the cumulative
           zone start position.
        4. Broadcast all zone-level scalar properties (ID, type, flags,
           heat value) uniformly across the zone's cells.

        Must be called explicitly after construction before accessing any
        cell-level attributes.

    Notes
    -----
    **Uniform spacing within zones** — each zone uses a single cell size
    ``dz = L_z / n_cells_z``. Non-uniform or graded meshes (e.g., wall
    refinement) are not supported by this class.

    **Cell volume assumes a circular cross-section** — the area is computed
    as ``pi * d^2 / 4`` from ``domain.diameter``. For non-circular ducts,
    the area calculation must be overridden at the solver level.

    **`cell_heat_value` units** — the value is copied verbatim from
    :attr:`Zone.heatSourceValue`, which can represent either a total power
    [W] or a volumetric rate [W/m³] depending on how the zone was configured
    (see :meth:`Zone.zoneAssignHeating`). The solver is responsible for
    dividing by ``cell_volumes`` if total power is used.

    **Zero-length zones** — zones with ``length <= 0.0`` are silently
    skipped in :meth:`meshCreate` but remain in :attr:`meshZones`. This
    allows zones to be conditionally disabled without removing them from
    the configuration list.

    Examples
    --------
    >>> z1 = Zone(length=0.05, zoneType='fluid')
    >>> z1.zoneAssign(heating=False, reaction=True)
    >>> z2 = Zone(length=0.02, zoneType='heater')
    >>> z2.zoneAssignHeating(1000.0)
    >>>
    >>> domain = domainSetup(diameter=0.05)
    >>> mesh = Mesh(domain=domain, zoneList=[z1, z2], sizing=0.005)
    >>> mesh.meshCreate()
    >>>
    >>> mesh.n_cells         # 50 + 20 = 70 cells (at 0.005 m each)
    14
    >>> mesh.cell_centers[:3]
    array([0.0025, 0.0075, 0.0125])
    >>> mesh.cell_heat_flag[-1]
    True
    """
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

class scalarField:
    """
    Represents a scalar field variable defined over the cells of a
    :class:`Mesh`.

    A thin wrapper that associates a named physical quantity with a
    1D NumPy array aligned to the mesh cell layout. Intended to be used
    for species mass fractions, temperature, pressure, or any other
    cell-centred scalar transported by the solver.

    Parameters
    ----------
    variable : str
        Name of the physical quantity represented by this field
        (e.g., ``'T'``, ``'p'``, ``'Y_H2O'``). Used for identification
        and post-processing labelling.
    field_type : str, optional
        Semantic category of the field. Accepted values (convention,
        not enforced):
            - ``'specie'``      : Species mass fraction field (default).
            - ``'temperature'`` : Temperature field.
            - ``'pressure'``    : Pressure field.
        Default is ``'specie'``.

    Attributes
    ----------
    variable : str
        Field name as provided at construction.
    field_type : str
        Field type label as provided at construction.
    cellField : np.ndarray or None
        Cell-centred scalar values, shape ``(n_cells,)``, aligned with
        :attr:`Mesh.cell_centers`. Initialized to ``None`` at construction;
        set to a zero array by :meth:`fieldInitialize`.

    Methods
    -------
    fieldInitialize(mesh)
        Allocate and zero-initialize the cell field array.

        Creates a zero-filled array with the same shape and dtype as
        ``mesh.cell_centers`` (``float64``), and assigns it to
        ``self.cellField``.

        Parameters
        ----------
        mesh : Mesh
            The active mesh object. The field length is inferred from
            ``mesh.cell_centers``.

        Notes
        -----
        Any previously stored values in ``cellField`` are overwritten.
        Re-calling this method resets the field to zero, which can be
        used to reinitialize between solver restarts or parametric runs.

    Notes
    -----
    The class name uses ``camelCase`` rather than the Python convention
    of ``PascalCase`` for classes (``ScalarField``). Renaming is advisable
    for consistency with the rest of the codebase.

    ``cellField`` is ``None`` until :meth:`fieldInitialize` is called.
    Any solver code accessing ``cellField`` before initialization will
    encounter ``None`` rather than an array, which may produce silent
    errors in downstream NumPy operations. A guard check is advisable:

    .. code-block:: python

        if self.cellField is None:
            raise RuntimeError(
                f"Field '{self.variable}' has not been initialized. "
                "Call fieldInitialize(mesh) first."
            )

    Examples
    --------
    >>> mesh = Mesh(domain=domain, zoneList=[z1, z2], sizing=0.005)
    >>> mesh.meshCreate()
    >>>
    >>> T_field = scalarField(variable='T', field_type='temperature')
    >>> T_field.fieldInitialize(mesh)
    >>> T_field.cellField
    array([0., 0., 0., ..., 0.])
    >>>
    >>> Y_H2O = scalarField(variable='Y_H2O', field_type='specie')
    >>> Y_H2O.fieldInitialize(mesh)
    >>> Y_H2O.cellField[:] = 0.1   # uniform initial condition
    """
    def __init__(self, variable: str, field_type: str = "specie"):
        self.variable   = variable
        self.field_type = field_type
        self.cellField  = None  
        
    def fieldInitialize(self, mesh):
        self.cellField  = np.zeros_like(mesh.cell_centers)

class solver:
    """
    Finite volume solver for steady-state 1D reacting flow in a duct.

    Solves the coupled species mass fraction and energy (temperature) transport
    equations on a :class:`Mesh` with a single :class:`Reaction`, using a
    sequential (segregated) iterative algorithm. Convection is treated with a
    first-order upwind scheme. Source terms from chemical reactions are linearized
    and under-relaxed for stability.

    The governing equations per cell ``i`` (index increasing downstream) are:

    **Species** (mass fraction ``Y_k``):

    .. math::

        \dot{m} Y_{k,i} - \dot{m} Y_{k,i-1} = \dot{\omega}_{k,i} V_i

    **Energy** (temperature ``T``):

    .. math::

        \dot{m} c_{p,i} T_i - \dot{m} c_{p,i-1} T_{i-1} = \dot{Q}_i V_i

    where :math:`\dot{m}` is the inlet mass flow rate [kg/s], :math:`V_i` is
    the cell volume [m³], :math:`\dot{\omega}_{k,i}` is the species mass source
    [kg/(m³·s)], and :math:`\dot{Q}_i` is the volumetric heat source [W/m³].

    Parameters
    ----------
    mesh : Mesh
        Discretized domain providing cell geometry, zone flags, and heat values.
    mixture : Mixture
        Mixture object providing density and heat capacity evaluations.
    reaction : Reaction
        Single reaction object providing rate constants, source terms,
        and their analytical Jacobians.
    specieFields : list[scalarField]
        Ordered list of initialized :class:`scalarField` instances, one per
        species. Order must match ``mixture.species`` and ``reaction.species``.
        Their ``cellField`` arrays are stacked into a ``(n_species, n_cells)``
        matrix at construction.
    inlet : Inlet
        Inlet boundary condition object providing mass flow rate, temperature,
        and species mass fractions.

    Attributes
    ----------
    mesh : Mesh
        Reference to the active mesh.
    mixture : Mixture
        Reference to the mixture object.
    reaction : Reaction
        Reference to the reaction object.
    inlet : Inlet
        Reference to the inlet boundary object.
    massFlux : float
        Inlet mass flow rate [kg/s], computed once from ``inlet.inletValues``
        at construction and held constant throughout the simulation.
    specieFields : np.ndarray
        Species mass fraction field, shape ``(n_species, n_cells)``.
        Assembled by vertically stacking the ``cellField`` arrays from the
        input ``specieFields`` list.
    temperatureField : scalarField
        Temperature field [K], initialized uniformly to ``inlet.temperature``.
    velocityField : scalarField
        Axial velocity field [m/s], initialized uniformly to ``inlet.velocity``.
        Updated each iteration from ``massFlux / (density * area)``.
    density : np.ndarray
        Mixture density field [kg/m³], shape ``(n_cells,)``. Updated via
        :meth:`update_density` after each composition or temperature change.
    massSources : np.ndarray
        Per-species volumetric mass source [kg/(m³·s)], shape ``(n_species, n_cells)``.
        Populated by :meth:`sourcesEvaluation`.
    massSourcesDerivative : np.ndarray
        Linearization coefficient ``dω_k/dY_k`` [kg/(m³·s)] per species per cell,
        shape ``(n_species, n_cells)``. Used for implicit source linearization.
    heatSources : np.ndarray
        Total volumetric heat source [W/m³], shape ``(n_cells,)``. Sum of
        prescribed zone heating and reaction heat release.
    heatSourcesDerivative : np.ndarray
        Linearization coefficient ``dQ/dT`` [W/(m³·K)], shape ``(n_cells,)``.
        Used for implicit energy equation linearization.
    heatReactionSources : np.ndarray
        Reaction-only contribution to ``heatSources`` [W/m³], shape ``(n_cells,)``.
        Stored separately to enable independent under-relaxation.
    heatReactionSourcesDerivative : np.ndarray
        Reaction-only contribution to ``heatSourcesDerivative`` [W/(m³·K)],
        shape ``(n_cells,)``.
    heatResidual : np.ndarray
        Energy equation residual vector [W], shape ``(n_cells,)``.
        Computed by :meth:`heatEquation`.
    specieResidual : np.ndarray
        Species equation residual matrix [kg/s], shape ``(n_species, n_cells)``.
        Computed by :meth:`specieScalarEquation`.
    reactionRates : np.ndarray
        Forward and backward reaction rates [mol/(m³·s)], shape ``(2, n_cells)``.
        Row 0 is forward; row 1 is backward. Updated by :meth:`sourcesEvaluation`.

    Methods
    -------
    concentrationArray()
        Compute molar concentrations from current density and mass fractions:

        .. math::

            C_k = \frac{\rho \cdot Y_k}{M_k} \quad \text{[mol/m³]}

        Returns
        -------
        np.ndarray
            Concentration array [mol/m³], shape ``(n_species, n_cells)``.

        Raises
        ------
        ValueError
            If ``specieFields`` is not 2D, or if species/cell counts are
            inconsistent with ``mixture.molarMasses`` or ``density``.

    update_density()
        Recompute ``self.density`` from the current temperature and species
        fields via :meth:`Mixture.idealGasDensity`. Must be called after any
        update to ``temperatureField`` or ``specieFields``.

    sourcesEvaluation(omega_heat=0.05, omega_mass=0.15, eq_rel_tol=1e-2, eq_abs_tol=1e-4)
        Evaluate and under-relax all reaction source terms for the current
        iteration.

        Procedure:

        1. Compute molar concentrations via :meth:`concentrationArray`.
        2. Evaluate forward and backward reaction rates via
           :meth:`Reaction.reactionRate`.
        3. Identify near-equilibrium cells using the tolerance mask
           (see :class:`Reaction` Notes). Cells outside reaction zones
           (``cell_mass_flag=False``) are also masked out.
        4. Compute species mass sources :math:`\dot{\omega}_k` and their
           Jacobians :math:`d\dot{\omega}_k/dY_k` via
           :meth:`Reaction.reactionMassSource` and
           :meth:`Reaction.rateDerivativeConcentration`, applying the
           chain rule :math:`dC_k/dY_k = \rho / M_k`.
        5. Compute reaction heat source :math:`\dot{Q}_{rxn}` and its
           Jacobian :math:`dQ/dT` via :meth:`Reaction.reactionHeatSource`
           and :meth:`Reaction.reactionHeatSourceDerivative`.
        6. Add prescribed zone heat values to ``heatSources`` where
           ``cell_heat_flag=True``.
        7. Apply under-relaxation to reaction sources:

           .. math::

               S^{new} = (1 - \omega) S^{old} + \omega S^{eval}

        Parameters
        ----------
        omega_heat : float, optional
            Under-relaxation factor for reaction heat sources and their
            derivatives. Default is ``0.05``.
        omega_mass : float, optional
            Under-relaxation factor for species mass sources and their
            derivatives. Default is ``0.15``.
        eq_rel_tol : float, optional
            Relative tolerance for near-equilibrium cell detection.
            Default is ``1e-2``.
        eq_abs_tol : float, optional
            Absolute tolerance for near-equilibrium cell detection.
            Default is ``1e-4``.

    matrixSpecieEquationAssembly(specieIndex)
        Assemble the linear system ``A Y_k = b`` for species ``k`` using
        first-order upwind convection with implicit source linearization.

        The source term is split as:

        .. math::

            S = S_U + S_P \cdot Y_k, \quad S_P = \min(dS/dY_k,\ 0)

        ``S_P`` is always non-positive (deferred to the matrix diagonal)
        to preserve diagonal dominance. ``S_U`` is treated explicitly on
        the right-hand side.

        Parameters
        ----------
        specieIndex : int
            Index of the species to assemble (row index into ``specieFields``).

        Returns
        -------
        A : np.ndarray
            System matrix, shape ``(n_cells, n_cells)``.
        b : np.ndarray
            Right-hand side vector, shape ``(n_cells,)``.

    matrixTemperatureEquationAssembly()
        Assemble the linear system ``A T = b`` for the energy equation using
        first-order upwind convection and implicit heat source linearization.

        The heat source is split analogously to the species equation:

        .. math::

            Q = S_U + S_P \cdot T, \quad S_P = \min(dQ/dT,\ 0)

        The cell heat capacity ``cp_mix`` is evaluated at the current
        temperature and composition, then frozen for the linear solve.

        Returns
        -------
        A : np.ndarray
            System matrix, shape ``(n_cells, n_cells)``.
        b : np.ndarray
            Right-hand side vector, shape ``(n_cells,)``.

    specieScalarEquation()
        Evaluate the species equation residual for the current field state
        and store it in ``self.specieResidual``. Used for monitoring
        convergence without assembling the full matrix.

    heatEquation()
        Evaluate the energy equation residual for the current field state
        and store it in ``self.heatResidual``. Used for convergence monitoring.

    initializeCase()
        Set all field arrays to the inlet boundary values and recompute density.

        Assigns ``inlet.speciesMassFractions`` uniformly across all cells and
        sets the temperature field to ``inlet.temperature``. Intended to be
        called once before :meth:`steadyState` to establish a physically
        consistent initial condition.

    steadyState(max_iter, relaxationFactorSpecie, relaxationFactorTemperature, convergenceCriteria)
        Run the segregated iterative solver to steady state.

        Each iteration proceeds as:

        1. Solve species equations sequentially for each species index.
        2. Clip species fields to ``[0, 1]`` and re-normalize to enforce
           :math:`\sum_k Y_k = 1`.
        3. Update density and velocity.
        4. Evaluate source terms via :meth:`sourcesEvaluation`.
        5. Solve the energy equation.
        6. Clip temperature to ``[200, 2000]`` K.
        7. Apply under-relaxation to both species and temperature:

           .. math::

               \phi^{n+1} = \phi^n + \alpha (\phi^* - \phi^n)

        8. Check convergence using scaled norms:

           .. math::

               \delta Y_{scaled} = \frac{\max|\Delta Y|}{\dot{m}}, \quad
               \delta T_{scaled} = \frac{\max|\Delta T|}{\bar{T}}

        Prints iteration diagnostics every 10 steps.

        Parameters
        ----------
        max_iter : int
            Maximum number of outer iterations.
        relaxationFactorSpecie : float
            Under-relaxation factor ``alpha`` for species fields. Typical
            range: ``0.3``–``0.8``.
        relaxationFactorTemperature : float
            Under-relaxation factor ``alpha`` for the temperature field.
            Typical range: ``0.2``–``0.5``.
        convergenceCriteria : float
            Absolute convergence threshold. Iteration stops when
            ``max(max|ΔY|, max|ΔT|) < convergenceCriteria``.

        Returns
        -------
        np.ndarray
            Outlet species mass fractions ``Y_k[-1]``, shape ``(n_species,)``.

    Notes
    -----
    **Segregated algorithm** — species and temperature equations are solved
    sequentially rather than simultaneously. This avoids assembling a
    large coupled block system at the cost of requiring more outer iterations
    and careful under-relaxation to maintain stability.

    **Mass flux is constant** — ``self.massFlux`` is computed once at
    construction from the inlet conditions. Density and velocity are updated
    each iteration, but the mass flow rate is not recomputed. This is
    consistent with the incompressible-flow assumption at fixed ``P_REF``.

    **Dense matrix solver** — ``scipy.linalg.solve`` is used to solve the
    tridiagonal systems in :meth:`matrixSpecieEquationAssembly` and
    :meth:`matrixTemperatureEquationAssembly`. For large meshes (``n_cells``
    > ~10³), replacing these with ``scipy.sparse.linalg.spsolve`` on CSR
    matrices would give a significant performance improvement, as the
    current dense assembly allocates an ``(n_cells, n_cells)`` matrix
    even though only two diagonals are populated.

    **Source term under-relaxation** — very low default values (``omega_heat=0.05``,
    ``omega_mass=0.15``) indicate that reaction sources are expected to
    change rapidly between iterations. If convergence is slow, increasing
    these factors (up to ~0.5) may accelerate the solution once the
    fields are near equilibrium.

    Examples
    --------
    Typical solver setup and execution:

    >>> mesh.meshCreate()
    >>> Y_H2O = scalarField('Y_H2O', 'specie')
    >>> Y_N2  = scalarField('Y_N2',  'specie')
    >>> Y_H2O.fieldInitialize(mesh)
    >>> Y_N2.fieldInitialize(mesh)
    >>>
    >>> slv = solver(
    ...     mesh=mesh,
    ...     mixture=mix,
    ...     reaction=rxn,
    ...     specieFields=[Y_H2O, Y_N2],
    ...     inlet=inlet,
    ... )
    >>> slv.initializeCase()
    >>> outlet_Y = slv.steadyState(
    ...     max_iter=500,
    ...     relaxationFactorSpecie=0.5,
    ...     relaxationFactorTemperature=0.3,
    ...     convergenceCriteria=1e-6,
    ... )
    """

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
    def sourcesEvaluation(self, omega_heat=0.05, omega_mass=0.15,
                        eq_rel_tol=1e-2, eq_abs_tol=1e-4):
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
        rateForward, rateBackward = self.reaction.reactionRate(T, C)
        self.reactionRates[0, :] = rateForward
        self.reactionRates[1, :] = rateBackward

        net_rate = rateForward - rateBackward
        rate_scale = np.maximum(np.maximum(np.abs(rateForward), np.abs(rateBackward)), 1.0)
        near_eq = np.abs(net_rate) <= (eq_rel_tol * rate_scale + eq_abs_tol)

        near_eq = near_eq & reactionMask

        massSources_all = self.reaction.reactionMassSource((rateForward, rateBackward))
        massSources_all[:, near_eq] = 0.0
        self.massSources[:, reactionMask] = massSources_all[:, reactionMask]

        M_species = self.reaction.molarMasses
        stoich = self.reaction.stochiometricCoefficients

        for k in range(self.specieFields.shape[0]):
            dRf_dCk, dRb_dCk = self.reaction.rateDerivativeConcentration(T, C, j=k)
            dOmega_dC = stoich[k] * M_species[k] * (dRf_dCk - dRb_dCk)
            dOmega_dC[near_eq] = 0.0

            dCk_dYk = rho / M_species[k]
            dOmega_dYk = dOmega_dC * dCk_dYk
            self.massSourcesDerivative[k, reactionMask] = dOmega_dYk[reactionMask]

        heatReactionSource = self.reaction.reactionHeatSource(T, (rateForward, rateBackward))
        dQ_dT = self.reaction.reactionHeatSourceDerivative(T, (rateForward, rateBackward))

        heatReactionSource[near_eq] = 0.0
        dQ_dT[near_eq] = 0.0

        self.heatReactionSources[reactionMask] = heatReactionSource[reactionMask]
        self.heatReactionSourcesDerivative[reactionMask] = dQ_dT[reactionMask]

        self.heatReactionSources = (
            (1.0 - omega_heat) * heat_rxn_old + omega_heat * self.heatReactionSources
        )
        self.heatReactionSourcesDerivative = (
            (1.0 - omega_heat) * heatd_rxn_old + omega_heat * self.heatReactionSourcesDerivative
        )

        self.heatSources += self.heatReactionSources
        self.heatSourcesDerivative[:] = self.heatReactionSourcesDerivative

        self.massSources = (1.0 - omega_mass) * mass_old + omega_mass * self.massSources
        self.massSourcesDerivative = (1.0 - omega_mass) * massd_old + omega_mass * self.massSourcesDerivative

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