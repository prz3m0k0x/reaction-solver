# solverReactor.py — Documentation

## Overview

`solverReactor.py` implements a 1D steady-state plug-flow reactor solver for compressible reacting gas mixtures. It supports **multiple simultaneous chemical reactions**, temperature-dependent thermodynamics (constant or NASA-polynomial heat capacities), reversible reactions via Gibbs-energy equilibrium constants, zone-based mesh discretization with optional heat/mass sources, and a segregated finite-volume solver using implicit species/temperature linearization solved with `scipy.sparse.linalg.spsolve`.

The module is organized into three layers:

1. **Physics/chemistry primitives** — `Specie`, `Reaction`, `Mixture`
2. **Domain/mesh/BC objects** — `domainSetup`, `Inlet`, `Outlet`, `Zone`, `Mesh`, `scalarField`
3. **Solver and orchestration** — `solver`, `build_reactor_from_context`, `ReactorPlotter`

---

## Module-Level Constants

| Name | Value | Meaning |
|---|---|---|
| `UNIVERSALGASCONSTANT` | 8.31446261815324 | Universal gas constant R [J/(mol·K)] |
| `T_REF` | 273.15 | Reference temperature [K] |
| `P_REF` | 101325.0 | Reference pressure [Pa] |
| `PI` | 3.141592653589793 | π |

---

## `Specie` (dataclass)

Represents a single chemical species with thermodynamic properties.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Species name |
| `molarMass` | `float` | — | Molar mass [kg/mol] |
| `heatCapacityModel` | `str` | `"const"` | `"const"` or `"polynomial"` |
| `enthalpyFormation` | `float` | `0.0` | Standard enthalpy of formation [J/mol] |
| `entropyFormation` | `float` | `0.0` | Standard entropy of formation [J/(mol·K)] |
| `heatCapacityValue` | `float` | `900.0` | Constant cp [J/(kg·K)], used if model is `"const"` |
| `heatCapacityCoefficients` | `list[float]` | `[]` | NASA-style polynomial coefficients (ascending order), used if model is `"polynomial"` |

Each instance gets an auto-incremented `id` via the class-level `counter`.

**Methods**

- `heatCapacity(T)` → heat capacity array. `"const"` returns `heatCapacityValue` broadcast to `T`'s shape; `"polynomial"` evaluates `cp_poly(T) * R`.
- `enthalpy(T)` → total enthalpy = `enthalpyFormation + sensible_enthalpy`, where sensible enthalpy is computed relative to `T_REF` (linear for `"const"`, integrated polynomial for `"polynomial"`).
- `from_dict(name, cfg)` (classmethod) → builds a `Specie` from a config dict; normalizes `None` polynomial coefficients to `[]`.

---

## `Reaction` (dataclass)

Encodes one reaction's stoichiometry, Arrhenius kinetics, and (optional) reversibility via Gibbs free energy.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Reaction label |
| `stochiometricCoefficients` | `np.ndarray` | `[]` | ν_i per species (reactants negative, products positive) |
| `speciesExponent` | `np.ndarray` | `[]` | Forward mass-action exponents α_i |
| `reversedSpecieExponent` | `np.ndarray` | `[]` | Backward mass-action exponents β_i |
| `isReversible` | `bool` | `True` | Enables/disables backward rate |
| `ahrreniusPreExponent` | `float` | `1.0` | Arrhenius A |
| `ahrreniusActivationEnergy` | `float` | `0.0` | Arrhenius Eₐ [J/mol] |
| `species` | `list[Specie]` | `[]` | Ordered species participating in this reaction |
| `entropyChange`, `enthalpyChange`, `molarMasses` | computed | — | Set in `__post_init__` |

Raises `ValueError` if `stochiometricCoefficients`, `speciesExponent`, `reversedSpecieExponent` don't all share the same length.

**Key formulas**

- Forward rate constant: `k_f(T) = A · exp(-Eₐ / RT)`
- Equilibrium constant: `K_p(T) = exp(-ΔG / RT)`, `ΔG = ΔH_rxn − TΔS_rxn`
- Backward rate constant: `k_b = k_f / K_c`, with `K_c = K_p · (P_ref / RT)^Δν`
- Reaction rate: `r_f = k_f · Π C_iᵅ`, `r_b = k_b · Π C_iᵝ`
- Mass source: `ṁ_i = ν_i · M_i · (r_f − r_b)`
- Heat source: `q̇ = -(r_f − r_b) · ΔH_rxn(T)`

**Methods**

| Method | Returns | Purpose |
|---|---|---|
| `from_dict(name, cfg, species)` | `Reaction` | Build from config dict + species list |
| `_equilibriumMask(rf, rb, rel_tol, abs_tol)` | `bool` array | Flags near-equilibrium cells to suppress stiff sources |
| `forwardRateConstant(T)` | array | k_f(T), T clipped ≥ 1e-12 |
| `equilibriumConstant(T)` | array | K_p(T), clipped to [1e-100, 1e100] |
| `backwardRateConstant(T)` | array | k_b(T), zero if irreversible |
| `enthalpyReactionChange(T)` | array | ΔH_rxn(T) = Σ ν_i·H_i(T) |
| `reactionRate(T, C)` | `(r_f, r_b)` | Forward/backward volumetric rates [mol/(m³·s)] |
| `reactionMassSource(rates, rel_tol, abs_tol)` | `(n_species, N)` | Per-species mass production/consumption [kg/(m³·s)] |
| `reactionHeatSource(T, rates, rel_tol, abs_tol)` | `(N,)` | Volumetric heat release [W/m³] |
| `reactionHeatSourceDerivative(T, rates, rel_tol, abs_tol)` | `(N,)` | Analytical dQ/dT Jacobian for implicit energy equation |
| `rateDerivativeConcentration(T, C, j)` | `(dRf/dCj, dRb/dCj)` | Jacobian of rates w.r.t. species j concentration |

**Notes**

- Concentrations are clipped to `1e-20` before power-law evaluation to avoid singularities at zero concentration with non-integer exponents.
- `K_c` is clipped to `[1e-50, ∞]` before inversion to `k_b`.
- Equilibrium masking prevents near-cancelling forward/backward rates from injecting stiff spurious source terms.

---

## `Mixture` (dataclass)

Represents a multi-species mixture and computes bulk thermophysical properties from mass fractions.

**Fields**

| Field | Default | Description |
|---|---|---|
| `densityModel` | `"ideal-incompressible-gas"` | or `"const"` |
| `densityValue` | `1.2225` | Used only if `densityModel == "const"` |
| `species` | `[]` | Ordered species list |
| `molarMasses` | computed | Assembled from `species` in `__post_init__` |

**Methods**

- `equivalentMolarMass(Y)` → harmonic-mean mixture molar mass: `M_mix = (Σ Y_i/M_i)⁻¹`, inverse-sum clipped to ≥1e-16.
- `idealGasDensity(T, Y)` → `ρ = P_ref · M_mix / (R·T)`.
- `mixtureHeatCapacity(T, Y)` → mass-weighted `cp_mix = Σ Y_i · cp_i(T)/M_i` [J/(kg·K)].
- `from_dict(cfg, species)` (classmethod).

**Note:** the `densityModel == "const"` branch is stored but dispatch logic must be handled by the caller (`Inlet.inletValues` does check it; other call sites should be verified).

---

## `domainSetup` (dataclass)

Lightweight geometry container.

| Field | Description |
|---|---|
| `diameter` | Duct inner diameter [m] |
| `inletMassFractions` | Optional reference inlet composition (unused elsewhere directly) |

Cross-sectional area used throughout: `A = π·d²/4`.

---

## `Inlet` (dataclass)

Defines inlet boundary conditions.

| Field | Default | Description |
|---|---|---|
| `position` | `0` | Cell/face index |
| `velocity` | `50.0` | Inlet velocity [m/s] |
| `temperature` | `700.0` | Inlet temperature [K] |
| `speciesMassFractions` | `[]` | Ordered Y_i at inlet |

**Methods**

- `inletValues(mixture, domain)` → `(massFlowrate, temperatureBC, specieFrac)`. Computes `ṅ = ρ·A·u` using `mixture.idealGasDensity` (or constant density) at the inlet state.
- `from_dict(cfg, species)` (classmethod) → normalizes species mass fractions by their sum (raises `ValueError` if sum ≤ 0).

---

## `Outlet` (dataclass)

Snapshot of solver state at a given cell (default: last cell).

| Field | Description |
|---|---|
| `position` | Cell index (default -1) |
| `temperature`, `density`, `velocity`, `massFlowrate` | Scalars |
| `speciesMassFractions`, `concentrations` | Per-species arrays |

**Methods**

- `fromSolver(slv, position=-1)` (classmethod) → builds an `Outlet` by reading solver fields at `position`; concentrations computed as `C_i = ρ·Y_i/M_i`.
- `asDict(species=None)` → dict representation; if `species` provided, mass fractions/concentrations are keyed by species name instead of index.
- `specieIndex(name, species)`, `massFraction(name, species)`, `concentration(name, species)` → convenience lookups by species name (case-insensitive).

---

## `Zone` (dataclass)

Represents one axial control-volume segment with optional heat/mass source activation.

| Field | Default | Description |
|---|---|---|
| `length` | `0.005` | Zone length [m] |
| `zoneType` | `"null"` | Label |

Auto-assigned `id` via class counter (increment-then-assign, starts at 1). Post-init sets `heatSource=False`, `massSource=False`, `heatSourceValue=0.0`.

**Methods**

- `from_dict(name, cfg)` (classmethod) → reads `length`, `zoneType`, `heatSource` (bool, default `False`), `massSource` (bool, default `True`), `heatSourceValue` (float, default `0.0`, `None` normalized to `0.0`).

**Reset note:** call `Zone.counter = 0` (and `Specie.counter = 0`) between independent simulation setups to avoid ID drift — `build_reactor_from_context` already does this.

---

## `Mesh`

Builds a 1D finite-volume mesh from an ordered list of `Zone` objects.

**Constructor:** `Mesh(domain, zoneList, sizing=0.005)`

**`meshCreate()`**
- Filters out zones with `length <= 0`.
- Computes cells-per-zone: `n_cells_zone = max(1, round(Lz / sizing))`, then adjusts actual cell size `dz_zone = Lz / n_cells_zone` to exactly fill each zone.
- Populates per-cell arrays: `cell_centers`, `cell_sizes`, `cell_volumes` (`= dz·A`), `cell_zone_id`, `cell_zone_type`, `cell_heat_flag`, `cell_mass_flag`, `cell_heat_value`.
- Sets `n_cells` (total) and `length` (sum of zone lengths).
- If no zones qualify, all arrays are set empty and `n_cells=0`.

**`from_dict(cfg, domain, zones)`** (classmethod) → reads `mesh.sizing` (default `0.005`) from config, constructs mesh, and calls `meshCreate()`.

---

## `scalarField`

Minimal container for a per-cell scalar field.

- `__init__(variable, field_type="specie")` — stores name/type; `cellField=None` until initialized.
- `fieldInitialize(mesh)` — allocates `cellField = zeros_like(mesh.cell_centers)`.

---

## `solver`

The core steady-state reactor solver. Supports **an arbitrary number of reactions** operating on a shared species/mixture definition.

### Constructor

```python
solver(mesh, mixture, reactions: List[Reaction], specieFields: List[scalarField], inlet: Inlet)
```

- Accepts either a single `Reaction` (auto-wrapped into a one-element list for backward compatibility) or a `List[Reaction]`.
- Computes initial `massFlux` from `inlet.inletValues(...)`.
- Stacks `specieFields` into `self.specieFields` (`shape = (n_species, n_cells)`).
- Initializes `temperatureField` and `velocityField` from inlet values.
- Computes initial `density` via `mixture.idealGasDensity`.
- Allocates zeroed arrays: `heatSourcesDerivative`, `massSources`, `massSourcesDerivative`, `heatSources`, `heatResidual`, `specieResidual`.
- Allocates `reactionRates` with shape `(n_reactions, 2, n_cells)` — axis 1 index 0 = forward rate, index 1 = backward rate, per reaction.

### Key Methods

**`concentrationArray()`**
Computes molar concentration field `C = ρ·Y/M` for all species, shape `(n_species, n_cells)`. Validates shape consistency between `Y`, `M`, and `ρ` and raises `ValueError` on mismatch.

**`update_density()`**
Recomputes `self.density` from current `T` and `Y` via `mixture.idealGasDensity`.

**`sourcesEvaluation(underRelaxationFactorHeatSource=0.05, underRelaxationFactorMassSource=0.15, eq_rel_tol=1e-8, eq_abs_tol=1e-20)`**

The multi-reaction accumulation core:

1. Saves old `massSources`/`massSourcesDerivative`/`heatReactionSources`/`heatReactionSourcesDerivative` for under-relaxation.
2. Zeroes all source accumulators, then adds any zone-prescribed heat (`mesh.cell_heat_value` where `cell_heat_flag` is true).
3. **Loops over every `Reaction` in `self.reactions` independently:**
   - Computes `(rateForward, rateBackward)` and stores them in `reactionRates[r_idx]`.
   - Computes each reaction's **own** equilibrium mask (`near_eq`), masked also by the zone's `cell_mass_flag`.
   - Computes and accumulates (`+=`) that reaction's contribution to `massSources`, `massSourcesDerivative` (looping over each species' concentration Jacobian via `rateDerivativeConcentration`), `heatReactionSources`, and `heatReactionSourcesDerivative`.
4. After the loop, applies a single under-relaxation blend (old vs. newly-summed) to the heat and mass source/derivative arrays.
5. Adds `heatReactionSources` into `heatSources`, and copies `heatReactionSourcesDerivative` into `heatSourcesDerivative`.

This design means each reaction can be independently near/at equilibrium without affecting the equilibrium detection of any other reaction sharing the same cell.

**`matrixSpecieEquationAssembly(specieIndex)`**
Builds a sparse tridiagonal-like (lower + main diagonal only — upwind advection) linear system `A·Y = b` for one species using implicit linearization of the source term (`dS_implicit = min(dS, 0)`), with inlet Dirichlet-style boundary injected into `b[0]`. Applies a near-zero mass fraction guard (`zero_tol = 1e-10`) that zeroes `dS` if the species is essentially absent everywhere, avoiding spurious stiff Jacobian terms.

**`matrixTemperatureEquationAssembly()`**
Analogous tridiagonal assembly for the energy equation, using `mixtureHeatCapacity` for the convective coefficient and `heatSourcesDerivative` for implicit source linearization.

**`specieScalarEquation()` / `heatEquation()`**
Compute explicit residuals (`specieResidual`, `heatResidual`) of the discretized species/energy balance for diagnostic/residual-tracking purposes (not used to drive the implicit solve itself, which uses the matrix assembly methods above).

**`initializeCase()`**
Sets all species fields and temperature field to inlet values uniformly across the domain; updates density.

**`steadyState(maxiter, relaxationFactorSpecie=0.4, relaxationFactorTemperature=0.4, convergenceCriteria=1e-6, temperatureClipLow=200, temperatureClipHigh=2000)`**

Main segregated iterative solve loop:

1. Updates density/velocity, then calls `sourcesEvaluation()`.
2. For each iteration:
   - Solves each species' linear system via `spsolve`, clips to `[0,1]`, applies under-relaxation.
   - Renormalizes all species mass fractions to sum to 1 per cell.
   - Updates density/velocity, re-evaluates sources.
   - Solves the temperature linear system, clips to `[temperatureClipLow, temperatureClipHigh]`, applies under-relaxation.
   - Computes scaled residuals `scaleddY` (normalized by mass flux) and `scaleddT` (normalized by mean temperature); logs progress every 10 iterations.
   - Breaks early if `max(dY, dT) < convergenceCriteria`.
3. Populates `self.outlet` via `Outlet.fromSolver(self)` and returns it.

---

## `build_reactor_from_context(case_ctx)`

Factory function that assembles a fully configured `solver` from a nested config dict (`case_ctx`), typically produced by `main.py`'s context-building pipeline.

**Expected `case_ctx` structure:**
```
case_ctx = {
    "chemistry": {"species": {...}, "reactions": {...}, "mixture": {...}},
    "mesh": {"zones": {...}, "mesh": {"sizing": ...}},
    "inlet": {"diameter": ..., "velocity": ..., "temperature": ..., "specie": {...}},
}
```

**Steps:**

1. Resets `Specie.counter` and `Zone.counter` to 0 (fresh ID space per case).
2. Builds `species` list from `chemistry["species"]`.
3. Builds **one or more** `Reaction` objects from every entry in `chemistry["reactions"]` (raises `ValueError` if zero reactions are defined — multi-reaction is fully supported, no upper limit).
4. **Validates** that every reaction's species ordering exactly matches the mixture's species ordering, raising `ValueError` on mismatch (prevents silent misalignment when summing sources across reactions).
5. Builds `Mixture`, `domainSetup`, `Zone` list, `Mesh`, `Inlet`.
6. Allocates a `scalarField` per species and constructs the `solver`.

**Returns:** `(slv, species)` tuple.

---

## `ReactorPlotter`

Visualization helper bound to a `solver` instance. All `save_*` methods accept `dpi=200` (default) for output resolution control.

| Method | Output | Notes |
|---|---|---|
| `get_axis()` | axial cell-center coordinates | Falls back to computing from `cell_lengths` if `cell_centers` absent |
| `_species_list()` | species list | Prefers `solver.mixture.species`; falls back to `solver.reactions[0].species`, then legacy `solver.reaction.species` |
| `save_temperature(path, dpi=200)` | temperature vs z | Single-axis line plot |
| `save_species(path, dpi=200)` | all species mass fractions vs z | One line per species |
| `save_species_subset(path, names, dpi=200)` | filtered species mass fractions vs z | `names` matched case-insensitively |
| `save_all(path, dpi=200)` | side-by-side T and Y_i plots | Two subplots sharing x-axis |
| `save_reaction_rates(path, dpi=200)` | forward/backward rate per reaction vs z | Requires `solver.reactionRates` + `solver.reactions`; dashed lines for backward rate of reversible reactions |
| `save_heat_source(path, dpi=200)` | reaction heat source vs z | Uses `heatReactionSources` if available, else `heatSources` |
| `save_concentrations(path, dpi=200)` | molar concentration per species vs z | Uses `solver.concentrationArray()` |

---

## Multi-Reaction Support Summary

This script fully supports an arbitrary number of simultaneous reactions:

- `solver.reactions` is a `List[Reaction]` (single `Reaction` auto-wrapped for compatibility).
- `sourcesEvaluation` loops over all reactions, computing **independent equilibrium masks per reaction**, and accumulates mass/heat sources and their Jacobians additively.
- `reactionRates` array shape is `(n_reactions, 2, n_cells)`, preserving per-reaction forward/backward rate history for diagnostics and plotting.
- `build_reactor_from_context` builds one `Reaction` per entry under `chemistry.reactions` in the YAML/dict config, with a species-ordering consistency check across all reactions and the mixture.
- `ReactorPlotter.save_reaction_rates` and `save_heat_source` visualize the combined/per-reaction picture.

---

## Known Limitations / Caveats

- **1D plug-flow only** — no radial profiles, no multi-dimensional transport.
- **Upwind advection scheme only** — the sparse matrix assembly uses first-order upwind (no central-difference or higher-order option).
- **Diagonal species Jacobian only** — `matrixSpecieEquationAssembly` linearizes each species equation independently; cross-species coupling terms (e.g., `dω_k/dY_j` for `j ≠ k`) are not included, which can slow convergence for strongly coupled multi-reaction systems.
- **Ideal gas density model** — `"const"` density dispatch is only implemented explicitly in `Inlet.inletValues`; verify it is handled consistently wherever `mixture.idealGasDensity` vs `densityValue` matters.
- **No pressure equation** — mass flux `F` is fixed from the inlet condition; pressure drop is not solved.
