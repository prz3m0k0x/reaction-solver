# PSOOPtimizer.py — Documentation

## Overview

`PSOOPtimizer.py` implements a **Multi-Objective Particle Swarm Optimizer (MOPSO)** using a Tchebycheff decomposition scheme with Das-Dennis reference weight vectors, neighborhood-based local guides, and constraint/bound handling via quadratic penalty functions. It gracefully degrades to standard single-objective PSO when `n_responses == 1`.

The module is organized into four classes:

1. **`PSOConfig`** — algorithm hyperparameters and problem definition (bounds, constraints, weight generation)
2. **`Swarm`** — particle population state, velocity update, personal/neighborhood bests, Tchebycheff scalarization
3. **`HistoryLogger`** — iteration logging and persistence
4. **`PSOOptimizer`** — orchestrates the evaluate → step → update loop

---

## `PSOConfig` (dataclass)

Holds all algorithm parameters and problem bounds/constraints.

**Fields**

| Field | Default | Description |
|---|---|---|
| `h_factor` | `25` | Divisions parameter for Das-Dennis weight generation; also population size when single-objective |
| `max_iter` | `15` | Number of PSO iterations |
| `n_params` | `2` | Dimensionality of the search space |
| `n_responses` | `1` | Number of objectives (`1` = single-objective PSO, `>1` = MOPSO) |
| `t_neighbors` | `5` | Neighborhood size T for local best updates (MOPSO only) |
| `w_init`, `w_finish` | `0.7`, `0.4` | Inertia weight schedule endpoints |
| `c1_init`, `c1_finish` | `0.6`, `0.8` | Cognitive coefficient schedule endpoints |
| `c2_init`, `c2_finish` | `0.5`, `0.6` | Social coefficient schedule endpoints |
| `v_max_factor` | `0.2` | Fraction of `(x_ub - x_lb)` used as max velocity per dimension |
| `x_lb`, `x_ub` | `[]` | Per-parameter lower/upper bounds; default to `±inf` if empty |
| `constr_matrix` | `[]` | Linear constraint matrix `A`, shape `(n_constraints, n_params)` |
| `constr_lb`, `constr_ub` | `[]` | Linear constraint bounds: `constr_lb <= A·x <= constr_ub` |

**`__post_init__`**
- Normalizes `x_lb`/`x_ub` to `np.ndarray` of shape `(n_params,)`, defaulting to `±inf` if not provided.
- Normalizes constraint arrays; `constr_matrix` defaults to an empty `(0, n_params)` array.
- **Validation** (raises `ValueError`):
  - `x_lb`/`x_ub` must have shape `(n_params,)`.
  - `constr_matrix` must have `n_params` columns.
  - `constr_lb`/`constr_ub` must each have either 0 or `n_constraints` entries.

**Methods**

- **`pop_size()`** → population size.
  - If `n_responses == 1`: returns `h_factor` directly.
  - Otherwise: returns `math.comb(h_factor + n_responses - 1, n_responses - 1)` — the number of Das-Dennis simplex-lattice points for `n_responses` objectives with `h_factor` divisions. Raises `ValueError` if `h_factor <= n_responses`.

- **`das_dennis_weights(m, H)`** (static) → `np.ndarray` of shape `(n_weights, m)`. Recursively enumerates all integer compositions of `H` into `m` non-negative parts, then normalizes by `H` to produce the classic Das-Dennis simplex-lattice weight vectors used in decomposition-based multi-objective algorithms (e.g., MOEA/D).

- **`algorithm_velocity_parameters`** (property) → `np.ndarray` of shape `(max_iter, 3)`. Linearly interpolates `[w, c1, c2]` from `(w_init, c1_init, c2_init)` at iteration 0 to `(w_finish, c1_finish, c2_finish)` at the final iteration — a standard time-varying PSO coefficient schedule (decreasing inertia, increasing social/cognitive pull).

---

## `Swarm`

Holds and evolves the particle population. Internally stores particles as `(n_params, pop_size)` arrays (parameter-major layout).

### Constructor

```python
Swarm(config: PSOConfig, initial_particles: np.ndarray, initial_gains: np.ndarray)
```

- `initial_particles`: shape `(n_params, pop_size)`.
- `initial_gains`: shape `(n_responses, pop_size)` — objective values for each particle.

**Initialization steps:**
1. Computes `v_max = v_max_factor · (x_ub - x_lb)` per dimension (broadcast over particles).
2. Initializes `velocity` uniformly in `[-v_max, v_max]`.
3. Sets `single_objective = (n_responses == 1)`.
4. **Neighborhood/weight setup:**
   - If single-objective: `weights = None`; every particle's "neighborhood" is the entire swarm (`neighbors` = tiled full index range) — effectively global-best PSO.
   - If multi-objective: generates Das-Dennis `weights` via `config.das_dennis_weights`, computes pairwise Euclidean distances between weight vectors, and selects each particle's `T = min(t_neighbors, pop_size)` closest weight-vector neighbors — the MOEA/D-style neighborhood structure.
5. Initializes ideal point `z_ref = min(initial_gains, axis=1)` and nadir point `z_nad = max(initial_gains, axis=1)` — used to normalize objectives in the Tchebycheff scalarization.
6. Sets `pbest_positions`/`pbest_gains` to the initial particles/gains, and computes `pbest_scalars` via `_tcheby_scalars`.
7. Initializes `gbest_positions`/`gbest_gains` (zeros) then immediately populates them via `_update_neighborhood_bests()`.

### Properties

- **`global_best_position`** → the single best particle position across the *entire* swarm (by minimum `pbest_scalars`), regardless of neighborhood structure. Useful for reporting a single "best" solution even in the MOPSO case.

### Methods

**`_penalty(particles)`**
Computes a quadratic exterior penalty for bound and linear constraint violations:
- Bound violation: `Σ (max(0, lb - x))² + (max(0, x - ub))²` per particle.
- Linear constraint violation (if `constr_matrix` non-empty): `Ax = constr_matrix @ particles`, then `Σ (max(0, clb - Ax))² + (max(0, Ax - cub))²`.
- Returns combined penalty, shape `(pop_size,)`. This penalty is added directly to the Tchebycheff scalar, so infeasible particles are always ranked worse than feasible ones with the same objective value.

**`_update_neighborhood_bests()`**
For each particle `i`, finds the neighbor (within `self.neighbors[i]`) with the lowest `pbest_scalars`, and copies that neighbor's position/gains into `gbest_positions[:, i]` / `gbest_gains[:, i]`. This is what makes the algorithm decomposition-based rather than a single global attractor: each particle is pulled toward the best-known solution *within its own weight neighborhood*.

**`_tcheby_scalars(gains, particles)`**
Computes the (penalized) weighted Tchebycheff scalarizing function used to rank particles:
- **Single-objective:** `|gain - z_ref| / scale`, where `scale = z_nad - z_ref` (or `1.0` if the range is degenerate, `< 1e-8`).
- **Multi-objective:** normalizes each objective by `(z_nad - z_ref)`, multiplies by the corresponding weight vector component, and takes `max` over objectives per particle (the defining operation of Tchebycheff decomposition — approximating the Pareto front by minimizing worst-case weighted deviation from the ideal point).
- Adds `_penalty(particles)` to the result in both branches.

**`step(w, c1, c2)`**
Standard PSO velocity/position update using per-neighborhood bests instead of a single global best:

\[
v \leftarrow w \cdot v + c_1 r_1 (p_{best} - x) + c_2 r_2 (g_{best} - x)
\]

- `r1`, `r2` are independent uniform random matrices in `[0,1]`, same shape as particles.
- Velocity is clipped to `±v_max`.
- Particles are updated (`x += v`) then clipped to `[x_lb, x_ub]`.

**`update_bests(new_gains)`**
After evaluating the stepped particles:
1. Updates `z_ref`/`z_nad` (ideal/nadir points) using the new gains — these move monotonically to always bound the best/worst observed values.
2. Recomputes `pbest_scalars` for the *existing* personal bests using the updated ideal/nadir (since the normalization scale may have shifted).
3. Computes scalars for the new particle positions and identifies `improved` particles (`new_scalar < pbest_scalar`).
4. Replaces personal bests (`pbest_positions`, `pbest_gains`, `pbest_scalars`) for improved particles only.
5. Calls `_update_neighborhood_bests()` to propagate improvements into each particle's neighborhood guide.

---

## `HistoryLogger`

Tracks optimization progress across iterations for later analysis/plotting.

**Constructor:** `HistoryLogger(config: PSOConfig)`
- Allocates `particle_history` of shape `(n_params, pop_size, max_iter + 1)` (all positions at every iteration, including iteration 0).
- `best_gain_history` starts as an empty list.

**Methods**

- **`log(iteration, particles, best_gain)`** → stores the current particle positions into `particle_history[:, :, iteration]`, appends `best_gain` to the history list, and prints a progress line (`Iter NN best_gain=X.XXXXXX`).
- **`save(path)`** → persists `particle_history` as a `.npy` file and `best_gain_history` as a semicolon-delimited `.csv`, both under the given `pathlib.Path` directory.

---

## `PSOOptimizer`

Top-level driver that wires together the objective function, `Swarm`, and `HistoryLogger`.

### Constructor

```python
PSOOptimizer(config: PSOConfig, objective_function: Callable[[np.ndarray, int], np.ndarray], initial_particles: np.ndarray)
```

- `objective_function(particle, iteration) -> array-like`: evaluates a **single** particle (shape `(n_params,)`) at a given iteration and returns its objective vector (length `n_responses`).
- Evaluates all `initial_particles` (shape `(n_params, pop_size)`) via `_evaluate` to get `initial_gains`.
- Constructs `self.swarm = Swarm(config, initial_particles, initial_gains)`.
- Constructs `self.logger = HistoryLogger(config)` and logs iteration 0.

### Classmethod

**`from_random(config, objective_function)`** → convenience constructor that samples `pop_size()` particles uniformly within `[x_lb, x_ub]` per dimension, then delegates to `__init__`. This is the typical entry point (used in `main.py`'s `PSOOptimizer.from_random(pso_cfg, objective_function=objective_function)`).

### Methods

**`_evaluate(particles, iteration=0)`**
Loops over each particle column, calls `objective_function(particle, iteration)`, collects results, and stacks them into a `(n_responses, pop_size)` array via `np.column_stack`. This is a **serial** evaluation loop — each particle's objective (which in this project's context means running a full reactor simulation) is evaluated one at a time, not in parallel.

**`run()`**
Main optimization loop:
1. Precomputes the full `[w, c1, c2]` schedule via `config.algorithm_velocity_parameters`.
2. For each iteration `j` in `range(max_iter)`:
   - Reads `(w, c1, c2)` for this iteration from the schedule.
   - Calls `swarm.step(w, c1, c2)` to move particles.
   - Evaluates the new particle positions via `_evaluate`.
   - Calls `swarm.update_bests(new_gains)` to update personal/neighborhood bests.
   - Logs the iteration (`j + 1`) with the current minimum `pbest_scalars` as `best_gain`.
3. Returns the final `Swarm` object (from which `global_best_position`, `pbest_gains`, etc. can be read).

---

## Algorithm Summary

This is a **MOEA/D-style Tchebycheff-decomposition PSO**:

- Each particle is permanently associated with one Das-Dennis weight vector (defining a decomposition subproblem of the multi-objective problem).
- Instead of a single global best, each particle is attracted toward the best solution found *within its weight-vector neighborhood* (`t_neighbors` closest weight vectors by Euclidean distance).
- Objectives are normalized on the fly using running ideal (`z_ref`) and nadir (`z_nad`) points, then combined via the weighted Tchebycheff scalarizing function — a proven approach for approximating non-convex Pareto fronts, unlike simple weighted-sum decomposition.
- Constraint handling uses exterior quadratic penalties added directly to the scalar fitness, applied uniformly to bounds and linear constraints.
- When `n_responses == 1`, the machinery degenerates cleanly to classic global-best PSO: `weights = None`, all particles share one neighborhood (the whole swarm), and the Tchebycheff scalar reduces to a simple normalized absolute deviation from the best-known value.

---

## Usage Pattern (as seen in `main.py`)

```python
pso_cfg = PSOConfig(
    h_factor=..., max_iter=..., n_params=..., n_responses=...,
    t_neighbors=..., w_init=..., w_finish=..., c1_init=..., c1_finish=...,
    c2_init=..., c2_finish=..., v_max_factor=...,
    x_lb=[...], x_ub=[...],
    constr_matrix=[...], constr_lb=[...], constr_ub=[...],
)

def objective_function(particle, iteration):
    # apply particle values to a case context, run the reactor solver,
    # and return a list of objective values (one per output definition)
    ...
    return [obj1, obj2, ...]

optimizer = PSOOptimizer.from_random(pso_cfg, objective_function=objective_function)
swarm = optimizer.run()
best_particle = swarm.global_best_position
```

Each PSO particle here represents a vector of reactor design/operating parameters (e.g., inlet SO2 fraction, inlet temperature), and each objective evaluation triggers a full `build_reactor_from_context` → `steadyState` simulation run.

---

## Known Limitations / Caveats

- **Serial evaluation only** — `_evaluate` loops over particles one at a time; there is no built-in parallelization, which matters when each evaluation is an expensive reactor simulation.
- **No convergence/early-stopping criterion** — `run()` always executes exactly `max_iter` iterations; there is no tolerance-based early exit.
- **`h_factor` must exceed `n_responses`** for multi-objective mode, and population size grows combinatorially with `h_factor` and `n_responses` (`C(h_factor + n_responses - 1, n_responses - 1)`) — large `h_factor` with 3+ objectives can produce very large populations.
- **No re-initialization of `v_max` or weight vectors mid-run** — both are fixed at `Swarm` construction time based on the initial config.
- **Neighborhood topology is static** — computed once from the Das-Dennis weight vectors at initialization and never recomputed, even though `t_neighbors` could conceptually vary.
