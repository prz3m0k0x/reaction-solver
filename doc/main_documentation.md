# main.py — Documentation

## Overview

`main.py` is the orchestration entry point that ties together the reactor solver (`scripts.solverReactor`), the multi-objective PSO optimizer (`scripts.PSOOPtimizer`), and a YAML-driven configuration/expression system (`scripts.usrExpr`). It:

1. Loads a set of YAML config files into a single nested **context** dictionary.
2. Resolves user-defined expressions embedded in that context (e.g., derived inlet values).
3. Runs a full 1D reactor simulation per PSO particle via `run_case`.
4. Extracts scalar objective values from simulation results for the PSO optimizer to minimize/maximize.
5. Persists per-case YAML records and diagnostic plots for every particle evaluation.
6. Drives `PSOOptimizer` to search the parameter space and reports the best particle found.

This file has no classes — it's a pipeline of small, composable functions plus a `main()` driver.

---

## Imports

| Import | Source | Purpose |
|---|---|---|
| `UserExpression` | `scripts.usrExpr` | Parses/evaluates string expressions against a context dict |
| `PSOConfig`, `PSOOptimizer` | `scripts.PSOOPtimizer` | Multi-objective PSO algorithm |
| `build_reactor_from_context`, `Outlet`, `ReactorPlotter` | `scripts.solverReactor` | Reactor model construction, outlet snapshot, plotting |

---

## Config Loading Functions

### `load_yaml(path)`
Reads a YAML file and returns its parsed content, or `{}` if the file is empty/null.

### `_save_yaml(path, data)`
Writes `data` to `path` as YAML (`sort_keys=False` to preserve insertion order), creating parent directories as needed.

### `load_expression_registry(path, expr_cls=UserExpression)`
Loads a YAML file expected to be a flat mapping of `name: expression_string`, and wraps each string in an `expr_cls` instance (default `UserExpression`). Raises `TypeError` if the file isn't a dict, or if any value isn't a string. Returns `{name: UserExpression(...)}`.

### `build_context(config_dir, expr_cls=UserExpression)`
Assembles the full simulation context from the following files under `config_dir`:

| File | Context key(s) |
|---|---|
| `meshConfig.yaml` | `ctx["mesh"]` |
| `inletConfig.yaml` | `ctx["inlet"]` (`diameter`, `velocity`, `temperature`, `specie`) |
| `solverNumerics.yaml` | `ctx["solver"]` |
| `speciesConfig.yaml` | `ctx["chemistry"]` |
| `psoAlgorithm.yaml` | `ctx["pso"]` |
| `plottingConfig.yaml` (optional) | `ctx["plotting"]` — defaults to `{}` if the file doesn't exist |
| `userExpressions.yaml` | `ctx["expressions"]` — expression registry |
| `outletConfig.yaml` (optional) | `ctx["outletExpressions"]` — expression registry, `{}` if the file doesn't exist |

Prints the loaded outlet expression names for visibility, then returns the assembled `ctx` dict.

---

## Expression Resolution

### `resolve_value(value, root_ctx, expr_registry)`
Recursive helper that walks a value (scalar, dict, or list):
- If `value` is a string matching a key in `expr_registry`, evaluates that expression against `root_ctx` and returns `(result, True)`.
- If `value` is a dict, recurses into each item (skipping the `"expressions"`/`"outletExpressions"` keys to avoid self-referential resolution), mutating in place.
- If `value` is a list, recurses into each element, mutating in place.
- Otherwise, returns `(value, False)` unchanged.
- Returns `(possibly_updated_value, changed_any)`.

### `resolve_expressions_in_context(ctx, max_passes=10)`
Repeatedly calls `resolve_value(ctx, ctx, ctx.get("expressions", {}))` until no further changes occur (fixed-point iteration), allowing expressions that depend on other expressions to resolve in the correct order regardless of declaration order. Raises `RuntimeError` if resolution doesn't converge within `max_passes`.

### `evaluate_named_expressions(expr_registry, context)`
Similar fixed-point resolver, but for a **separate** named registry (typically `outletExpressions`) evaluated against a results context rather than mutating the context itself:
- Attempts to evaluate every pending expression each pass, extending the context with already-resolved results so later expressions can reference earlier ones.
- Catches `KeyError`/`ValueError`/`TypeError` per-expression (treated as "not ready yet") and retries in the next pass.
- Runs up to 20 passes; raises `RuntimeError` listing any expressions that never resolved.
- Returns a flat `{name: value}` dict (used to populate `result_ctx["derived"]`).

---

## Dotted-Path Helpers

### `set_by_dotted_path(data, path, value)`
Sets a nested value using a dot-separated path string (e.g., `"inlet.specie.so2"`). Raises `KeyError` if any intermediate key or the final key doesn't already exist in `data` — this is a **strict setter**, it will not create new keys.

### `get_by_dotted_path(data, path)`
Reads a nested value using the same dotted-path convention. Raises `KeyError` on any missing key or non-dict intermediate node.

### `apply_particle_to_context(ctx, particle, parameter_defs)`
For each `(x, param)` pair zipped from a PSO particle vector and the `parameter_defs` list (from `pso.parameters` config), sets `ctx` at `param["key"]` to `float(x)`. This is how raw PSO particle coordinates get mapped back into meaningful simulation inputs (e.g., inlet SO2 fraction, inlet temperature).

---

## Objective Extraction

### `extract_objectives_for_pso(result_ctx, output_defs)`

Converts a completed case's `result_ctx` into a list of scalar objective values, one per entry in `output_defs` (from `pso.outputs` config). For each `outdef` with a `key` and optional `goal` (`"minimize"` default, or `"maximize"`):

1. **Derived expression lookup** — if `key` is present in `result_ctx["derived"]`, uses that value directly.
2. **`expr3`** (hardcoded) — SO2 conversion fraction: `(inlet_so2 - outlet_so2) / inlet_so2`.
3. **`expr4`** (hardcoded) — temperature drop: `inlet_temperature - outlet_temperature`.
4. **Fallback** — resolves `key` as a dotted path directly into `result_ctx` via `get_by_dotted_path`.

Each extracted `value` is negated (`pso_value = -value`) if `goal == "maximize"`, since the underlying `PSOOptimizer` always minimizes. Verbose `print` statements log every step for debugging. Returns the list of `pso_value`s in output-definition order — this is exactly what `PSOOptimizer`'s `objective_function` callback must return.

**Note:** the `expr3`/`expr4` branches are hardcoded fallbacks specific to the SO2 conversion reactor use case; general studies should prefer defining the same logic via `outletExpressions` so it appears in `result_ctx["derived"]` instead.

---

## Plotting Integration

### `generate_plots(slv, species, case_dir, plotting_cfg)`

Drives `ReactorPlotter` based on a `plotting` config block:

```yaml
plotting:
  enabled: true
  temperature: true
  species: true
  profiles: true
  reaction_rates: true
  heat_source: true
  species_subset: [SO2, O2, SO3]
  dpi: 200
```

- If `plotting_cfg.enabled` is explicitly `False`, skips all plotting for that case (useful to speed up large PSO sweeps).
- Reads `dpi` (default `200`) and `species_subset` (optional list of species names) from the config.
- Uses an inner `wants(flag_name, default=True)` helper to check whether a given plot type is enabled, defaulting to `True` for the "core" plots (`temperature`, `species`, `profiles`) and `False` for the "extra" plots (`reaction_rates`, `heat_source`, `concentrations`).
- Calls the corresponding `ReactorPlotter` method for each enabled plot type, always passing `dpi=dpi`:
  - `save_temperature` → `temperature.png`
  - `save_species_subset` (if `species_subset` given and the plotter supports it) or `save_species` → `species.png`
  - `save_all` → `profiles.png`
  - `save_reaction_rates` (if plotter supports it) → `reaction_rates.png`
  - `save_heat_source` (if plotter supports it) → `heat_source.png`
  - `save_concentrations` (if plotter supports it) → `concentrations.png`
- The `hasattr(plotter, ...)` guards mean this function is forward-compatible: new plot types added to `ReactorPlotter` in the future activate automatically without further changes here, and if a method is missing it's silently skipped rather than raising.

---

## Case Execution

### `run_case(case_ctx, case_dir)`

Executes one full reactor simulation for a given resolved context and writes debug artifacts:

1. Builds the solver: `slv, species = build_reactor_from_context(case_ctx)`.
2. Initializes fields to inlet conditions: `slv.initializeCase()`.
3. Reads solver numerics from `case_ctx["solver"]` (`underRelaxationFactors.species`, `underRelaxationFactors.temperature`, `maxIter` default `1000`, `scaledResidual`, `temperatureClipLow`/`High`).
4. Runs `slv.steadyState(...)` to convergence; if it returns `None` (unexpected), falls back to constructing an `Outlet` directly via `Outlet.fromSolver(slv)`.
5. Converts outlet state to a plain dict via `outlet_obj.asDict(species=species)`.
6. Saves `reactorDebug.yaml` containing mesh cell count/length and the full outlet dict.
7. Calls `generate_plots(slv, species, case_dir, case_ctx.get("plotting", {}))` to produce configured diagnostic plots.
8. Returns a flattened summary dict: `temperature`, `specie` (mass fractions by name), `density`, `velocity`, `massFlowrate`, `concentrations` (by name) — all cast to plain Python floats for YAML/JSON serializability.

---

## PSO Configuration Builder

### `make_pso_config_from_context(ctx)`

Translates the `ctx["pso"]` YAML block into a `PSOConfig` instance:

- `pso_block = ctx["pso"]["pso"]` supplies algorithm hyperparameters (`hfactor`, `maxiter`, `tneighbors`, `winit`/`wfinish`, `c1init`/`c1finish`, `c2init`/`c2finish`, `vmaxfactor`).
- `parameter_defs = ctx["pso"]["parameters"]` — each entry has a `bounds: [lb, ub]` and a `key` (dotted path); `n_params = len(parameter_defs)`, and `x_lb`/`x_ub` are extracted directly from the bounds.
- `output_defs = ctx["pso"]["outputs"]` — defines `n_responses = len(output_defs)`.
- `constraints = ctx["pso"].get("constraints", {})` — optional; reads `linear` constraint entries, each with `A` (row of the constraint matrix), `lb`, `ub`. Empty lists if no constraints are defined.
- Returns the fully constructed `PSOConfig`.

---

## `main()`

The top-level driver:

1. Builds the base context from `config/` and resolves all top-level expressions (`resolve_expressions_in_context`).
2. Builds `pso_cfg` via `make_pso_config_from_context` and prints it.
3. Defines a closure `objective_function(particle, iteration)` that, for each PSO particle evaluation:
   - Deep-copies `base_ctx` into a fresh `case_ctx` (so particles don't interfere with each other or the base config).
   - Applies the particle's parameter values into `case_ctx` via `apply_particle_to_context`.
   - Logs the applied inlet SO2 fraction and temperature for traceability.
   - Re-resolves expressions in `case_ctx` (since parameter changes may affect derived quantities like normalized species fractions).
   - Determines a unique case directory via `make_case_dir("cases", case_ctx["pso"]["study"]["name"])` (auto-incrementing suffix, e.g. `cases/so2study1`, `cases/so2study2`, ...).
   - Saves the fully resolved, serializable case context to `caseSetup.yaml`.
   - Runs the simulation via `run_case(case_ctx, case_dir)` to get `reactor_result`.
   - Builds a `result_ctx` (serializable copy of `case_ctx` plus `outlet = reactor_result`) and evaluates any `outletExpressions` against it via `evaluate_named_expressions`, storing results under `result_ctx["derived"]`.
   - Saves `outlet.yaml` containing both the raw `outlet` result and the `derived` expression values.
   - Returns `extract_objectives_for_pso(result_ctx, case_ctx["pso"]["outputs"])` — the objective vector for this particle.
4. Constructs `optimizer = PSOOptimizer.from_random(pso_cfg, objective_function=objective_function)` and runs it: `swarm = optimizer.run()`.
5. Reads `swarm.global_best_position` and prints the optimization result.

### Guard

```python
if __name__ == "__main__":
    main()
```

---

## Data Flow Summary

```
config/*.yaml
      │
      ▼
build_context ──► base_ctx (with expression registries)
      │
      ▼
resolve_expressions_in_context ──► fully resolved base_ctx
      │
      ▼
make_pso_config_from_context ──► PSOConfig
      │
      ▼
PSOOptimizer.from_random(pso_cfg, objective_function)
      │
      ├─ for each particle, each iteration:
      │     deep-copy base_ctx → case_ctx
      │     apply_particle_to_context (write PSO params into case_ctx)
      │     resolve_expressions_in_context (re-resolve derived values)
      │     make_case_dir (new "cases/<study><n>/" directory)
      │     run_case:
      │         build_reactor_from_context ──► solver, species
      │         solver.initializeCase()
      │         solver.steadyState(...) ──► Outlet
      │         generate_plots (ReactorPlotter, per plottingConfig)
      │         write reactorDebug.yaml
      │     evaluate_named_expressions (outletExpressions ──► derived)
      │     write outlet.yaml
      │     extract_objectives_for_pso ──► objective vector
      │
      ▼
swarm.global_best_position ──► best PSO particle (design/operating parameters)
```

---

## Notable Design Choices / Caveats

- **Strict dotted-path setters/getters** (`set_by_dotted_path`, `get_by_dotted_path`) require every intermediate key to already exist in the base config — this is intentional to catch typos in `pso.parameters[*].key` early, but means new parameter paths must be pre-declared with a placeholder value in the relevant YAML file.
- **Deep-copy per particle** (`copy.deepcopy(base_ctx)`) guarantees PSO particles don't share mutable state, but this can be a performance bottleneck for very large contexts evaluated over large populations/many iterations.
- **Hardcoded `expr3`/`expr4` objective branches** in `extract_objectives_for_pso` couple `main.py` to a specific SO2-conversion use case; migrating this logic into `outletExpressions` YAML entries would make the pipeline fully generic.
- **Case directories accumulate indefinitely** — `make_case_dir` always creates a new incrementing directory (`study1`, `study2`, ...) and never cleans up old ones, so long PSO runs (many particles × many iterations) will produce a large number of `cases/` subdirectories, each with 3+ PNG plots plus YAML files.
- **No parallelism** — since `PSOOptimizer._evaluate` calls `objective_function` serially, `run_case` (a full nonlinear reactor solve plus plotting) runs once per particle per iteration with no concurrency; runtime scales linearly with `pop_size() × max_iter`.
