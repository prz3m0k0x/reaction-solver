# main

## Overview

`main.py` is the orchestration layer of the project. It connects the YAML configuration files, the safe expression engine, the reactor solver, and the PSO optimizer into one executable optimization workflow. [file:66]

The script does not implement the reactor physics or the PSO algorithm itself. Instead, it prepares context dictionaries, applies optimization variables to those contexts, resolves dependent expressions, runs one reactor case per particle evaluation, extracts objective values, and finally launches the optimizer. [file:66]

## Role in the project

The script imports `UserExpression` from `scripts.usrExpr`, `PSOConfig` and `PSOOptimizer` from `scripts.PSOOPtimizer`, and `build_reactor_from_context`, `Outlet`, and `ReactorPlotter` from `scripts.solverReactor`. This makes `main.py` the integration point where configuration, chemistry simulation, and optimization are tied together. [file:66]

In practical terms, the script is responsible for: [file:66]

- Loading YAML-based reactor and PSO settings. [file:66]
- Building a runtime context dictionary. [file:66]
- Loading named user expressions and outlet expressions. [file:66]
- Reapplying optimization variables to a fresh unresolved context for every particle. [file:66]
- Resolving dependent expressions after particle application. [file:66]
- Running a reactor simulation for each optimization sample. [file:66]
- Evaluating derived output expressions and converting them to PSO objective values. [file:66]
- Creating case folders and writing YAML/PNG diagnostics. [file:66]

## Imported dependencies

The script uses these external and internal modules: [file:66]

| Import | Purpose |
| --- | --- |
| `copy` | Deep-copy of case contexts before expression resolution and particle application. [file:66] |
| `pathlib.Path` | File and directory handling. [file:66] |
| `yaml` | YAML loading and writing. [file:66] |
| `scripts.usrExpr.UserExpression` | Safe evaluation of symbolic expressions from configuration files. [file:66] |
| `scripts.PSOOPtimizer.PSOConfig` | PSO configuration container. [file:66] |
| `scripts.PSOOPtimizer.PSOOptimizer` | Optimization engine. [file:66] |
| `scripts.solverReactor.build_reactor_from_context` | Builds a solver instance from the resolved case context. [file:66] |
| `scripts.solverReactor.Outlet` | Fallback outlet extraction if the solver does not return one directly. [file:66] |
| `scripts.solverReactor.ReactorPlotter` | Writes reactor plots into the case directory. [file:66] |

## High-level workflow

The top-level workflow implemented in `main()` is: [file:66]

1. Build an unresolved base context from the `config` directory. [file:66]
2. Resolve a temporary copy of that context to safely construct the PSO configuration. [file:66]
3. Create an objective function that, for each particle, deep-copies the raw base context, applies particle values, resolves expressions, runs the reactor case, evaluates outputs, and returns objective values. [file:66]
4. Create a `PSOOptimizer` instance from random initialization. [file:66]
5. Run the optimizer and print the best particle at the end. [file:66]

This separation between unresolved template context and per-particle resolved context is a key design feature of the current script. It prevents dependent expressions from being permanently collapsed before optimization variables are updated. [file:66]

## Utility functions

### `load_yaml(path)`

This helper opens a YAML file, parses it using `yaml.safe_load`, and returns an empty dictionary if the file is empty. It is used as the base loader for all configuration blocks. [file:66]

### `_save_yaml(path, data)`

This helper writes a Python object to YAML. It ensures the parent directory exists and then uses `yaml.safe_dump(..., sort_keys=False)` to preserve a human-friendly key order in saved case files. [file:66]

This function is used to save generated case data such as `caseSetup.yaml`, `outlet.yaml`, and `reactorDebug.yaml`. [file:66]

### `load_expression_registry(path, expr_cls=UserExpression)`

This function loads a YAML mapping of expression names to expression strings and converts each string into a `UserExpression` object. It validates that the loaded YAML content is a dictionary and that every expression value is a string. [file:66]

The return value is a dictionary of the form: [file:66]

```python
{
    "expr1": UserExpression("..."),
    "expr2": UserExpression("..."),
}
```

It is used for both `userExpressions.yaml` and `outletConfig.yaml`. [file:66]

### `make_serializable_context(ctx)`

This helper deep-copies a context dictionary and removes the `expressions` and `outletExpressions` registries. These objects contain `UserExpression` instances, so removing them produces a cleaner YAML-serializable snapshot of the case. [file:66]

The function is used before saving `caseSetup.yaml` and before building `result_ctx` for output extraction. [file:66]

### `make_case_dir(base_dir, study_name)`

This function creates numbered case folders such as `cases/<study_name>1`, `cases/<study_name>2`, and so on. It increments the suffix until it finds a folder name that does not already exist. [file:66]

This gives each particle evaluation its own unique output directory. [file:66]

## Context construction

### `build_context(config_dir, expr_cls=UserExpression)`

This function loads the full project configuration from a directory. It reads: [file:66]

- `meshConfig.yaml` [file:66]
- `inletConfig.yaml` [file:66]
- `solverNumerics.yaml` [file:66]
- `speciesConfig.yaml` [file:66]
- `psoAlgorithm.yaml` [file:66]
- `userExpressions.yaml` [file:66]
- optionally `outletConfig.yaml` [file:66]

It then reorganizes these into a normalized runtime context dictionary with top-level keys: [file:66]

- `mesh` [file:66]
- `inlet` [file:66]
- `solver` [file:66]
- `chemistry` [file:66]
- `pso` [file:66]
- `expressions` [file:66]
- `outletExpressions` [file:66]

The inlet block is reshaped so that `diameter`, `velocity`, `temperature`, and `specie` are directly accessible in the runtime format expected by the solver builder and expression logic. [file:66]

## Expression resolution

### `resolve_value(value, root_ctx, expr_registry)`

This is the recursive worker used to replace symbolic expression names embedded inside the context with their numeric values. If `value` is a string that matches a key in the expression registry, it is evaluated immediately using the root context. [file:66]

If `value` is a dictionary or list, the function recursively resolves its children. It explicitly skips the `expressions` and `outletExpressions` keys to avoid trying to overwrite the expression registries themselves. [file:66]

### `resolve_expressions_in_context(ctx, max_passes=10)`

This function repeatedly calls `resolve_value()` until no more replacements occur or `max_passes` is exceeded. It exists because one expression may depend on another expression that becomes resolvable only after an earlier substitution pass. [file:66]

If the loop never stabilizes within the pass limit, the function raises `RuntimeError`. On success, it returns the mutated context with symbolic references replaced by numbers. [file:66]

### `evaluate_named_expressions(expr_registry, context)`

This function is different from `resolve_expressions_in_context()`. Instead of modifying the main context tree, it evaluates a registry of named expressions into a separate result dictionary. [file:66]

It repeatedly attempts to evaluate pending expressions, augmenting the available evaluation context with already-resolved results. This allows derived outputs to depend on each other as long as the dependency chain eventually resolves. [file:66]

If any expressions remain unresolved after the iteration budget, the function raises `RuntimeError`. [file:66]

## Dotted-path utilities

### `set_by_dotted_path(data, path, value)`

This helper assigns a value into a nested dictionary using a dotted key path such as `inlet.specie.so2`. It walks through the dictionary hierarchy and raises `KeyError` if an intermediate or final key does not exist. [file:66]

### `get_by_dotted_path(data, path)`

This helper reads a value from a nested dictionary using a dotted key path. It validates that each traversal step remains inside a dictionary and raises `KeyError` on invalid paths. [file:66]

These two helpers are central to parameter injection and output extraction. [file:66]

## Optimization-variable application

### `apply_particle_to_context(ctx, particle, parameter_defs)`

This function applies one PSO particle vector to the case context. It loops over the particle coordinates and the corresponding parameter definitions and writes each coordinate into the nested context using the parameter's `key` field. [file:66]

For example, if a parameter definition contains `key: inlet.specie.so2`, the corresponding particle value is written directly into `ctx["inlet"]["specie"]["so2"]`. [file:66]

## Objective extraction

### `extract_objectives_for_pso(result_ctx, output_defs)`

This function converts the solved reactor result into the response vector returned to the optimizer. It supports several output modes: [file:66]

- Values already present in `result_ctx["derived"]`. [file:66]
- Special-case outputs `expr3` and `expr4`. [file:66]
- General dotted-path extraction from the result context. [file:66]

The special cases are: [file:66]

- `expr3`: computes SO2 conversion as `(inlet_so2 - outlet_so2) / inlet_so2`. [file:66]
- `expr4`: computes temperature drop as `inlet_temperature - outlet_temperature`. [file:66]

Because the optimizer is written in minimization form, any output whose goal is `
maximize` is negated before being returned. Outputs marked as `minimize` are passed through unchanged. [file:66]

## Running one reactor case

### `run_case(case_ctx, case_dir)`

This function performs one full reactor simulation for one resolved case context. It builds the solver, initializes the case, reads numerical settings from the `solver` block, and calls `steadyState()` with the configured iteration limits, under-relaxation factors, residual criterion, and temperature clipping bounds. [file:66]

If the solver returns `None`, the function falls back to `Outlet.fromSolver(slv)`. After solving, it converts the outlet object to a labeled dictionary using `asDict(species=species)`. [file:66]

The function also writes a debug YAML file called `reactorDebug.yaml` containing mesh metadata and the outlet data. It then saves three plots into the case directory using `ReactorPlotter`: [file:66]

- `temperature.png` [file:66]
- `species.png` [file:66]
- `profiles.png` [file:66]

The return value is a compact outlet dictionary with temperature, species fractions, density, velocity, mass flow rate, and concentrations. [file:66]

## PSO configuration builder

### `make_pso_config_from_context(ctx)`

This function translates the `pso` block of the runtime context into a `PSOConfig` object. It extracts: [file:66]

- Global PSO settings from `ctx["pso"]["pso"]`. [file:66]
- Variable definitions from `ctx["pso"]["parameters"]`. [file:66]
- Response definitions from `ctx["pso"]["outputs"]`. [file:66]
- Optional linear constraints from `ctx["pso"].get("constraints", {})`. [file:66]

It builds lower and upper parameter bounds from each parameter definition and converts linear constraints into `constr_matrix`, `constr_lb`, and `constr_ub` arrays expected by `PSOConfig`. [file:66]

The returned configuration object contains swarm size and schedule settings such as `h_factor`, `max_iter`, `t_neighbors`, `w_init`, `w_finish`, `c1_init`, `c1_finish`, `c2_init`, `c2_finish`, and `v_max_factor`, along with bounds and constraints. [file:66]

## `main()` function

### Purpose

`main()` is the executable entry point of the script. It constructs the unresolved template context, derives the PSO configuration, defines the objective function closure, runs the optimizer, and prints the best result. [file:66]

### Detailed flow

#### 1. Build unresolved base context

The script starts with: [file:66]

```python
base_ctx_raw = build_context("config", UserExpression)
```

This preserves symbolic expressions in the template context rather than resolving them immediately. [file:66]

#### 2. Build a temporary resolved context for PSO settings

Next, `main()` creates a temporary resolved copy: [file:66]

```python
pso_ctx = resolve_expressions_in_context(copy.deepcopy(base_ctx_raw))
pso_cfg = make_pso_config_from_context(pso_ctx)
```

This allows PSO-related expressions to be evaluated without mutating the raw base template that will later be reused for each particle. [file:66]

#### 3. Define `objective_function(particle, iteration)`

The nested `objective_function` is the core callback passed to the optimizer. For every particle evaluation, it performs the following sequence: [file:66]

1. Deep-copy `base_ctx_raw` so that the new case starts from unresolved symbolic definitions. [file:66]
2. Apply the particle values into the copied context with `apply_particle_to_context(...)`. [file:66]
3. Print diagnostic information about the particle and selected inlet variables. [file:66]
4. Resolve all dependent expressions in the copied case context. [file:66]
5. Create a new numbered case directory under `cases/`. [file:66]
6. Save the resolved case setup to `caseSetup.yaml`. [file:66]
7. Run the reactor simulation through `run_case(...)`. [file:66]
8. Build `result_ctx` by combining the case input context and the reactor outlet data. [file:66]
9. Evaluate outlet expressions into `derived`. [file:66]
10. Save `outlet.yaml` containing both the raw outlet and the derived values. [file:66]
11. Convert the result to a PSO objective vector using `extract_objectives_for_pso(...)`. [file:66]

This design is especially important for dependent inlet expressions such as oxygen or nitrogen fractions defined in terms of another optimized variable. Because the context is copied first and resolved only after particle application, the dependent values are recalculated for every particle rather than frozen from startup. [file:66]

#### 4. Create and run the optimizer

After defining the objective function, `main()` creates the optimizer using: [file:66]

```python
optimizer = PSOOptimizer.from_random(pso_cfg, objective_function=objective_function)
swarm = optimizer.run()
```

After the optimization finishes, it reads and prints `swarm.global_best_position`. [file:66]

## Case output files

For each particle evaluation, the script writes a dedicated case directory under `cases/<study_name>N`. Inside that directory, the current script writes at least these artifacts: [file:66]

| File | Contents |
| --- | --- |
| `caseSetup.yaml` | Resolved input case used for the reactor run. [file:66] |
| `outlet.yaml` | Reactor outlet data plus evaluated derived outputs. [file:66] |
| `reactorDebug.yaml` | Mesh metadata and labeled outlet information. [file:66] |
| `temperature.png` | Axial temperature profile. [file:66] |
| `species.png` | Axial species mass-fraction profiles. [file:66] |
| `profiles.png` | Combined summary plot. [file:66] |

This output structure makes the optimization trace inspectable and reproducible after the run. [file:66]

## Data structures used by the script

The script relies on a nested context dictionary that acts as the shared data model between configuration loading, expression resolution, solver setup, and output processing. [file:66]

A simplified structure is: [file:66]

```python
ctx = {
    "mesh": {...},
    "inlet": {
        "diameter": ...,
        "velocity": ...,
        "temperature": ...,
        "specie": {...},
    },
    "solver": {...},
    "chemistry": {...},
    "pso": {...},
    "expressions": {...},
    "outletExpressions": {...},
}
```

The same structure is reused throughout the run, with particle values injected into selected locations and outlet data added later in a separate `result_ctx`. [file:66]

## Important design decision

One of the most important features of this version of `main.py` is that it keeps `base_ctx_raw` unresolved and only resolves expressions after particle values are applied inside the objective function. This prevents dependent symbolic inputs from being evaluated too early and becoming fixed before optimization begins. [file:66]

That design directly supports parameterizations where one optimized variable controls other inlet quantities through expressions. [file:66]

## Minimal execution example

```python
if __name__ == "__main__":
    main()
```

When executed, the script expects a `config/` directory containing the YAML files required by `build_context()`. It then launches the full reactor optimization workflow automatically. [file:66]

## Function summary

| Function | Purpose |
| --- | --- |
| `load_yaml` | Read one YAML file. [file:66] |
| `_save_yaml` | Write one YAML file and create parent folders if needed. [file:66] |
| `load_expression_registry` | Load named expressions as `UserExpression` objects. [file:66] |
| `make_serializable_context` | Remove non-serializable expression registries from a context snapshot. [file:66] |
| `make_case_dir` | Create a unique numbered case folder. [file:66] |
| `build_context` | Build the main runtime context from YAML configuration files. [file:66] |
| `resolve_value` | Recursive helper for symbolic-value replacement. [file:66] |
| `resolve_expressions_in_context` | Resolve embedded expressions inside the context. [file:66] |
| `evaluate_named_expressions` | Evaluate named derived expressions into a separate result dictionary. [file:66] |
| `set_by_dotted_path` | Write nested dictionary values via dotted keys. [file:66] |
| `get_by_dotted_path` | Read nested dictionary values via dotted keys. [file:66] |
| `apply_particle_to_context` | Inject one particle vector into the case context. [file:66] |
| `extract_objectives_for_pso` | Convert case results into PSO objective values. [file:66] |
| `run_case` | Execute one resolved reactor case and save diagnostics. [file:66] |
| `make_pso_config_from_context` | Convert context settings into `PSOConfig`. [file:66] |
| `main` | Launch the full optimization workflow. [file:66] |
