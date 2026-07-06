
# main.py
import copy
from pathlib import Path
import yaml

from scripts.usrExpr import UserExpression
from scripts.PSOOPtimizer import PSOConfig, PSOOptimizer
from scripts.solverReactor import build_reactor_from_context, Outlet, ReactorPlotter


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_expression_registry(path, expr_cls=UserExpression):
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a mapping of name: expression")

    registry = {}
    for name, expr in raw.items():
        if not isinstance(expr, str):
            raise TypeError(
                f"{path}: expression \'{name}\' must be a string, got {type(expr).__name__}"
            )
        registry[name] = expr_cls(expr)

    return registry


def make_serializable_context(ctx):
    out = copy.deepcopy(ctx)
    out.pop("expressions", None)
    out.pop("outletExpressions", None)
    return out


def make_case_dir(base_dir, study_name):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        case_dir = base_dir / f"{study_name}{i}"
        if not case_dir.exists():
            case_dir.mkdir(parents=True, exist_ok=False)
            return case_dir
        i += 1


def build_context(config_dir, expr_cls=UserExpression):
    config_dir = Path(config_dir)

    mesh_cfg = load_yaml(config_dir / "meshConfig.yaml")
    inlet_cfg = load_yaml(config_dir / "inletConfig.yaml")
    solver_cfg = load_yaml(config_dir / "solverNumerics.yaml")
    species_cfg = load_yaml(config_dir / "speciesConfig.yaml")
    pso_cfg = load_yaml(config_dir / "psoAlgorithm.yaml")
    plotting_cfg = load_yaml(config_dir / "plottingConfig.yaml") if (config_dir / "plottingConfig.yaml").exists() else {}

    expr_registry = load_expression_registry(config_dir / "userExpressions.yaml", expr_cls)

    outlet_expr_path = config_dir / "outletConfig.yaml"
    outlet_expr_registry = load_expression_registry(outlet_expr_path) if outlet_expr_path.exists() else {}

    print("Loaded outlet expressions:", outlet_expr_registry.keys())

    ctx = {
        "mesh": mesh_cfg,
        "inlet": {
            "diameter": inlet_cfg["diameter"],
            "velocity": inlet_cfg["inletVelocity"],
            "temperature": inlet_cfg["inletTemperature"],
            "specie": inlet_cfg["speciesYInlet"],
        },
        "solver": solver_cfg,
        "chemistry": species_cfg,
        "pso": pso_cfg,
        "plotting": plotting_cfg,
        "expressions": expr_registry,
        "outletExpressions": outlet_expr_registry,
    }

    return ctx


def resolve_value(value, root_ctx, expr_registry):
    if isinstance(value, str) and value in expr_registry:
        return expr_registry[value].value(root_ctx), True

    if isinstance(value, dict):
        changed_any = False
        for key, subval in list(value.items()):
            if key in ("expressions", "outletExpressions"):
                continue
            newval, changed = resolve_value(subval, root_ctx, expr_registry)
            value[key] = newval
            changed_any = changed_any or changed
        return value, changed_any

    if isinstance(value, list):
        changed_any = False
        for i, item in enumerate(value):
            newval, changed = resolve_value(item, root_ctx, expr_registry)
            value[i] = newval
            changed_any = changed_any or changed
        return value, changed_any

    return value, False


def resolve_expressions_in_context(ctx, max_passes=10):
    expr_registry = ctx.get("expressions", {})

    for _ in range(max_passes):
        _, changed = resolve_value(ctx, ctx, expr_registry)
        if not changed:
            break
    else:
        raise RuntimeError("Expression resolution exceeded max_passes")

    return ctx


def evaluate_named_expressions(expr_registry, context):
    results = {}
    pending = dict(expr_registry)

    for _ in range(20):
        changed = False

        for name in list(pending.keys()):
            expr = pending[name]
            extended_context = dict(context)
            extended_context.update(results)

            try:
                results[name] = expr.value(extended_context)
                del pending[name]
                changed = True
            except (KeyError, ValueError, TypeError):
                pass

        if not changed:
            break

    if pending:
        raise RuntimeError(f"Could not resolve expressions: {list(pending.keys())}")

    return results


def set_by_dotted_path(data, path, value):
    parts = path.split(".")
    node = data

    for part in parts[:-1]:
        if part not in node:
            raise KeyError(f"Unknown path while setting value: {path}")
        node = node[part]

    last = parts[-1]
    if last not in node:
        raise KeyError(f"Unknown final key while setting value: {path}")
    node[last] = value


def get_by_dotted_path(data, path):
    parts = path.split(".")
    node = data

    for part in parts:
        if not isinstance(node, dict):
            raise KeyError(f"Cannot descend into non-dict for path: {path}")
        if part not in node:
            raise KeyError(f"Unknown path: {path}")
        node = node[part]

    return node


def apply_particle_to_context(ctx, particle, parameter_defs):
    for x, param in zip(particle, parameter_defs):
        set_by_dotted_path(ctx, param["key"], float(x))


def extract_objectives_for_pso(result_ctx, output_defs):
    values = []

    for outdef in output_defs:
        key = outdef["key"]
        goal = str(outdef.get("goal", "minimize")).lower()

        if key in result_ctx.get("derived", {}):
            value = float(result_ctx["derived"][key])
            print("derived source:", key, value)

        elif key == "expr3":
            inlet_so2 = float(result_ctx["inlet"]["specie"]["so2"])
            outlet_so2 = float(result_ctx["outlet"]["specie"]["so2"])
            value = (inlet_so2 - outlet_so2) / inlet_so2
            print("expr3 inputs:", inlet_so2, outlet_so2, value)

        elif key == "expr4":
            inlet_t = float(result_ctx["inlet"]["temperature"])
            outlet_t = float(result_ctx["outlet"]["temperature"])
            value = inlet_t - outlet_t
            print("expr4 inputs:", inlet_t, outlet_t, value)

        else:
            value = float(get_by_dotted_path(result_ctx, key))
            print("path source:", key, value)

        pso_value = -value if goal == "maximize" else value
        print(f"objective {key}: raw={value}, goal={goal}, pso={pso_value}")
        values.append(pso_value)

    return values


def generate_plots(slv, species, case_dir, plotting_cfg):
    """
    Drive ReactorPlotter based on a plotting config block, e.g.:

    plotting:
      enabled: true
      temperature: true
      species: true
      profiles: true
      reaction_rates: true
      heat_source: true
      species_subset: [SO2, O2, SO3]
      dpi: 200
    """
    if plotting_cfg and not plotting_cfg.get("enabled", True):
        return

    dpi = int(plotting_cfg.get("dpi", 200)) if plotting_cfg else 200
    subset = plotting_cfg.get("species_subset") if plotting_cfg else None

    plotter = ReactorPlotter(slv)
    case_dir = Path(case_dir)

    def wants(flag_name, default=True):
        if not plotting_cfg:
            return default
        return bool(plotting_cfg.get(flag_name, default))

    if wants("temperature"):
        plotter.save_temperature(case_dir / "temperature.png", dpi=dpi)

    if wants("species"):
        if subset and hasattr(plotter, "save_species_subset"):
            plotter.save_species_subset(case_dir / "species.png", names=subset, dpi=dpi)
        else:
            plotter.save_species(case_dir / "species.png", dpi=dpi)

    if wants("profiles"):
        plotter.save_all(case_dir / "profiles.png", dpi=dpi)

    if wants("reaction_rates", default=False) and hasattr(plotter, "save_reaction_rates"):
        plotter.save_reaction_rates(case_dir / "reaction_rates.png", dpi=dpi)

    if wants("heat_source", default=False) and hasattr(plotter, "save_heat_source"):
        plotter.save_heat_source(case_dir / "heat_source.png", dpi=dpi)

    if wants("concentrations", default=False) and hasattr(plotter, "save_concentrations"):
        plotter.save_concentrations(case_dir / "concentrations.png", dpi=dpi)


def run_case(case_ctx, case_dir):
    slv, species = build_reactor_from_context(case_ctx)
    slv.initializeCase()

    solver_cfg = case_ctx["solver"]
    urf = solver_cfg["underRelaxationFactors"]

    outlet_obj = slv.steadyState(
        maxiter=int(solver_cfg.get("maxIter", 1000)),
        relaxationFactorSpecie=float(urf["species"]),
        relaxationFactorTemperature=float(urf["temperature"]),
        convergenceCriteria=float(solver_cfg["scaledResidual"]),
        temperatureClipLow=float(solver_cfg["temperatureClipLow"]),
        temperatureClipHigh=float(solver_cfg["temperatureClipHigh"]),
    )

    if outlet_obj is None:
        outlet_obj = Outlet.fromSolver(slv)

    outlet_data = outlet_obj.asDict(species=species)

    _save_yaml(
        Path(case_dir) / "reactorDebug.yaml",
        {
            "mesh": {
                "n_cells": int(slv.mesh.n_cells),
                "length": float(slv.mesh.length),
            },
            "outlet": outlet_data,
        },
    )

    plotting_cfg = case_ctx.get("plotting", {})
    generate_plots(slv, species, case_dir, plotting_cfg)

    return {
        "temperature": float(outlet_data["temperature"]),
        "specie": {k: float(v) for k, v in outlet_data["speciesMassFractions"].items()},
        "density": float(outlet_data["density"]),
        "velocity": float(outlet_data["velocity"]),
        "massFlowrate": float(outlet_data["massFlowrate"]),
        "concentrations": {k: float(v) for k, v in outlet_data["concentrations"].items()},
    }


def make_pso_config_from_context(ctx):
    pso_block = ctx["pso"]["pso"]
    parameter_defs = ctx["pso"]["parameters"]
    output_defs = ctx["pso"]["outputs"]
    constraints = ctx["pso"].get("constraints", {})

    x_lb = [p["bounds"][0] for p in parameter_defs]
    x_ub = [p["bounds"][1] for p in parameter_defs]

    linear = constraints.get("linear", [])
    constr_matrix = [c["A"] for c in linear] if linear else []
    constr_lb = [c["lb"] for c in linear] if linear else []
    constr_ub = [c["ub"] for c in linear] if linear else []

    return PSOConfig(
        h_factor=pso_block["hfactor"],
        max_iter=pso_block["maxiter"],
        n_params=len(parameter_defs),
        n_responses=len(output_defs),
        t_neighbors=pso_block["tneighbors"],
        w_init=pso_block["winit"],
        w_finish=pso_block["wfinish"],
        c1_init=pso_block["c1init"],
        c1_finish=pso_block["c1finish"],
        c2_init=pso_block["c2init"],
        c2_finish=pso_block["c2finish"],
        v_max_factor=pso_block["vmaxfactor"],
        x_lb=x_lb,
        x_ub=x_ub,
        constr_matrix=constr_matrix,
        constr_lb=constr_lb,
        constr_ub=constr_ub,
    )


def main():
    base_ctx = build_context("config", UserExpression)
    base_ctx = resolve_expressions_in_context(base_ctx)

    pso_cfg = make_pso_config_from_context(base_ctx)
    print(pso_cfg)

    def objective_function(particle, iteration):
        case_ctx = copy.deepcopy(base_ctx)

        apply_particle_to_context(
            case_ctx,
            particle,
            case_ctx["pso"]["parameters"],
        )
        print("applied inlet.so2:", case_ctx["inlet"]["specie"]["so2"])
        print("applied inlet.temperature:", case_ctx["inlet"]["temperature"])
        case_ctx = resolve_expressions_in_context(case_ctx)

        study_name = case_ctx["pso"]["study"]["name"]
        case_dir = make_case_dir("cases", study_name)

        _save_yaml(case_dir / "caseSetup.yaml", make_serializable_context(case_ctx))

        reactor_result = run_case(case_ctx, case_dir)

        result_ctx = copy.deepcopy(make_serializable_context(case_ctx))
        result_ctx["outlet"] = reactor_result

        outlet_expr_values = evaluate_named_expressions(
            case_ctx.get("outletExpressions", {}),
            result_ctx,
        )

        result_ctx["derived"] = outlet_expr_values

        _save_yaml(
            case_dir / "outlet.yaml",
            {
                "outlet": reactor_result,
                "derived": outlet_expr_values,
            },
        )
        print("derived =", outlet_expr_values)
        print("output defs =", case_ctx["pso"]["outputs"])

        return extract_objectives_for_pso(result_ctx, case_ctx["pso"]["outputs"])

    optimizer = PSOOptimizer.from_random(pso_cfg, objective_function=objective_function)
    swarm = optimizer.run()

    best_particle = swarm.global_best_position
    print("Optimization finished.")
    print("Best particle:", best_particle)


if __name__ == "__main__":
    main()