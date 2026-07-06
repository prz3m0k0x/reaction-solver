
from dataclasses import dataclass, field
from typing import Callable
import math
import numpy as np
import pathlib


@dataclass
class PSOConfig:
    h_factor: int = 25
    max_iter: int = 15
    n_params: int = 2
    n_responses: int = 1
    t_neighbors: int = 5

    w_init: float = 0.7
    c1_init: float = 0.6
    c2_init: float = 0.5
    w_finish: float = 0.4
    c1_finish: float = 0.8
    c2_finish: float = 0.6
    v_max_factor: float = 0.2

    x_lb: list = field(default_factory=list)
    x_ub: list = field(default_factory=list)

    constr_matrix: list = field(default_factory=list)
    constr_lb: list = field(default_factory=list)
    constr_ub: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x_lb = np.full(self.n_params, -np.inf, dtype=float) if len(self.x_lb) == 0 else np.array(self.x_lb, dtype=float)
        self.x_ub = np.full(self.n_params, np.inf, dtype=float) if len(self.x_ub) == 0 else np.array(self.x_ub, dtype=float)

        self.constr_lb = np.array(self.constr_lb, dtype=float) if len(self.constr_lb) > 0 else np.array([], dtype=float)
        self.constr_ub = np.array(self.constr_ub, dtype=float) if len(self.constr_ub) > 0 else np.array([], dtype=float)
        self.constr_matrix = np.array(self.constr_matrix, dtype=float) if len(self.constr_matrix) > 0 else np.empty((0, self.n_params), dtype=float)

        if self.x_lb.shape != (self.n_params,):
            raise ValueError("x_lb must have length n_params")
        if self.x_ub.shape != (self.n_params,):
            raise ValueError("x_ub must have length n_params")

        if self.constr_matrix.shape[0] > 0:
            if self.constr_matrix.shape[1] != self.n_params:
                raise ValueError("constr_matrix must have n_params columns")
            n_constr = self.constr_matrix.shape[0]
            if self.constr_lb.size not in (0, n_constr):
                raise ValueError("constr_lb must have one value per constraint")
            if self.constr_ub.size not in (0, n_constr):
                raise ValueError("constr_ub must have one value per constraint")

    def pop_size(self) -> int:
        if self.n_responses == 1:
            return self.h_factor
        if self.h_factor <= self.n_responses:
            raise ValueError("h_factor must be greater than number of responses")
        return math.comb(self.h_factor + self.n_responses - 1, self.n_responses - 1)

    @staticmethod
    def das_dennis_weights(m: int, H: int) -> np.ndarray:
        def enumerate_weights(m_, H_, current=None):
            current = [] if current is None else current
            if m_ == 1:
                yield current + [H_]
            else:
                for i in range(H_ + 1):
                    yield from enumerate_weights(m_ - 1, H_ - i, current + [i])

        return np.array(list(enumerate_weights(m, H)), dtype=float) / H

    @property
    def algorithm_velocity_parameters(self) -> np.ndarray:
        return np.linspace(
            start=[self.w_init, self.c1_init, self.c2_init],
            stop=[self.w_finish, self.c1_finish, self.c2_finish],
            num=self.max_iter,
        )


class Swarm:
    def __init__(self, config: PSOConfig, initial_particles: np.ndarray, initial_gains: np.ndarray):
        self.config = config
        self.particles = initial_particles.copy()
        self.v_max = config.v_max_factor * (np.array(config.x_ub) - np.array(config.x_lb))[:, np.newaxis]

        self.velocity = np.random.uniform(-1.0, 1.0, self.particles.shape) * self.v_max

        pop_size = self.particles.shape[1]
        self.single_objective = config.n_responses == 1

        if self.single_objective:
            # No weight-vector decomposition needed; every particle shares
            # the same global-best neighborhood (classic gbest PSO).
            self.weights = None
            self.neighbors = np.tile(np.arange(pop_size), (pop_size, 1))
        else:
            self.weights = config.das_dennis_weights(m=initial_gains.shape[0], H=config.h_factor)
            diff = self.weights[:, None, :] - self.weights[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)
            T = min(config.t_neighbors, dist_matrix.shape[0])
            self.neighbors = np.argsort(dist_matrix, axis=1)[:, :T]

        self.z_ref = np.min(initial_gains, axis=1)
        self.z_nad = np.max(initial_gains, axis=1)

        self.pbest_positions = initial_particles.copy()
        self.pbest_gains = initial_gains.copy()
        self.pbest_scalars = self._tcheby_scalars(initial_gains, self.particles)

        self.gbest_positions = np.zeros_like(initial_particles)
        self.gbest_gains = np.zeros_like(initial_gains)
        self._update_neighborhood_bests()

    @property
    def global_best_position(self) -> np.ndarray:
        return self.pbest_positions[:, np.argmin(self.pbest_scalars)].copy()

    def _penalty(self, particles: np.ndarray) -> np.ndarray:
        lb = np.array(self.config.x_lb)[:, np.newaxis]
        ub = np.array(self.config.x_ub)[:, np.newaxis]

        lb_viol = np.maximum(0, lb - particles) ** 2
        ub_viol = np.maximum(0, particles - ub) ** 2
        bound_penalty = np.sum(lb_viol + ub_viol, axis=0)

        if self.config.constr_matrix.shape[0] == 0:
            return bound_penalty

        Ax = self.config.constr_matrix @ particles
        clb = self.config.constr_lb[:, np.newaxis]
        cub = self.config.constr_ub[:, np.newaxis]
        c_viol = np.maximum(0, clb - Ax) ** 2 + np.maximum(0, Ax - cub) ** 2
        constr_penalty = np.sum(c_viol, axis=0)
        return bound_penalty + constr_penalty

    def _update_neighborhood_bests(self) -> None:
        for i, neighbors in enumerate(self.neighbors):
            best_idx = neighbors[np.argmin(self.pbest_scalars[neighbors])]
            self.gbest_positions[:, i] = self.pbest_positions[:, best_idx]
            self.gbest_gains[:, i] = self.pbest_gains[:, best_idx]

    def _tcheby_scalars(self, gains: np.ndarray, particles: np.ndarray) -> np.ndarray:
        if self.single_objective:
            # Plain (unweighted) objective value, still normalized by the
            # running ideal/nadir range so it stays on a comparable scale
            # with the penalty term.
            scale = self.z_nad[0] - self.z_ref[0] if abs(self.z_nad[0] - self.z_ref[0]) > 1e-8 else 1.0
            tcheby = np.abs(gains[0] - self.z_ref[0]) / scale
        else:
            scale = np.where(np.abs(self.z_nad - self.z_ref) > 1e-8, self.z_nad - self.z_ref, 1.0)
            deviations = np.abs(gains - self.z_ref[:, np.newaxis]) / scale[:, np.newaxis]
            weighted = self.weights.T * deviations
            tcheby = np.max(weighted, axis=0)
        return tcheby + self._penalty(particles)

    def step(self, w: float, c1: float, c2: float) -> None:
        n_params, pop_size = self.particles.shape
        r1 = np.random.uniform(0, 1, (n_params, pop_size))
        r2 = np.random.uniform(0, 1, (n_params, pop_size))

        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.pbest_positions - self.particles)
            + c2 * r2 * (self.gbest_positions - self.particles)
        )
        self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)
        self.particles += self.velocity

        lb = np.array(self.config.x_lb)[:, np.newaxis]
        ub = np.array(self.config.x_ub)[:, np.newaxis]
        self.particles = np.clip(self.particles, lb, ub)

    def update_bests(self, new_gains: np.ndarray) -> None:
        self.z_ref = np.minimum(self.z_ref, np.min(new_gains, axis=1))
        self.z_nad = np.maximum(self.z_nad, np.max(new_gains, axis=1))

        # z_ref / z_nad just changed, so pbest_scalars computed on an older
        # normalization is stale. Recompute it on the *current* pbest_gains
        # under the updated ideal/nadir point before comparing.
        self.pbest_scalars = self._tcheby_scalars(self.pbest_gains, self.pbest_positions)

        new_scalars = self._tcheby_scalars(new_gains, self.particles)
        improved = new_scalars < self.pbest_scalars

        self.pbest_positions[:, improved] = self.particles[:, improved]
        self.pbest_gains[:, improved] = new_gains[:, improved]
        self.pbest_scalars[improved] = new_scalars[improved]
        self._update_neighborhood_bests()


class HistoryLogger:
    def __init__(self, config: PSOConfig):
        self.config = config
        self.best_gain_history = []
        self.particle_history = np.zeros((config.n_params, config.pop_size(), config.max_iter + 1))

    def log(self, iteration: int, particles: np.ndarray, best_gain: float) -> None:
        self.particle_history[:, :, iteration] = particles
        self.best_gain_history.append(best_gain)
        print(f"Iter {iteration:02d} best_gain={best_gain:.6f}")

    def save(self, path: pathlib.Path) -> None:
        np.save(path / "particle_history.npy", self.particle_history)
        np.savetxt(path / "gain_history.csv", np.array(self.best_gain_history), delimiter=";")


class PSOOptimizer:
    def __init__(self, config: PSOConfig, objective_function: Callable[[np.ndarray, int], np.ndarray], initial_particles: np.ndarray):
        self.config = config
        self.objective_function = objective_function

        initial_gains = self._evaluate(initial_particles, iteration=0)
        self.swarm = Swarm(config, initial_particles, initial_gains)
        self.logger = HistoryLogger(config)
        self.logger.log(0, self.swarm.particles, float(np.min(self.swarm.pbest_scalars)))

    @classmethod
    def from_random(cls, config: PSOConfig, objective_function: Callable[[np.ndarray, int], np.ndarray]) -> "PSOOptimizer":
        lb = np.array(config.x_lb)[:, np.newaxis]
        ub = np.array(config.x_ub)[:, np.newaxis]
        pop_size = config.pop_size()
        particles = np.random.uniform(0, 1, (config.n_params, pop_size)) * (ub - lb) + lb
        return cls(config, objective_function, particles)

    def _evaluate(self, particles: np.ndarray, iteration: int = 0) -> np.ndarray:
        gains = []
        for i in range(particles.shape[1]):
            y = self.objective_function(particles[:, i], iteration)
            gains.append(np.asarray(y, dtype=float))
        return np.column_stack(gains)

    def run(self) -> Swarm:
        algo = self.config.algorithm_velocity_parameters

        for j in range(self.config.max_iter):
            w, c1, c2 = algo[j]
            self.swarm.step(w, c1, c2)
            new_gains = self._evaluate(self.swarm.particles, iteration=j + 1)
            self.swarm.update_bests(new_gains)
            self.logger.log(j + 1, self.swarm.particles, float(np.min(self.swarm.pbest_scalars)))

        return self.swarm