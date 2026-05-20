from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as typing
import pathlib, math
from typing import Callable
from solverReactor import Specie, Reaction, Mixture, domainSetup, Inlet, Zone, Mesh, scalarField, solver


def wrapper(f, param1, param2, i):
    print("wrapper i =", i, "len =", len(param1))
    y1 = np.zeros_like(param1)
    y2 = np.zeros_like(param1)
    for i in range(len(param1)):
        result = f(param1[i], param2[i])
        y1[i] = - result[0]
        y2[i] = - result[1]
    return y1, y2

def runner(param1, param2):
    print("runner start", param1, param2)
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
        ahrreniusPreExponent= 50e10,
        ahrreniusActivationEnergy= 165000.,
        species=species
    )

    z0 = Zone(length=.5, type="reaction")
    z0.zoneAssign(heating=False, reaction=True)
    z0.zoneAssignHeating(0.0)

    z1 = Zone(length=.5, type="heating")
    z1.zoneAssign(heating=True, reaction=False)
    z1.zoneAssignHeating(-50000.0)

    z2 = Zone(length=.5, type="reaction")
    z2.zoneAssign(heating=False, reaction=True)
    z2.zoneAssignHeating(0.0)

    Y_so2 = param1
    Y_so3 = 1e-6
    Y_o2 = 0.21 * (1 - Y_so2 - Y_so3)
    Y_n2 = 1 - Y_so2 - Y_o2 - Y_so3
    
    domain = domainSetup(
        diameter=2.5,
        inletMassFractions=np.array([Y_so2, Y_o2, Y_so3, Y_n2])
    )
    mesh = Mesh(domain=domain, zoneList=[z0, z1, z2], sizing=0.0025)
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
    
    inlet = Inlet(0, 1, param2, [Y_so2, Y_o2, Y_so3, Y_n2])
    mixture = Mixture(
        densityModel="ideal-incompressible-gas",
        densityValue=0.457,
        species=species
    )

    sol = solver(mesh=mesh, mixture=mixture, reaction=reaction, specieFields=specieFields, inlet=inlet)
    sol.initializeCase()

    results = sol.steadyState(max_iter=200, relaxationFactorSpecie=0.1, relaxationFactorTemperature=0.1, convergenceCriteria=1e-5)
    so3_productivity = sol.massFlux * results[2]
    conversion = results[2] * 64. / (Y_so2 * 80.)
    return so3_productivity, conversion

    print("runner end")

def TestFunction(particles: np.ndarray):
    """
    Multi-objective test function.
    Objective 1 : quadratic form, minumim at x=(0,1), f=0
    Objective 2 : Rastrigin function — global min at x=(0,0), f=0

    particles : (n_params, N_PARTICLES)
    returns   : y1 (N_PARTICLES,), y2 (N_PARTICLES,)
    """
    x1 = particles[0, :]
    x2 = particles[1, :]

    #quadratic
    y1 = (x1-3)**2 + (x2-3)**2
    # y2 = (x1+5)**2 + (x2+5)**2
    #  Rastrigin
    A  = 10
    y2 = (2 * A
           + x1**2 - A * np.cos(2 * np.pi * x1)
           + x2**2 - A * np.cos(2 * np.pi * x2))

    return y1, y2


@dataclass
class PSOConfig:
    h_factor    : int = 3
    max_iter    : int = 1
    n_params    : int = 2
    n_responses : int = 2
    t_neighbors : int = 3

    w_init      : float = 0.7
    c1_init     : float = 0.6
    c2_init     : float = 0.5
    w_finish    : float = 0.4
    c1_finish   : float = 0.8
    c2_finish   : float = 0.6
    v_max_factor: float = 0.2

    x_lb : list = field(default_factory=lambda: [])
    x_ub : list = field(default_factory=lambda: [])

    constr_matrix : list = field(default_factory=lambda: [])
    constr_lb : list = field(default_factory=lambda: [])
    constr_ub : list = field(default_factory=lambda: [])
    


    def __post_init__(self) -> None:
        self.x_lb = np.full(self.n_params, -np.inf, dtype=float) if len(self.x_lb) == 0 else np.array(self.x_lb, dtype=float)
        self.x_ub = np.full(self.n_params,  np.inf, dtype=float) if len(self.x_ub) == 0 else np.array(self.x_ub, dtype=float)

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

    def POP_SIZE(self) -> int:
        if self.h_factor <= self.n_responses:
            raise ValueError("h_factor must be greater than number of responses!")
        return math.comb(self.h_factor + self.n_responses - 1, self.n_responses - 1)
    

    @staticmethod
    def das_dennis_weights(m: int, H: int) -> np.ndarray:
        """
        Generate weight matrix using Das Dennis method.
        m : number of objective functions
        H : partition parameter (divisions per axis)
        Returns: np.ndarray of shape (N, m), each row is a weight vector
        """
        def enumerate_weights(m, H, current=[]):
            if m == 1:
                yield current + [H]
            else:
                for i in range(H + 1):
                    yield from enumerate_weights(m - 1, H - i, current + [i])
        
        weights = np.array(list(enumerate_weights(m, H)), dtype=float) / H
        return weights
    
    @property
    def algorithm_velocity_parameters(self) -> np.typing.ArrayLike :
        return np.linspace(start = [self.w_init, self.c1_init, self.c2_init],
                                                         stop  = [self.w_finish, self.c1_finish, self.c2_finish],
                                                         num= self.max_iter)


@dataclass
class Swarm:
    """Owns particle positions, velocities, personal and global bests."""
    def __init__(self, config : PSOConfig, initial_particles: np.ndarray,
                initial_gains: np.ndarray):

        self.config   = config
        self.particles = initial_particles.copy()
        self.velocity  = np.zeros_like(self.particles)
        self.v_max     = config.v_max_factor * (
                            np.array(config.x_ub) - np.array(config.x_lb)
                        )[:, np.newaxis] 

        self.weights = config.das_dennis_weights(
            m=initial_gains.shape[0], H=config.h_factor
        )                                                                   # (N_PARTICLES, n_responses)

        diff           = self.weights[:, None, :] - self.weights[None, :, :]  # (N, N, M)
        dist_matrix    = np.linalg.norm(diff, axis=-1)                        # (N, N)
        T              = min(config.t_neighbors, dist_matrix.shape[0])
        self.neighbors = np.argsort(dist_matrix, axis=1)[:, :T]               # (N_PARTICLES, T)


        self.z_ref = np.min(initial_gains, axis=1)                            # (n_responses,)
        self.z_nad = np.max(initial_gains, axis=1)

        self.pbest_positions = initial_particles.copy()                       # (n_params, N_PARTICLES)
        self.pbest_gains     = initial_gains.copy()                           # (n_responses, N_PARTICLES)
        self.pbest_scalars   = self._tcheby_scalars(initial_gains, self.particles)            # (N_PARTICLES,)

        self.gbest_positions = np.zeros_like(initial_particles)               # (n_params, N_PARTICLES)
        self.gbest_gains     = np.zeros_like(initial_gains)                   # (n_responses, N_PARTICLES)
        self._update_neighborhood_bests()

    @property
    def global_best_position(self) -> np.ndarray:
        """Single best particle across the whole swarm."""
        return self.pbest_positions[:, np.argmin(self.pbest_scalars)].copy()

    def _penalty(self, particles: np.ndarray) -> np.ndarray:
        lb  = np.array(self.config.x_lb)[:, np.newaxis]
        ub  = np.array(self.config.x_ub)[:, np.newaxis]

        lb_viol = np.maximum(0, lb - particles) ** 2          # (n_params, N_PARTICLES)
        ub_viol = np.maximum(0, particles - ub) ** 2
        bound_penalty = np.sum(lb_viol + ub_viol, axis=0)     # (N_PARTICLES,)

        Ax      = self.config.constr_matrix @ particles        # (N_CONSTR, N_PARTICLES)
        clb     = self.config.constr_lb[:, np.newaxis]
        cub     = self.config.constr_ub[:, np.newaxis]
        c_viol  = (np.maximum(0, clb - Ax) ** 2
                + np.maximum(0, Ax  - cub) ** 2)
        constr_penalty = np.sum(c_viol, axis=0)                # (N_PARTICLES,)

        return (bound_penalty + constr_penalty)
    
    def _update_neighborhood_bests(self) -> None:
        """
        For each particle i, find the neighbour with the lowest pbest_scalar
        and set it as that particle's gbest.
        """
        for i, neighbors in enumerate(self.neighbors):
            best_in_neighborhood   = neighbors[np.argmin(self.pbest_scalars[neighbors])]
            self.gbest_positions[:, i] = self.pbest_positions[:, best_in_neighborhood]
            self.gbest_gains[:, i]     = self.pbest_gains[:, best_in_neighborhood]

    def _tcheby_scalars(self, gains: np.ndarray,
                        particles: np.ndarray) -> np.ndarray:
        scale      = np.where(np.abs(self.z_nad - self.z_ref) > 1e-8,
                            self.z_nad - self.z_ref, 1.0)

        deviations = np.abs(gains - self.z_ref[:, np.newaxis]) / scale[:, np.newaxis]
        weighted   = self.weights.T * deviations
        tcheby     = np.max(weighted, axis=0)

        return tcheby + self._penalty(particles)


    def step(self, w: float, c1: float, c2: float) -> None:
        n_params, pop_size = self.particles.shape
        r1 = np.random.uniform(0, 1, (n_params, pop_size))
        r2 = np.random.uniform(0, 1, (n_params, pop_size))

        self.velocity = (
            w  * self.velocity
        + c1 * r1 * (self.pbest_positions - self.particles)
        + c2 * r2 * (self.gbest_positions - self.particles)
        )
        self.velocity  = np.clip(self.velocity, -self.v_max, self.v_max)
        self.particles += self.velocity

    def update_bests(self, new_gains: np.ndarray) -> None:

        self.z_ref = np.minimum(self.z_ref, np.min(new_gains, axis=1))
        self.z_nad = np.maximum(self.z_nad, np.max(new_gains, axis=1))


        new_scalars = self._tcheby_scalars(new_gains, self.particles)

        improved = new_scalars < self.pbest_scalars
        self.pbest_positions[:, improved] = self.particles[:, improved]
        self.pbest_gains[:, improved]     = new_gains[:, improved]
        self.pbest_scalars[improved]      = new_scalars[improved]


        self._update_neighborhood_bests()

class HistoryLogger:

    def __init__(self, config: PSOConfig):
        self.config            = config
        self.best_gain_history = []
        self.particle_history  = np.zeros((config.n_params, config.POP_SIZE(), config.max_iter))

    def log(self, iteration: int, particles: np.ndarray, best_gain: float) -> None:
        self.particle_history[:, :, iteration] = particles
        self.best_gain_history.append(best_gain)
        print(f"Iter {iteration+1:02d}  best_gain={best_gain:.6f}")

    def save(self, path: pathlib.Path) -> None:
        np.save(path / "particle_history.npy", self.particle_history)
        np.savetxt(path / "gain_history.csv",
                   np.array(self.best_gain_history), delimiter=";")
   
class PSOOptimizer:

    def __init__(self, config: PSOConfig, initial_particles: np.ndarray):
        self.config = config

        initial_gains = self._evaluate(initial_particles)               # (n_responses, N_PARTICLES)

        self.swarm  = Swarm(config, initial_particles, initial_gains)
        self.logger = HistoryLogger(config)

    @classmethod
    def from_random(cls, config: PSOConfig) -> "PSOOptimizer":
        """Seeds the swarm with uniformly random particles within bounds."""
        lb        = np.array(config.x_lb)[:, np.newaxis]
        ub        = np.array(config.x_ub)[:, np.newaxis]
        pop_size  = config.POP_SIZE()
        particles = np.random.uniform(0, 1, (config.n_params, pop_size)) * (ub - lb) + lb
        return cls(config, particles)


    def _evaluate(self, particles: np.ndarray, iteration: int = 0) -> np.ndarray:
        y1, y2 = wrapper(runner, particles[0, :], particles[1, :], iteration)
        return np.array([y1, y2])


    def run(self) -> Swarm:
        algo             = self.config.algorithm_velocity_parameters
        particle_history = [self.swarm.particles.copy()]
        gbest_history    = [self.swarm.global_best_position.copy()]

        for j in range(self.config.max_iter):
            w, c1, c2 = algo[j]
            self.swarm.step(w, c1, c2)

            new_gains = self._evaluate(self.swarm.particles, iteration=j)
            self.swarm.update_bests(new_gains)

            self.logger.log(j, self.swarm.particles,
                            float(np.min(self.swarm.pbest_scalars)))
            particle_history.append(self.swarm.particles.copy())
            gbest_history.append(self.swarm.global_best_position.copy())

        # visualize(self.config, particle_history, gbest_history)
        return self.swarm
class VisualizePSO:
    def __init__(self, config: PSOConfig, history: HistoryLogger, swarm: Swarm, optimizer: PSOOptimizer):
        self.gbest_history = history.best_gain_history
        self.particle_history = history.particle_history
        self.optimizer = optimizer
        self.config = config
        self.swarm = swarm

    @staticmethod
    def _pareto_mask(objectives: np.ndarray) -> np.ndarray:
        """
        objectives: shape (n_obj, n_points), minimization assumed
        returns: mask of shape (n_points,), True for non-dominated points
        """
        costs = objectives.T   # shape (n_points, n_obj)
        n = costs.shape[0]
        is_nd = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_nd[i]:
                continue
            dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            dominated[i] = False
            if np.any(dominated):
                is_nd[i] = False

        # Correct non-dominance test
        is_nd[:] = True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    is_nd[i] = False
                    break
        return is_nd

    def plotter(self, plot_type: str = "pareto", axes: typing.ArrayLike = (0, 1), show_history: bool = True):
        """
        Supported plot types:
        - 'design': parameter-space scatter of final particles
        - 'pareto': objective-space scatter of final pbest solutions
        """

        if plot_type == "design":
            a, b = axes
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            if show_history and self.particle_history.shape[2] > 0:
                n_particles = self.particle_history.shape[1]
                n_iter = self.particle_history.shape[2]

                for p in range(n_particles):
                    traj_x = self.particle_history[a, p, :]
                    traj_y = self.particle_history[b, p, :]
                    ax.plot(traj_x, traj_y, color='white', alpha=0.12, linewidth=0.8)

            ax.scatter(
                self.swarm.pbest_positions[a, :],
                self.swarm.pbest_positions[b, :],
                color='cyan',
                edgecolors='white',
                s=45,
                alpha=0.9,
                label='Personal bests'
            )

            ax.scatter(
                self.swarm.gbest_positions[a, :],
                self.swarm.gbest_positions[b, :],
                color='magenta',
                edgecolors='white',
                s=28,
                alpha=0.7,
                label='Neighborhood bests'
            )

            g = self.swarm.global_best_position
            ax.scatter(
                g[a], g[b],
                color='yellow',
                edgecolors='black',
                s=120,
                marker='*',
                label='Global best scalar'
            )

            ax.set_xlabel(f"Parameter {a}", color="white")
            ax.set_ylabel(f"Parameter {b}", color="white")
            ax.set_title("PSO design-space evolution", color="white")
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.35)
            ax.tick_params(colors="white")
            ax.legend(facecolor='#0d0d1a', edgecolor='white', labelcolor='white')
            plt.tight_layout()
            plt.show()

        elif plot_type == "pareto":
            gains = self.swarm.pbest_gains.copy()   # shape (2, N), minimization
            nd_mask = self._pareto_mask(gains)

            y1 = -gains[0, :]   # back to "maximize" interpretation for plotting
            y2 = -gains[1, :]

            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            ax.scatter(
                y1[~nd_mask], y2[~nd_mask],
                color='gray',
                alpha=0.45,
                s=35,
                label='Dominated'
            )

            ax.scatter(
                y1[nd_mask], y2[nd_mask],
                color='lime',
                edgecolors='white',
                s=60,
                alpha=0.95,
                label='Pareto front'
            )

            if np.sum(nd_mask) > 1:
                order = np.argsort(y1[nd_mask])
                ax.plot(
                    y1[nd_mask][order],
                    y2[nd_mask][order],
                    color='lime',
                    linewidth=1.4,
                    alpha=0.85
                )

            ax.set_xlabel("SO3 productivity", color="white")
            ax.set_ylabel("Conversion", color="white")
            ax.set_title("Final Pareto set", color="white")
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.35)
            ax.tick_params(colors="white")
            ax.legend(facecolor='#0d0d1a', edgecolor='white', labelcolor='white')
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("plot_type must be 'design' or 'pareto'")

# Execution
config = PSOConfig(
    h_factor= 25,
    max_iter= 15,
    n_params    = 2,
    n_responses = 2,
    x_lb        = [0.1, 600],  # Fixed bounds to be identical shape
    x_ub        = [0.2, 800]
)

optimizer = PSOOptimizer.from_random(config)
swarm = optimizer.run()

visualizer = VisualizePSO(config, optimizer.logger, swarm, optimizer)
visualizer.plotter(plot_type="design")
visualizer.plotter(plot_type="pareto")