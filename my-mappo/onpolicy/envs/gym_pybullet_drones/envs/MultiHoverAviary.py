import numpy as np

from onpolicy.envs.gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from onpolicy.envs.gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def _center_template(template: np.ndarray) -> np.ndarray:
    template = template.copy()
    template -= template.mean(axis=0)
    return template


def _build_plane_template(num_drones, spacing=1.0):
    side = int(np.ceil(np.sqrt(num_drones)))
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append([i * spacing, j * spacing, 0.0])
            if len(coords) >= num_drones:
                return _center_template(np.array(coords, dtype=np.float32))
    return _center_template(np.array(coords, dtype=np.float32))


def _build_cube_template(num_drones, spacing=1.0):
    side = int(np.ceil(num_drones ** (1.0 / 3.0)))
    coords = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                coords.append([x * spacing, y * spacing, z * spacing])

    # Build full side^3 lattice, then select exactly num_drones points.
    lattice = np.array(coords, dtype=np.float32)
    lattice_center = np.array([
        (side - 1) * spacing / 2.0,
        (side - 1) * spacing / 2.0,
        (side - 1) * spacing / 2.0,
    ], dtype=np.float32)

    # Decreasing order of distance from lattice center.
    dist_sq = np.sum((lattice - lattice_center) ** 2, axis=1)
    order = np.argsort(-dist_sq, kind='mergesort')
    selected = lattice[order[:num_drones]]

    return _center_template(selected)


def _build_sphere_template(num_drones, spacing=1.0):
    if num_drones == 1:
        return np.zeros((1, 3), dtype=np.float32)

    # Area-per-point heuristic to keep spacing scale roughly consistent.
    radius = spacing * np.sqrt(max(num_drones, 2) / (4.0 * np.pi))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    coords = np.zeros((num_drones, 3), dtype=np.float32)

    for i in range(num_drones):
        y = 1.0 - 2.0 * i / (num_drones - 1)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = i * golden_angle
        coords[i, 0] = radius * r * np.cos(theta)
        coords[i, 1] = radius * r * np.sin(theta)
        coords[i, 2] = radius * y

    return _center_template(coords)


def _build_pyramid_template(num_drones, spacing=1.0):
    num_layers = 0
    total = 0
    while total < num_drones:
        total += (num_layers + 1) ** 2
        num_layers += 1

    full_layers_count = num_layers - 1
    full_layers_total = sum((k + 1) ** 2 for k in range(full_layers_count))
    remainder = num_drones - full_layers_total

    def layer_coords(layer, z):
        """Generate centered (x, y, z) coords for a given layer index."""
        side = layer + 1
        offset = (side - 1) * spacing / 2.0
        return [
            [xi * spacing - offset, yi * spacing - offset, z]
            for xi in range(side)
            for yi in range(side)
        ]

    coords = []

    for layer in range(full_layers_count):
        z = (num_layers - 1 - layer) * spacing
        coords.extend(layer_coords(layer, z))

    last_layer_pts = np.array(
        layer_coords(num_layers - 1, z=0.0),
        dtype=np.float32
    )

    dist_sq = np.sum(last_layer_pts[:, :2] ** 2, axis=1)
    order = np.argsort(-dist_sq, kind='mergesort')
    selected_last = last_layer_pts[order[:remainder]]

    coords.extend(selected_last.tolist())

    selected = np.array(coords, dtype=np.float32)
    return _center_template(selected)


def _build_formation_template(num_drones, spacing=1.0, formation_type="polygon"):
    """
    Build a fixed formation template centered at the origin.
    
    For formation_type="line": straight line along x-axis for any N.
    For formation_type="plane": 2D grid on XY plane.
    For formation_type="cube": 3D cubic lattice.
    For formation_type="sphere": quasi-uniform points on sphere surface.
    For formation_type="pyramid": stepped square pyramid lattice.
    For formation_type="polygon":
    - 2 drones: line along x-axis.
    - 3 drones: equilateral triangle in the XY plane.
    - N drones: regular polygon in the XY plane.
    
    Parameters
    ----------
    num_drones : int
        Number of drones.
    spacing : float
        Inter-drone distance (meters).
    formation_type : str
        One of: "polygon", "line", "plane", "cube", "sphere", "pyramid".
        
    Returns
    -------
    ndarray
        (num_drones, 3) formation template centered at origin (z=0).
    """
    template = np.zeros((num_drones, 3))
    if num_drones == 1:
        return template

    if formation_type == "line":
        # Equally spaced along x-axis and centered at origin.
        x_coords = np.linspace(0.0, spacing * (num_drones - 1), num_drones)
        x_coords -= np.mean(x_coords)
        template[:, 0] = x_coords
        return template

    if formation_type == "plane":
        return _build_plane_template(num_drones, spacing)

    if formation_type == "cube":
        return _build_cube_template(num_drones, spacing)

    if formation_type == "sphere":
        return _build_sphere_template(num_drones, spacing)

    if formation_type == "pyramid":
        return _build_pyramid_template(num_drones, spacing)

    if formation_type != "polygon":
        raise ValueError(
            f"Unsupported formation_type '{formation_type}'. "
            "Use one of: polygon, line, plane, cube, sphere, pyramid."
        )

    if num_drones == 2:
        template[0] = [-spacing / 2, 0, 0]
        template[1] = [ spacing / 2, 0, 0]
    elif num_drones == 3:
        # Equilateral triangle
        template[0] = [0, 0, 0]
        template[1] = [spacing, 0, 0]
        template[2] = [spacing / 2, spacing * np.sqrt(3) / 2, 0]
    else:
        # Regular polygon
        for i in range(num_drones):
            angle = 2 * np.pi * i / num_drones
            template[i] = [spacing * np.cos(angle), spacing * np.sin(angle), 0]

    centroid = template.mean(axis=0)
    template -= centroid
    return template


def _apply_formation(template, center, yaw, perturbation_std=0.0):
    """
    Place a formation template at a given center with a yaw rotation and optional perturbation.
    
    Parameters
    ----------
    template : ndarray
        (N, 3) formation template centered at origin.
    center : ndarray
        (3,) center position [x, y, z].
    yaw : float
        Rotation angle in radians (around z-axis).
    perturbation_std : float
        Std-dev of Gaussian noise added to each drone position (meters).
        
    Returns
    -------
    ndarray
        (N, 3) drone positions.
    """
    # 2D rotation matrix around z-axis
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1],
    ])
    positions = (R @ template.T).T + center
    if perturbation_std > 0:
        positions += np.random.normal(0, perturbation_std, positions.shape)
    # Clamp z to be above ground
    positions[:, 2] = np.clip(positions[:, 2], 0.05, None)
    return positions


class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: formation control with navigation."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 formation_spacing: float=1.0,
                 perturbation_std: float=0.05,
                 arena_xy_bound: float=1.5,
                 arena_z_range: tuple=(0.2, 1.0),
                 target_distance_range: tuple=(0.5, 2.0),
                 formation_type: str = 'polygon',
                 max_neighbors: int = 3,
                 neighbour_radius: float = 0.0,
                 min_dynamic_neighbours: int = 1,
                 max_dynamic_neighbours: int = 8,
                 ):
        """Initialization of a multi-agent RL environment for formation control.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
            If None, positions are generated from the formation template.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        formation_spacing : float
            Inter-drone distance in the formation template (meters).
        perturbation_std : float
            Std-dev of Gaussian noise added to initial positions (meters).
        arena_xy_bound : float
            Max absolute x/y value for random formation center.
        arena_z_range : tuple
            (z_min, z_max) for random formation center height.
        target_distance_range : tuple
            (min, max) distance from initial center to target center.
        formation_type : str
            One of: polygon, line, plane, cube, sphere, pyramid, dynamic.
            If dynamic, each episode samples one from [cube, sphere, pyramid, plane].
        max_neighbors : int
            Maximum nearest neighbors per drone used in KIN observation.
        neighbour_radius : float
            Radius (meters) for dynamic neighbor selection. 0 disables.
        min_dynamic_neighbours : int
            Minimum neighbor slots always filled.
        max_dynamic_neighbours : int
            Maximum neighbor slots (defines obs_dim when dynamic).
        """
        self.EPISODE_LEN_SEC = 8
        
        # Store formation parameters
        self._formation_spacing = formation_spacing
        self._formation_type = formation_type
        self._dynamic_formation_types = ["cube", 
                                         "sphere", 
                                         "pyramid", 
                                         "plane"]
        self._current_episode_formation_type = formation_type
        self._perturbation_std = perturbation_std
        self._arena_xy_bound = arena_xy_bound
        self._arena_z_range = arena_z_range
        self._target_distance_range = target_distance_range

        if self._formation_type == "dynamic":
            self._current_episode_formation_type = np.random.choice(self._dynamic_formation_types)

        # Build episode formation template (relative positions, centered at origin)
        self._formation_template = _build_formation_template(
            num_drones,
            formation_spacing,
            formation_type=self._current_episode_formation_type,
        )
        
        # Generate initial formation-based positions for first episode
        if initial_xyzs is None:
            initial_xyzs = self._generate_formation_positions()
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act,
                         max_neighbors=max_neighbors,
                         neighbour_radius=neighbour_radius,
                         min_dynamic_neighbours=min_dynamic_neighbours,
                         max_dynamic_neighbours=max_dynamic_neighbours,
                         )
        # Override speed limit: 0.05 * MAX_SPEED gives ~0.42 m/s, reduces RPM differentials
        # that cause large tilt from high-speed maneuvers
        if hasattr(self, 'SPEED_LIMIT'):
            self.SPEED_LIMIT = 0.05 * self.MAX_SPEED_KMH * (1000/3600)
        
        # Generate target positions: same formation at a different random center
        self.TARGET_POS = self._generate_target_positions()

    ################################################################################

    def _generate_formation_positions(self):
        """
        Generate drone positions from the formation template with random
        center, random yaw orientation, and small perturbations.
        
        Returns
        -------
        ndarray
            (NUM_DRONES, 3) positions.
        """
        center = np.array([
            np.random.uniform(-self._arena_xy_bound, self._arena_xy_bound),
            np.random.uniform(-self._arena_xy_bound, self._arena_xy_bound),
            np.random.uniform(*self._arena_z_range),
        ])
        # yaw = np.random.uniform(0, 2 * np.pi)
        yaw = 0
        positions = _apply_formation(
            self._formation_template, center, yaw,
            perturbation_std=self._perturbation_std,
        )
        self._current_init_center = center
        self._current_init_yaw = yaw
        return positions

    def _generate_target_positions(self):
        """
        Generate target positions: same formation template placed at a
        random target center (different from initial center).
        
        The formation orientation at the target may also be randomised
        so the policy learns rotation invariance.
        
        Returns
        -------
        ndarray
            (NUM_DRONES, 3) target positions.
        """
        # Random target center at a reachable distance from initial center
        dist = np.random.uniform(*self._target_distance_range)
        angle_xy = np.random.uniform(0, 2 * np.pi)
        z_target = np.random.uniform(*self._arena_z_range)
        
        target_center = np.array([
            self._current_init_center[0] + dist * np.cos(angle_xy),
            self._current_init_center[1] + dist * np.sin(angle_xy),
            z_target,
        ])
        # Clamp target center inside arena
        target_center[0] = np.clip(target_center[0], -self._arena_xy_bound - 1, self._arena_xy_bound + 1)
        target_center[1] = np.clip(target_center[1], -self._arena_xy_bound - 1, self._arena_xy_bound + 1)
        target_center[2] = np.clip(target_center[2], self._arena_z_range[0], self._arena_z_range[1])
        
        # Random yaw at target (may differ from initial yaw for rotation invariance)
        # target_yaw = np.random.uniform(0, 2 * np.pi)
        target_yaw = 0
        
        target_positions = _apply_formation(
            self._formation_template, target_center, target_yaw,
            perturbation_std=0.0,  # No perturbation on targets
        )
        return target_positions

    ################################################################################
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with formation-based initial and target positions.
        
        Each episode:
        0. If formation_type is dynamic, sample one of cube/sphere/pyramid/plane
        1. Random formation center within arena bounds
        2. Random formation yaw orientation
        3. Small perturbation on each drone position
        4. Random target center (same formation shape) at a reachable distance
        """
        if self._formation_type == "dynamic":
            self._current_episode_formation_type = np.random.choice(self._dynamic_formation_types)
            self._formation_template = _build_formation_template(
                self.NUM_DRONES,
                self._formation_spacing,
                formation_type=self._current_episode_formation_type,
            )

        # Re-generate formation-based initial positions
        self.INIT_XYZS = self._generate_formation_positions()
        
        # Re-generate formation-based target positions
        self.TARGET_POS = self._generate_target_positions()
        
        # Call parent reset which will use the updated INIT_XYZS
        return super().reset(seed=seed, options=options)
    
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # Check if ALL drones reached their targets
        # Using 0.05m (5cm) threshold per drone (matching DMPC-Swarm)
        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
            if dist > 0.05:  # 5cm threshold per drone
                return False
        return True  # All drones within 5cm of targets

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 5.0 or abs(states[i][1]) > 5.0 or states[i][2] > 5.0 # Truncate when a drone is too far away
                # or abs(states[i][7]) > 1.3 or abs(states[i][8]) > 1.3 # Truncate when a drone is too tilted (57 deg)
            ):
                # if abs(states[i][0]) > 5.0 or abs(states[i][1]) > 5.0 or states[i][2] > 5.0:
                #     print("Truncated due to leaving arena: step_counter =", self.step_counter, "drone", i)
                # else:
                #     print("Truncated due to tilt: step_counter =", self.step_counter, "drone", i)
                
                # print("Truncated due to safety limit: step_counter =", self.step_counter, "drone", i)
                return True
            
        # BaseAviary computes truncation before incrementing step_counter.
        # Use the projected counter after this control step to avoid a +1 ctrl-step delay.
        next_step_counter = self.step_counter + self.PYB_STEPS_PER_CTRL
        if next_step_counter/self.PYB_FREQ >= self.EPISODE_LEN_SEC:
            # print("Truncated due to time limit: step_counter =", next_step_counter, "time =", next_step_counter/self.PYB_FREQ, "s")
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
