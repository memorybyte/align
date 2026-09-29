import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from onpolicy.envs.gym_pybullet_drones.envs.BaseAviary import BaseAviary
from onpolicy.envs.gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from onpolicy.envs.gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 max_neighbors: int = 3,
                 neighbour_radius: float = 0.0,
                 min_dynamic_neighbours: int = 1,
                 max_dynamic_neighbours: int = 8,
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)
        max_neighbors : int, optional
            Maximum nearest neighbors per drone to include in KIN observation.
            Used when dynamic neighbors is disabled (neighbour_radius=0).
        neighbour_radius : float, optional
            Radius (meters) for dynamic neighbor selection.
            0 disables dynamic neighbors and falls back to fixed k-nearest.
        min_dynamic_neighbours : int, optional
            Minimum neighbor slots always filled (closest are always included
            even if outside radius).
        max_dynamic_neighbours : int, optional
            Maximum neighbor slots (defines obs_dim when dynamic neighbors is
            enabled). Remaining slots beyond actual neighbors are zero-padded.

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.MAX_NEIGHBORS = max(0, int(max_neighbors))
        self.NEIGHBOUR_RADIUS = float(neighbour_radius)
        self.MIN_DYNAMIC_NEIGHBOURS = max(0, int(min_dynamic_neighbours))
        self.MAX_DYNAMIC_NEIGHBOURS = max(1, int(max_dynamic_neighbours))
        self.USE_DYNAMIC_NEIGHBOURS = self.NEIGHBOUR_RADIUS > 0.0
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
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
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES, obs_dim) depending on the observation type.
            
        For KIN observation type with VEL action type:
            obs_dim = 9 + 6 * k
            - Own state: x, y, z, vx, vy, vz (6 dims)
            - Relative target position: dx, dy, dz (3 dims)
            - Neighbor states: Δx, Δy, Δz, Δvx, Δvy, Δvz for k neighbors (6 dims each)

            When dynamic neighbors is enabled (neighbour_radius > 0):
                k = min(MAX_DYNAMIC_NEIGHBOURS, NUM_DRONES - 1)
                Unused slots are zero-padded at runtime.
            When disabled:
                k = min(MAX_NEIGHBORS, NUM_DRONES - 1)

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE: own_state(6) + rel_target(3) + neighbors(6*k)
            lo = -np.inf
            hi = np.inf
            if self.USE_DYNAMIC_NEIGHBOURS:
                k = min(self.MAX_DYNAMIC_NEIGHBOURS, self.NUM_DRONES - 1)
            else:
                k = min(self.MAX_NEIGHBORS, self.NUM_DRONES - 1)
            obs_dim = 6 + 3 + 6 * k
            obs_lower_bound = np.array([[lo] * obs_dim for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi] * obs_dim for i in range(self.NUM_DRONES)])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES, obs_dim) depending on the observation type.
            
        For KIN observation type:
            obs_dim = 6 + 3 + 6*k, where k = min(MAX_NEIGHBORS, NUM_DRONES - 1)
            Structure per drone:
            - Own state: [x, y, z, vx, vy, vz] (6 dims)
            - Relative target: [dx, dy, dz] (3 dims)  
            - Neighbor states: [x, y, z, vx, vy, vz] for k nearest neighbors (6 dims each)

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS: own_state(6) + rel_target(3) + neighbors(6*k)
            #### Dynamic: k = min(MAX_DYNAMIC_NEIGHBOURS, NUM_DRONES-1)
            ####   - radius filter + min/max clamping + zero-pad
            #### Fixed:   k = min(MAX_NEIGHBORS, NUM_DRONES-1)
            
            # Get all drone states first
            all_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            positions = all_states[:, 0:3]
            velocities = all_states[:, 10:13]
            
            # Get target positions (set by subclass like MultiHoverAviary)
            if hasattr(self, 'TARGET_POS'):
                target_pos = self.TARGET_POS
            else:
                # Default: target is 1m above initial position
                target_pos = self.INIT_XYZS + np.array([[0, 0, 1.0] for _ in range(self.NUM_DRONES)])
            
            own_state = np.concatenate([positions, velocities], axis=1)  # (N, 6)
            rel_target = target_pos - positions  # (N, 3)

            if self.USE_DYNAMIC_NEIGHBOURS:
                # Dynamic neighbor selection with radius filter
                k = min(self.MAX_DYNAMIC_NEIGHBOURS, self.NUM_DRONES - 1)
                obs_dim = 6 + 3 + 6 * k

                if k > 0 and self.NUM_DRONES > 1:
                    # Pairwise relative states
                    rel_pos_all = positions[None, :, :] - positions[:, None, :]   # (N, N, 3)
                    rel_vel_all = velocities[None, :, :] - velocities[:, None, :] # (N, N, 3)
                    dist_sq = np.sum(rel_pos_all * rel_pos_all, axis=2)           # (N, N)
                    np.fill_diagonal(dist_sq, np.inf)

                    radius_sq = self.NEIGHBOUR_RADIUS ** 2
                    min_k = min(self.MIN_DYNAMIC_NEIGHBOURS, self.NUM_DRONES - 1)

                    # Sort all other drones by distance for each drone
                    sorted_idx = np.argsort(dist_sq, axis=1)  # (N, N-1+) indices sorted by distance
                    sorted_dist_sq = np.take_along_axis(dist_sq, sorted_idx, axis=1)

                    # Initialize zero-padded neighbor features
                    neighbor_features = np.zeros((self.NUM_DRONES, 6 * k), dtype=np.float32)
                    row_idx = np.arange(self.NUM_DRONES)

                    for i in range(self.NUM_DRONES):
                        # Count how many are within radius
                        in_radius = int(np.sum(sorted_dist_sq[i, :] <= radius_sq))
                        # Clamp between min and max
                        n_actual = max(min_k, min(in_radius, k))
                        # Ensure we don't exceed available drones
                        n_actual = min(n_actual, self.NUM_DRONES - 1)

                        if n_actual > 0:
                            nn_idx = sorted_idx[i, :n_actual]  # closest n_actual drones
                            nn_rel_pos = rel_pos_all[i, nn_idx, :]  # (n_actual, 3)
                            nn_rel_vel = rel_vel_all[i, nn_idx, :]  # (n_actual, 3)
                            feats = np.concatenate([nn_rel_pos, nn_rel_vel], axis=1).flatten()  # (n_actual*6,)
                            neighbor_features[i, :len(feats)] = feats
                else:
                    neighbor_features = np.zeros((self.NUM_DRONES, 0), dtype=np.float32)

            else:
                # --- Fixed k-nearest neighbor selection (original behavior) ---
                max_neighbors = min(self.MAX_NEIGHBORS, self.NUM_DRONES - 1)
                obs_dim = 6 + 3 + 6 * max_neighbors

                if max_neighbors > 0:
                    # Pairwise relative states from each drone i to each drone j.
                    rel_pos_all = positions[None, :, :] - positions[:, None, :]   # (N, N, 3)
                    rel_vel_all = velocities[None, :, :] - velocities[:, None, :] # (N, N, 3)
                    dist_sq = np.sum(rel_pos_all * rel_pos_all, axis=2)           # (N, N)
                    np.fill_diagonal(dist_sq, np.inf)

                    # Select nearest neighbors with deterministic distance ordering.
                    if self.NUM_DRONES - 1 > max_neighbors:
                        nn_idx = np.argpartition(dist_sq, kth=max_neighbors - 1, axis=1)[:, :max_neighbors]
                        nn_dist_sq = np.take_along_axis(dist_sq, nn_idx, axis=1)
                        nn_order = np.argsort(nn_dist_sq, axis=1)
                        nn_idx = np.take_along_axis(nn_idx, nn_order, axis=1)
                    else:
                        nn_idx = np.argsort(dist_sq, axis=1)[:, :max_neighbors]

                    row_idx = np.arange(self.NUM_DRONES)[:, None]
                    nn_rel_pos = rel_pos_all[row_idx, nn_idx, :]  # (N, k, 3)
                    nn_rel_vel = rel_vel_all[row_idx, nn_idx, :]  # (N, k, 3)
                    neighbor_features = np.concatenate([nn_rel_pos, nn_rel_vel], axis=2).reshape(self.NUM_DRONES, 6 * max_neighbors)
                else:
                    neighbor_features = np.zeros((self.NUM_DRONES, 0), dtype=np.float32)

            obs = np.concatenate([own_state, rel_target, neighbor_features], axis=1).astype(np.float32)
            return obs
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
