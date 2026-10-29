# Copyright (c) 2026, BipedalSF Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground velocity tracking and balance for the SF-TRON2A biped."""

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from BipedalSF.assets.robots.sf_tron2a import SF_TRON2A_CFG


@configclass
class FlatSceneCfg(MySceneCfg):
    """Simple ground and biped sensors; keep the base velocity scene contract."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.3)),
    )


@configclass
class SFTron2ACommandsCfg(CommandsCfg):
    """Walking commands include zero-speed balance episodes."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.8),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


@configclass
class SFTron2AObservationsCfg(ObservationsCfg):
    """Policy observations for a flat-ground biped, without a terrain scanner."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception and commanded velocity."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SFTron2ARewardsCfg(RewardsCfg):
    """Reward upright balance, command tracking, and alternating steps."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[LR]_Link"),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[LR]_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_[LR]_Link"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    undesired_contacts = None
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class SFTron2ATerminationsCfg(TerminationsCfg):
    """End episodes when the torso falls or the robot tips over."""

    fallen = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.55})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})


@configclass
class SFTron2ACurriculumCfg(CurriculumCfg):
    """No terrain curriculum is needed for the flat plane scene."""

    terrain_levels = None


@configclass
class SFTron2AEnvCfg(LocomotionVelocityRoughEnvCfg):
    """SF-TRON2A flat-ground walking and standing balance environment."""

    scene: FlatSceneCfg = FlatSceneCfg(num_envs=512, env_spacing=2.5)
    observations: SFTron2AObservationsCfg = SFTron2AObservationsCfg()
    commands: SFTron2ACommandsCfg = SFTron2ACommandsCfg()
    rewards: SFTron2ARewardsCfg = SFTron2ARewardsCfg()
    terminations: SFTron2ATerminationsCfg = SFTron2ATerminationsCfg()
    curriculum: SFTron2ACurriculumCfg = SFTron2ACurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = replace(SF_TRON2A_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/[^/]*"
        self.scene.contact_forces.update_period = self.sim.dt
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.actions.joint_pos.scale = 0.35
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.base_com.params["asset_cfg"].body_names = "base_Link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_Link"
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(3.0, 3.0, 2.2), lookat=(0.0, 0.0, 0.8))
