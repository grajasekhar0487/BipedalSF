# Copyright (c) 2026, BipedalSF Project.
# SPDX-License-Identifier: BSD-3-Clause

"""SF-TRON2A articulation configuration using the supplied robot USD."""

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .. import BIPEDALSF_ASSETS_DIR

SF_TRON2A_USD = BIPEDALSF_ASSETS_DIR / "robots" / "SF_TRON2A" / "usd" / "robot.usd"

SF_TRON2A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SF_TRON2A_USD),
        activate_contact_sensors=True,
        variants={"Physics": "PhysX", "Robot": "Robot", "Sensor": "Sensors"},
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "high_torque_joints": ImplicitActuatorCfg(
            joint_names_expr=["proximal_pitch_.*_Joint", "proximal_roll_.*_Joint", "knee_.*_Joint"],
            joint_effort_limit=150.0,
            joint_velocity_limit=18.3,
            stiffness=80.0,
            damping=5.0,
        ),
        "low_torque_joints": ImplicitActuatorCfg(
            joint_names_expr=["proximal_yaw_.*_Joint", "ankle_pitch_.*_Joint"],
            joint_effort_limit=60.0,
            joint_velocity_limit=22.52,
            stiffness=45.0,
            damping=3.0,
        ),
    },
)
"""Biped articulation with the ten revolute joints defined by ``robot.usd``."""
