from typing_extensions import *

import sapien
import torch
import numpy as np
import random

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation


def get_box(
    scene: ManiSkillScene | sapien.ArticulationBuilder,
    size: Tuple[float, float, float],
    thickness: float,
    name: str,
    radius: float = 0.02,
    center: Optional[Tuple[float, float, float]] = None,
    color: Optional[Tuple[float, float, float, float]] = None,
):
    if center is None:  # randomize a mass center
        center = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), thickness / 2]
    assert len(center) == 3

    if color is None:  # RGBA
        color = [0, 0.1, 0.2, 1.0]
    assert len(color) == 4

    material = sapien.render.RenderMaterial(base_color=color)

    builder = scene.create_actor_builder()
    # * bottom
    half_bottom_size = [size[0] / 2, size[1] / 2, thickness / 2]
    bottom_position = sapien.Pose(center)
    builder.add_box_collision(half_size=half_bottom_size, pose=bottom_position)
    builder.add_box_visual(
        half_size=half_bottom_size,
        material=material,
        pose=bottom_position,
    )

    # * surrounding walls
    # front and back walls
    wall_pose = sapien.Pose(
        [
            center[0] - size[0] / 2 + thickness / 2,
            center[1],
            size[2] / 2,
        ]
    )
    half_wall_size = [thickness / 2, size[1] / 2, size[2] / 2]
    builder.add_box_collision(
        half_size=half_wall_size,
        pose=wall_pose,
    )
    builder.add_box_visual(
        half_size=half_wall_size,
        material=material,
        pose=wall_pose,
    )

    wall_pose = sapien.Pose(
        [
            center[0] + size[0] / 2 - thickness / 2,
            center[1],
            size[2] / 2,
        ]
    )
    builder.add_box_collision(
        half_size=half_wall_size,
        pose=wall_pose,
    )
    builder.add_box_visual(
        half_size=half_wall_size,
        material=material,
        pose=wall_pose,
    )

    # left and right walls
    wall_pose = sapien.Pose(
        [
            center[0],
            center[1] - size[1] / 2 + thickness / 2,
            size[2] / 2,
        ]
    )
    half_wall_size = [size[0] / 2, thickness / 2, size[2] / 2]
    builder.add_box_collision(
        half_size=half_wall_size,
        pose=wall_pose,
    )
    builder.add_box_visual(
        half_size=half_wall_size,
        material=material,
        pose=wall_pose,
    )

    wall_pose = sapien.Pose(
        [
            center[0],
            center[1] + size[1] / 2 - thickness / 2,
            size[2] / 2,
        ]
    )
    builder.add_box_collision(
        half_size=half_wall_size,
        pose=wall_pose,
    )
    builder.add_box_visual(
        half_size=half_wall_size,
        material=material,
        pose=wall_pose,
    )

    builder.set_initial_pose(sapien.Pose(p=center, q=[1, 0, 0, 0]))
    return builder.build(name=name)
