from typing_extensions import *
import sapien
import torch
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor
from .get_box import *


@register_env("MatryoshkaBox-v0", max_episode_steps=100)
class MatryoshkaBoxEnv(BaseEnv):
    """
    Task:
        given a set of boxes (of different sizes) and some small items,
        put smaller boxes into larger boxes,
        and put the items into the smallest box.

        Small items are initially placed in the largest box,
        so robot needs to reason out the order of putting boxes into each other.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        num_things=2,
        **kwargs,
    ):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        self.num_things = num_things

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_agent(self, options, initial_agent_poses=sapien.Pose(p=[-0.615, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.thickness = 0.01
        self.center1 = (0, -0.2, 0.005)
        self.size1 = (0.2, 0.3, 0.1)
        self.lv1 = get_box(
            scene=self.scene,
            size=self.size1,
            thickness=self.thickness,
            name="lv1",
            center=self.center1,
        )

        self.center2 = (0.3, 0, 0.005)
        self.size2 = (0.175, 0.275, 0.088)
        self.lv2 = get_box(
            scene=self.scene,
            size=self.size2,
            thickness=self.thickness,
            name="lv2",
            center=self.center2,
        )

        self.center3 = (0.1, 0.04, 0.005)
        self.size3 = (0.15, 0.25, 0.076)
        self.lv3 = get_box(
            scene=self.scene,
            size=self.size3,
            thickness=self.thickness,
            name="lv3",
            center=self.center3,
        )
        self.levels = [self.lv1, self.lv2, self.lv3]
        self.sizes = [self.size1, self.size2, self.size3]
        self.centers = [self.center1, self.center2, self.center3]

        # create one ball in the self.lv1
        self.ball = self.scene.create_actor_builder()
        self.ball.add_sphere_collision(radius=0.04)
        self.ball.add_sphere_visual(
            radius=0.04,
            material=sapien.render.RenderMaterial(base_color=[0, 0.7, 0.7, 1.0]),
            pose=sapien.Pose(
                p=[self.center1[0], self.center1[1], self.center1[2] + 0.04]
            ),
        )
        self.ball = self.ball.build(name="ball")

    def _initialize_episode(self, env_idx, options):
        self.table_scene.initialize(env_idx)
        with torch.device(self.device):
            b = len(env_idx)

            for i, pos in enumerate(self.levels):
                #! container position
                center_pos = torch.ones(b, 3)
                center_pos[:, 0] = torch.rand((b,)) * 0.3 - 0.15
                center_pos[:, 1] = torch.rand((b,)) * 0.3 - 0.15
                center_pos[:, 2] = self.thickness / 2
                pos.set_pose(Pose.create_from_pq(p=center_pos, q=[1, 0, 0, 0]))
                if i == 0:
                    self.ball.set_pose(
                        Pose.create_from_pq(
                            p=center_pos
                            + torch.tensor([0, 0, 0.04], device=self.device),
                            q=[1, 0, 0, 0],
                        )
                    )
                    self.ball.set_linear_velocity(
                        torch.zeros((b, 3), device=self.device)
                    )
                    self.ball.set_angular_velocity(
                        torch.zeros((b, 3), device=self.device)
                    )

                pos.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                pos.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        with torch.device(self.device):
            pose = torch.zeros((len(self.levels), 3))
            success = torch.zeros(len(self.levels), dtype=torch.bool)
            distance = []

            # check if a smaller box level[i] is inside a larger box level[i-1]
            pose[0, :] = self.levels[0].pose.p
            for i in range(1, len(self.levels)):
                pose[i, :] = self.levels[i].pose.p
                success[i] = (
                    (
                        pose[i - 1, 0] - self.sizes[i - 1][0] / 2
                        <= pose[i, 0] - self.sizes[i][0] / 2
                        <= pose[i, 0] + self.sizes[i][0] / 2
                        <= pose[i - 1, 0] + self.sizes[i - 1][0] / 2
                    )
                    and (
                        pose[i - 1, 1] - self.sizes[i - 1][1] / 2
                        <= pose[i, 1] - self.sizes[i][1] / 2
                        <= pose[i, 1] + self.sizes[i][1] / 2
                        <= pose[i - 1, 1] + self.sizes[i - 1][1] / 2
                    )
                    and (
                        pose[i - 1, 2] - self.thickness[i - 1][2] / 2
                        <= pose[i, 2] - self.thickness[i][2] / 2
                        <= pose[i, 2] + self.sizes[i][2]
                        <= pose[i - 1, 2] + self.sizes[i - 1][2]
                    )
                )

            # distance from ball to smallest box
            distance.append(
                torch.linalg.norm(self.ball.pose.p - self.levels[-1].pose.p)
            )

            # distance from smaller box's center to larger box's center
            for i in range(1, len(self.levels)):
                pose[i, :] = self.levels[i].pose.p
                distance.append(
                    torch.linalg.norm(self.levels[i - 1].pose.p - self.levels[i].pose.p)
                )

            success = torch.tensor([success.all()])
            distance = torch.tensor(distance)
            return {
                "success": success,
                "pos": pose,
                "distance": distance,
            }

    def compute_dense_reward(self, obs, action, info):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            success = info["success"]
            distance = info["distance"]
            reward = torch.where(success, reward + 10, reward)

            for i in range(len(distance)):
                reward += torch.exp(-10.8 * distance[i]) * 0.2

            return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalize the dense reward"""
        max_reward = (
            13.0  # Maximum possible reward (success + all intermediate rewards)
        )
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )

        # Side view camera
        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[0.5, 0, 0.5], target=[0, 0, 0.2]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
        return [top_camera, side_camera]

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
