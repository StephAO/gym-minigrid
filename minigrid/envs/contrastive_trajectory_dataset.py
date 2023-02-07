from __future__ import annotations

import itertools
import os
import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, NON_BASE_OBJ_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv


class ContrastiveTrajectoryDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """
    def __init__(self, size=7, numObjs=1, splits=(0.7, 0.1, 0.2), max_steps: int | None = None, **kwargs):

        # self.numObjs = numObjs
        self.size = size
        # Types of objects to be generated

        self.obj_types = NON_BASE_OBJ_NAMES
        self.col_types = COLOR_NAMES
        self.all_compositions = list(itertools.product(self.col_types, self.obj_types))
        random.shuffle(self.all_compositions)
        splits = int(splits[0] * len(self.all_compositions)), int(sum(splits[:2]) * len(self.all_compositions))
        self.splits = {'train': self.all_compositions[:splits[0]],
                       'val': self.all_compositions[splits[0]:splits[1]],
                       'test': self.all_compositions[splits[1]:]}
        self.curr_split = 'train'


        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.col_types, self.obj_types],
        )

        if max_steps is None:
            max_steps = 5 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            highlight=False,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def set_split(self, split):
        self.curr_split = split
        self.curr_comp_idx = 0

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"A {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        obj_color, obj_type = self.splits[self.curr_split][self.curr_comp_idx]
        self.curr_comp_idx = (self.curr_comp_idx + 1) % len(self.splits[self.curr_split])
        obj = WorldObj.decode(OBJECT_TO_IDX[obj_type], COLOR_TO_IDX[obj_color], 0)
        pos = self.place_obj(obj)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        self.targetType, self.target_color = obj_type, obj_color
        self.target_pos = pos

        # Create distractor
        distr_options = [i for i in range(len(self.splits[self.curr_split])) if i != self.curr_comp_idx]
        distr_idx = self._rand_elem(distr_options)
        dist_color, dist_type = self.splits[self.curr_split][distr_idx]
        dist = WorldObj.decode(OBJECT_TO_IDX[dist_type], COLOR_TO_IDX[dist_color], 0)
        self.place_obj(dist)

        self.label = f"A {self.target_color} {self.targetType}"
        self.mission = self.label
        # print(self.mission)

    # def reset(self, *args, seed=None, options=None):
    #     obs, _ = super().reset(*args, seed=seed, options=options)
    #     print('a', obs)
    #     return obs, self.target_pos

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = False

        # Toggle/pickup action terminates the episode
        if self.carrying:
            if self.carrying.color == self.target_color and \
                    self.carrying.type == self.targetType:
                reward = 1
                done = True
            else:
                reward = -1
                done = True

        done = done or terminated or truncated

        print(done, reward)

        return obs, reward, done, info



if __name__ == "__main__":
    import argparse
    import gymnasium as gym
    from minigrid.utils.window import Window
    from minigrid.oracle_agent import OracleAgent
    from PIL import Image
    from pathlib import Path

    gym.register(
        id="ContrastiveTrajectoryDataset-v0",
        entry_point="minigrid.envs:ContrastiveTrajectoryDataset",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="ContrastiveTrajectoryDataset-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--num-per-obj", type=int, help="Number of instances to create for each color/type combination", default=2
    )
    parser.add_argument(
        "--agent-view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size, render_mode='rgb_array')

    # env = FullyObsWrapper(env)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    window = Window("minigrid - " + str(env.__class__))

    for split in ['train', 'val', 'test']:
        env.set_split(split)
        num_instances = len(env.splits[split]) * args.num_per_obj
        Path(f'contrastive_dataset/{split}').mkdir(parents=True, exist_ok=True)
        print(f'creating {num_instances} for split: {split}')

        oracle = OracleAgent(env, visualize=True, agent_view=args.agent_view)
        demos = oracle.generate_demos(1)
        mission, obss, actions, rewards, target_cell, label = demos[0]

        print(mission)
        print(obss)
        print(actions)
        print(rewards)
        print(target_cell)
        print(label)
        #
        # for i in range(num_instances):
        #     env.reset(seed=args.seed)
        #     frame = env.get_frame(agent_pov=args.agent_view, highlight=False)
        #     img = Image.fromarray(frame)
        #     img.save(f'contrastive_dataset/{split}/{env.mission}.{i}.png')
            #
            # print(type(frame))
            # window.show_img(frame)
            #
            # window.show(block=False)

            # input('hit enter')



