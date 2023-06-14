from __future__ import annotations

import itertools
import numpy as np
import os
import random
from tqdm import tqdm

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, NON_BASE_OBJ_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


ACTION_VERBS = {'turn left': [Actions.left], 'turn right': [Actions.right], 'go straight': [Actions.forward],
                'turn around': [Actions.right, Actions.right],
                'turn 90 degrees clockwise': [Actions.right],
                'turn 180 degrees clockwise': [Actions.right, Actions.right],
                'turn 270 degrees clockwise': [Actions.right, Actions.right, Actions.right],
                'turn 90 degrees counterclockwise': [Actions.left],
                'turn 180 degrees counterclockwise': [Actions.left, Actions.left],
                'turn 270 degrees counterclockwise': [Actions.left, Actions.left, Actions.left]
                }

DIRECTIONS_IDX_TO_STR = ['east', 'south', 'west', 'north']


class DirectionsDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=7, max_verbs=4, splits=(0.7, 0.1, 0.2), simple_obs=True, **kwargs):
        self.size = size
        self.simple_obs = simple_obs

        self.all_sequences = []
        for i in range(max_verbs + 1):
            self.all_sequences += list(itertools.product(ACTION_VERBS.keys(), repeat=i))
        random.shuffle(self.all_sequences)

        splits = int(splits[0] * len(self.all_sequences)), int(sum(splits[:2]) * len(self.all_sequences))
        self.splits = {'train': self.all_sequences[:splits[0]],
                       'val': self.all_sequences[splits[0]:splits[1]],
                       'test': self.all_sequences[splits[1]:]}
        self.curr_split = 'train'
        self.curr_idx = -1
        self.curr_dir = -1

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[[0, 1, 2, 3], self.all_sequences],
        )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            highlight=False,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_verbs * 10,
            **kwargs,
        )

    def set_split(self, split):
        self.curr_split = split
        self.curr_idx = -1
        self.curr_dir = -1

    @staticmethod
    def _gen_mission(starting_dir: str, sequence: str):
        mission = f'An agent is facing {DIRECTIONS_IDX_TO_STR[starting_dir]}'
        for i, verb in enumerate(sequence):
            if i == 0:
                mission += f'. They {verb}'
            else:
                mission += f', then they {verb}'
        mission += '. The agent is now facing<mask>.'
        return mission

    def get_obs(self):
        if self.simple_obs:
            return np.eye(4)[self.agent_dir].tolist()
        else:
            return self.get_obs()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Get next sequence
        self.curr_dir += 1
        if self.curr_dir == 4:
            self.curr_dir = 0
            self.curr_idx += 1
            if self.curr_idx >= len(self.splits[self.curr_split]):
                print(self.curr_idx, self.curr_split, len(self.splits[self.curr_split]))
        self.curr_seq = self.splits[self.curr_split][self.curr_idx]

        # Place agent in middle with next orientation
        self.place_agent(top=((self.size - 1) // 2, (self.size - 1) // 2), size=(1, 1))
        self.agent_dir = self.curr_dir

        self.mission = self._gen_mission(self.agent_dir, self.curr_seq)
        self.curr_verb_step = 0
        self.curr_action_step = 0
        self.traj_obss = [self.get_obs()]
        self.traj_actions = []

    def step(self, _):
        if len(self.curr_seq) == 0:
            action = Actions.stay
        else:
            curr_verb = self.curr_seq[self.curr_verb_step]
            action = ACTION_VERBS[curr_verb][self.curr_action_step]
        obs, reward, terminated, truncated, info = super().step(action)

        self.traj_obss.append(self.get_obs())
        self.traj_actions.append(action)

        self.curr_action_step += 1
        if len(self.curr_seq) == 0:
            terminated = True
            self.answer = f' {DIRECTIONS_IDX_TO_STR[self.agent_dir]}'
        elif self.curr_action_step >= len(ACTION_VERBS[curr_verb]):
            self.curr_action_step = 0
            self.curr_verb_step += 1
            if self.curr_verb_step >= len(self.curr_seq):
                terminated = True
                self.answer = f' {DIRECTIONS_IDX_TO_STR[self.agent_dir]}'
        return obs, reward, terminated, truncated, info

    def get_trajectory_info(self):
        return self.mission, self.traj_obss, self.traj_actions, self.answer


if __name__ == "__main__":
    import argparse
    import gymnasium as gym
    from minigrid.utils.window import Window
    from PIL import Image
    from pathlib import Path
    import json

    gym.register(
        id="DirectionsDataset-v0",
        entry_point="minigrid.envs:DirectionsDataset",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="DirectionsDataset-v0"
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

    env: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size)

    # env = FullyObsWrapper(env)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    window = Window("minigrid - " + str(env.__class__))

    Path(f'directions_dataset/').mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        with open(Path(f'directions_dataset/') / f'{split}_dataset.txt', 'w') as dataset_file:
            offsets = [0]
            env.set_split(split)
            num_instances = 4 * len(env.splits[split]) # for each direction
            print(f'creating {num_instances} for split: {split}')
            for i in tqdm(range(num_instances)):
                env.reset(seed=args.seed)
                done = False
                while not done:
                    _, _, done, _, _ = env.step(None)
                mission, obss, actions, answer = env.get_trajectory_info()
                traj_str = json.dumps((mission, obss, actions, answer))
                dataset_file.write(traj_str)
                offsets.append(offsets[-1] + len(traj_str))
        with open(Path(f'directions_dataset/') / f'{split}_offset.txt', 'w') as offset_file:
            json.dump(offsets, offset_file)

