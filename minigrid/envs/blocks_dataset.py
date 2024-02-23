from __future__ import annotations

import copy
import itertools
import numpy as np
import os
from PIL import Image
import random
from tqdm import tqdm
from typing import List, Union, Tuple, Dict

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, NON_BASE_OBJ_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj, Block, GrippedBlock
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

INT_TO_WORD = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

ACTION_VERBS = [
                'stacks the <c1> block on the <c2> block',
                # 'picks up the <c1> block and puts it on the <c2> block',
                # 'moves the <c1> block onto the <c2> block',
                # 'grabs the <c1> block and places it on the <c2> block',
                # 'uses the <c2> block and stacks the <c1> block on it'
                ]

ALL_COLORS = ['red', 'green', 'blue', 'yellow', 'purple']#, 'cyan', 'purple']
# C1_COLORS = ['red', 'green', 'blue']

class BlocksDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, max_actions=2, max_blocks=5, obs_type='grid', pretrain_version=False, **kwargs):
        self.size = max_blocks + 2
        self.max_actions = max_actions
        self.max_blocks = max_blocks
        self.obs_type = obs_type
        self.tile_size = 16
        self.pretrain_version = pretrain_version
        self.render_mode = "human"

        if pretrain_version:
            # 131072
            self.splits = {'pretrain': 131072, 'pretrain_val': 1024, 'train': 1024, 'val': 1024, 'test': 1024,
                           'rel': 1024, 'abs': 1024, 'exact': 1024, 'compositional': 1024, 'length3': 1024, 'length+1': 1024}
            self.set_split('pretrain')
        else:
            self.splits = {'train': 50000, 'val': 1000, 'test': 1000}
            self.set_split('train')
        self.class_distributions = {k: {} for k in self.splits.keys()}

        mission_space = MissionSpace(mission_func=lambda : '')

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            highlight=False,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_actions * 10,
            **kwargs,
        )
        self.rel_question_fns = (self.rel_q_number_of_blocks_touching, self.rel_q_relative_height, self.rel_q_tower_height)
        self.abs_question_fns = (self.abs_q_shortest_tower, self.abs_q_tallest_tower, self.abs_q_number_of_towers)
        self.exact_question_fns = (self.exact_position_question,)

    def set_split(self, split):
        self.curr_split = split

    @staticmethod
    def _gen_mission(starting_blocks: List[str], sequence: List[str], question_block: str):
        pass

    def env_pos_to_human_pos(self, pos):
        x, y = pos
        return (x - 1, self.size - y - 2)

    def get_obs(self):
        if self.obs_type == 'simple':
            obs = np.zeros((self.size, self.size))
            obs[self.get_height_of_stack_in_col(self.block_pos[self.question_block])] = 1
            obs = obs.tolist()
        elif self.obs_type == 'image':
            obs = self.grid.render(64, (-1, -1)).transpose(2, 0, 1)
            img = Image.fromarray(obs.transpose(1, 2, 0))
            img.save(f'blocks_dataset/step_{self.step_count}.png')
        elif self.obs_type == 'grid':
            # obs = self.grid.encode()
            # # Get rid of walls, get rid of object type and object status (only keep color)
            # obs = obs[1:-1, 1:-1, 1, np.newaxis]
            obs = np.zeros((self.size - 2, self.size - 2, self.max_blocks))
            for i, color in enumerate(self.starting_blocks):
                x, y = self.block_pos[color]
                obs[y - 1, x - 1, i] = 1
        else:
            raise NotImplementedError(f'{self.obs_type} is not a supporter observation type')
        return obs

    @staticmethod
    def permutations(n, r):
        return np.math.factorial(n) // np.math.factorial(n - r)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Get starting blocks
        self.starting_blocks = copy.deepcopy(ALL_COLORS) #np.random.choice(ALL_COLORS, self.max_blocks, replace=False)
        self.block_pos = {}
        # Set up blocks
        for i, color in enumerate(self.starting_blocks):
            self.put_obj(Block(color), i + 1, self.height - 2)
            self.block_pos[color] = (i + 1, self.height - 2)

        # Pick number of actions
        if self.curr_split in 'length3':
            self.num_actions = 3
        elif self.curr_split == 'length+1':
            self.num_actions = self.max_actions + 1
        else:
            num_actions_p = np.array([self.permutations(self.max_blocks, i) for i in range(1, self.max_actions + 1)],
                                     dtype=float)
            # 3 is used for generalization testing
            num_actions_p[2] = 0
            num_actions_p = num_actions_p / np.sum(num_actions_p)
            self.num_actions = np.random.choice(np.arange(1, self.max_actions + 1), p=num_actions_p)

        self.curr_gripper_pos = (0, 0)
        self.is_grabbing_block = False
        self.curr_step = 0
        self.step_count = 0
        self.mission = ('A ' + ' '.join([f'{c},' for c in self.starting_blocks[:-1]]) +
                        f' and a {self.starting_blocks[-1]} block start on a table.')

        if self.curr_split in ['pretrain', 'pretrain_val']:
            self.question_block = np.random.choice(self.starting_blocks)
        elif self.curr_split in ['train', 'test', 'val', 'rel', 'abs', 'exact', 'length3', 'length+1']:
            self.question_block = np.random.choice(self.starting_blocks[:-1])
        elif self.curr_split == 'compositional':
            self.question_block = self.starting_blocks[-1]
        else:
            raise ValueError(f'Invalid split: {self.curr_split}')
        self.traj_obss = [self.get_obs()]
        self.traj_actions = []

    def base_step(self, action):
        self.step_count += 1

        blocks = []
        x, y = self.curr_gripper_pos
        block = self.grid.get(x, y)

        if action == 'grab':
            self.grid.set(x, y, GrippedBlock(block.color))
            self.is_grabbing_block = True
        elif action == 'letgo':
            self.grid.set(x, y, Block(block.color))
            self.is_grabbing_block = False
        elif self.is_grabbing_block:
            # Get each block affected by stack
            while block is not None:
                blocks.append(block)
                # Remove block from previous position
                self.grid.set(x, y, None)
                y -= 1
                block = self.grid.get(x, y)

            new_x, new_y = action
            for block in blocks:
                # Set block to new position
                self.grid.set(new_x, new_y, block)
                self.block_pos[block.color] = (new_x, new_y)
                new_y -= 1
            self.curr_gripper_pos = action
        else:
            self.curr_gripper_pos = action

        if self.render_mode == "human":
            self.render()

    def get_height_of_stack_in_col(self, col):
        y, height = self.height - 2, 0

        block = self.grid.get(col, y)
        while isinstance(block, Block):
            height += 1
            y -= 1
            block = self.grid.get(col, y)
        return height

    def step(self, _):
        # Get valid block to stack and valid block to stack it onto
        start_block = np.random.choice(list(self.block_pos.keys()))
        start_pos = self.block_pos[start_block]
        end_positions = []
        for pos in self.block_pos.values():
            # Cant stack block onto any block in the blocks current column
            if pos[0] == start_pos[0]:
                continue
            # Can only stack a block at the top of a current stack
            for i, pos2 in enumerate(end_positions):
                # Two blocks in the same column, only the highest one can be picked
                if pos[0] == pos2[0]:
                    if pos[1] < pos2[1]:
                        end_positions[i] = pos
                    break
            else:
                end_positions.append(pos)

        if len(end_positions) != 0:
            end_pos_i = np.random.choice(len(end_positions))
            # End position row should be one higher than block it will be stacked on
            end_pos = (end_positions[end_pos_i][0], end_positions[end_pos_i][1] - 1)

            c1, c2 = [self.grid.get(*start_pos).color, self.grid.get(end_pos[0], end_pos[1] + 1).color]

            # Move gripper to start pos, grab, move gripper to end pos, let go
            for action in [start_pos, 'grab', end_pos, 'letgo']:
                self.base_step(action)

            # Calculate actions
            start_idx = (start_pos[0] - 1) * self.max_blocks + (start_pos[1] - 1) # value of 0 - 24
            end_idx = (end_pos[0] - 1) * self.max_blocks + (end_pos[1] - 1) # value of 0 - 24
            action_idx = start_idx * self.max_blocks * self.max_blocks + end_idx
            self.traj_actions.append(action_idx)

            self.curr_step += 1
            rand_verb_phrase = np.random.choice(ACTION_VERBS)
            rand_verb_phrase = rand_verb_phrase.replace('<c1>', c1).replace('<c2>', c2)
            if self.curr_step == 1:
                self.mission += f' The robot {rand_verb_phrase}.'
            else:
                self.mission += f' Then the robot {rand_verb_phrase}.'

        self.traj_obss.append(self.get_obs())
        if (self.curr_step == self.num_actions or len(end_positions) == 0):
            if self.curr_split in ['pretrain', 'pretrain_val']:
                self.answer = ''
            else:
                q_probs = {'rel': [1, 0, 0], 'abs': [0, 1, 0], 'exact': [0, 0, 1]}
                if self.curr_split in q_probs:
                    q_type = np.random.choice(['rel', 'abs', 'exact'], p=q_probs[self.curr_split])
                else:
                    q_type = np.random.choice(['rel', 'abs', 'exact'])

                if q_type == 'rel':
                    np.random.choice(self.rel_question_fns)()
                elif q_type == 'abs':
                    np.random.choice(self.abs_question_fns)()
                else:
                    np.random.choice(self.exact_question_fns)()

            self.class_distributions[self.curr_split][self.answer] = self.class_distributions[self.curr_split].get(self.answer, 0) + 1
            return {'direction': np.array([]), 'image': np.array([]), 'mission': ''}, 0, True, False, {}
        else:
            return {'direction': np.array([]), 'image': np.array([]), 'mission': ''}, 0, False, False, {}

    def exact_position_question(self):
        x, y = self.block_pos[self.question_block]
        self.mission += f' The {self.question_block} block is now>'
        self.answer = f'in row {INT_TO_WORD[self.size - 1 - y]} and col {INT_TO_WORD[x]}'

    def rel_q_relative_height(self):
        other_block = None
        while other_block != self.question_block:
            other_block = np.random.choice(self.starting_blocks)
        heights = [(self.height - 1 - self.block_pos[qb][1]) for qb in [self.question_block, other_block]]
        self.mission += f' Relative to the {other_block} block, the {self.question_block} block is now>'
        if heights[0] > heights[1]:
            self.answer = ' higher'
        elif heights[0] < heights[1]:
            self.answer = ' lower'
        else:  # ==
            self.answer = ' level'

    def rel_q_tower_height(self):
        tower_height = self.get_height_of_stack_in_col(self.block_pos[self.question_block][0])
        self.answer = f' {INT_TO_WORD[tower_height]}'
        self.mission += f' The tower containing the {self.question_block} block now has a height of>'

    def rel_q_number_of_blocks_touching(self):
        x, y = self.block_pos[self.question_block]
        answer = 0
        for x_, y_ in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            block = self.grid.get(x + x_, y + y_)
            if isinstance(block, Block):
                answer += 1
        self.answer = INT_TO_WORD[answer]
        self.mission += f' The number of blocks that the {self.question_block} block is touching is>'

    def abs_q_tallest_tower(self):
        self.mission += f' The tallest stack has a height of>'
        col_heights = [self.get_height_of_stack_in_col(c) for c in range(1, self.width - 1)]
        col_heights = [h for h in col_heights if h > 0]
        self.answer = f' {INT_TO_WORD[max(col_heights)]}'

    def abs_q_shortest_tower(self):
        self.mission += f' The shortest stack has a height of>'
        col_heights = [self.get_height_of_stack_in_col(c) for c in range(1, self.width - 1)]
        col_heights = [h for h in col_heights if h > 0]
        self.answer = f' {INT_TO_WORD[max(col_heights)]}'

    def abs_q_number_of_towers(self):
        self.mission += f'. The number of distinct stacks is>'
        col_heights = [self.get_height_of_stack_in_col(c) for c in range(1, self.width - 1)]
        col_heights = [h for h in col_heights if h > 0]
        self.answer = f' {INT_TO_WORD[max(col_heights)]}'

    def get_trajectory_info(self):
        return self.mission, self.traj_obss, self.traj_actions, self.answer

    def get_class_distributions(self):
        return self.class_distributions


if __name__ == "__main__":
    import argparse
    import gymnasium as gym
    from minigrid.utils.window import Window
    from pathlib import Path
    import json

    gym.register(
        id="BlocksDataset-v0",
        entry_point="minigrid.envs:BlocksDataset",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="BlocksDataset-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=16
    )
    parser.add_argument(
        "--num-per-obj", type=int, help="Number of instances to create for each color/type combination", default=2
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="simple",
        help="Observation type. Can be 'simple', 'grid', 'image'",
    )
    parser.add_argument(
        "--max-verbs",
        type=int,
        default="2",
        help="Maximum number of verbs in mission",
    )
    parser.add_argument(
        "--agent-view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )
    parser.add_argument(
        "--pretrain",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )
    parser.add_argument('--base-dir', type=str, default=Path.cwd(),
                        help='Base directory to save dataset to.')

    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)

    env: MiniGridEnv = gym.make(args.env, max_actions=args.max_verbs, obs_type=args.obs_type, tile_size=args.tile_size,
                                pretrain_version=args.pretrain)
    metadata = {'obs_type': env.obs_type, 'max_actions': env.max_actions, 'grid_size': (env.size, env.size), 'split_sizes': {}}

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    print('STARTING')
    ## For examples
    # env.reset(seed=args.seed)
    # done = False
    # while not done:
    #     _, _, done, _, _ = env.step(None)
    # mission, obss, actions, answer = env.get_trajectory_info()
    #
    # print(mission)
    # print(obss)
    # print(answer)
    #
    # exit(0)
    #
    # # env = FullyObsWrapper(env)

    window = Window("minigrid - " + str(env.__class__))

    dataset_name = f'{env.obs_type}_{env.max_actions}verbs'
    if args.pretrain:
        dataset_name += '_pretrain'
    data_dir = Path(args.base_dir / 'blocks_dataset' / 'data' / dataset_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    if env.obs_type == 'image':
        obs_shape = (3, env.tile_size * env.size, env.tile_size * env.size)
        obs_type = 'float32'
    elif env.obs_type == 'grid':
        obs_shape = (env.size - 2, env.size - 2, 5)
        obs_type = 'int'
    elif env.obs_type == 'simple':
        obs_shape = (2)
        obs_type = 'int'
    else:
        obs_shape, obs_type = None, None

    memmap_max_obs = 32768

    metadata['memmap_shape'] = (memmap_max_obs, *obs_shape)
    metadata['memmap_type'] = obs_type
    metadata['grid_size'] = (env.size, env.size)
    for split in env.splits: # 'val'
        with open(data_dir / f'{split}_dataset.txt', 'w') as dataset_file:
            offsets = [0]
            curr_observation_file_idx = 1
            curr_observation_memmap = np.memmap(str(data_dir / f'obss_{split}_{curr_observation_file_idx}.memmap'),
                                                dtype=obs_type, mode='w+', shape=(memmap_max_obs, *obs_shape))
            curr_obs_idx = 0
            env.set_split(split)
            num_instances = env.splits[split]  # for each direction
            print(f'creating {num_instances} for split: {split}')
            for i in tqdm(range(num_instances)):
                env.reset(seed=args.seed)
                done = False
                while not done:
                    _, _, done, _, _ = env.step(None)
                mission, obss, actions, answer = env.get_trajectory_info()

                mission = mission.replace('<mask>', answer)
                num_obss = len(obss)
                traj_str = json.dumps((mission, curr_observation_file_idx, curr_obs_idx, num_obss, actions, answer))
                dataset_file.write(traj_str)
                offsets.append(offsets[-1] + len(traj_str))
                if curr_obs_idx + num_obss >= memmap_max_obs:
                    curr_observation_file_idx += 1
                    curr_observation_memmap = np.memmap(
                        str(data_dir / f'obss_{split}_{curr_observation_file_idx}.memmap'), dtype=obs_type, mode='w+',
                        shape=(memmap_max_obs, *obs_shape))
                    curr_obs_idx = 0
                for obs in obss:
                    curr_observation_memmap[curr_obs_idx] = obs
                    curr_obs_idx += 1
        with open(data_dir / f'{split}_offset.txt', 'w') as offset_file:
            json.dump(offsets, offset_file)
        metadata['split_sizes'][split] = num_instances
        metadata['class_distributions'] = env.get_class_distributions()

    print(metadata['class_distributions'])

    with open(data_dir / f'metadata', 'w') as metadata_file:
        json.dump(metadata, metadata_file)
