from __future__ import annotations

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

INT_TO_WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

ACTION_VERBS = ['pick up the <c1> block and put it on the <c2> block',
                'stack the <c1> block on the <c2> block',
                'move the <c1> block onto the <c2> block',
                'grab the <c1> block and place it on the <c2> block',
                ]

ALL_COLORS = ['red', 'green', 'blue', 'yellow', 'white', 'purple']
# C1_COLORS = ['red', 'green', 'blue']

class BlocksDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=6, max_verbs=2, max_blocks=4, obs_type='grid', **kwargs):
        self.size = size
        self.max_verbs = max_verbs
        self.max_blocks = max_blocks
        self.obs_type = obs_type
        self.tile_size = 16

        self.render_mode = "human"

        self.splits = {'train': 40000, 'val': 5000, 'test': 5000}
        self.set_split('train')
        self.class_distributions = {k: {} for k in self.splits.keys()}

        # self.all_color_pairs = list(itertools.permutations(GROUP_1_COLORS, r=2))

        # Base sequences
        # base_sequences = []
        # for i in range(max_verbs + 1):
        #     base_sequences += list(itertools.product(self.all_color_pairs, repeat=i))
        # random.shuffle(base_sequences)
        # splits = int(splits[0] * len(base_sequences)), int(sum(splits[:2]) * len(base_sequences))
        # Compositional sequences. Sequences that include at least one unseen verb
        # comp_seqs = [seq for seq in itertools.product(ALL_VERBS.keys(), repeat=max_verbs) if
        #              any(v in COMPOSITIONAL_VERBS.keys() for v in seq)]
        # random.shuffle(comp_seqs)
        # comp_seqs = comp_seqs[:10000]
        # Length sequences
        # longer_seqs = list(itertools.product(ACTION_VERBS.keys(), repeat=max_verbs + 1))
        # random.shuffle(longer_seqs)
        # longer_seqs = longer_seqs[:10000]
        # self.splits = {'train': base_sequences[:splits[0]],
        #                'val': base_sequences[splits[0]:splits[1]],
        #                'test': base_sequences[splits[1]:],}
                       # 'compositional': comp_seqs,
                       # 'length': longer_seqs}


        mission_space = MissionSpace(mission_func=lambda : '')

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

    @staticmethod
    def _gen_mission(starting_blocks: List[str], sequence: List[str], question_block: str):
        mission = 'A' + ' '.join([f'{c} block, ' for c in starting_blocks]) + ' all start on a table.'

        for i, (c1, c2) in enumerate(sequence):
            rand_verb_phrase = np.random.choice(ACTION_VERBS)
            rand_verb_phrase = rand_verb_phrase.replace('<c1>', c1).replace('<c2>', c2)
            if i == 0:
                mission += f'You ' + rand_verb_phrase
            else:
                mission += f', then you {rand_verb_phrase}'
        mission += f'. The {question_block} is at a height of <mask>.'
        return mission

    def env_pos_to_human_pos(self, pos):
        x, y = pos
        return (x, self.size - y - 1)

    def get_obs(self):
        if self.obs_type == 'simple':
            obs = np.zeros((self.size, self.size))
            obs[self.env_pos_to_human_pos(self.block_pos[self.question_block])] = 1
            obs = obs.tolist()
        elif self.obs_type == 'image':
            obs = self.grid.render(16, (-1, -1)).transpose(2, 0, 1)
            img = Image.fromarray(obs.transpose(1, 2, 0))
            img.save(f'blocks_dataset/step_{self.step_count}.png')
        elif self.obs_type == 'grid':
            obs = self.grid.encode()
        else:
            raise NotImplementedError(f'{self.obs_type} is not a supporter observation type')
        return obs

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Get starting blocks
        self.starting_blocks = np.random.choice(ALL_COLORS, 4, replace=False)

        self.block_pos = {}
        self.pos_block = {}
        # Set up blocks
        for i, color in enumerate(self.starting_blocks):
            self.put_obj(Block(color), i + 1, self.height - 2)
            self.block_pos[color] = (i + 1, self.height - 2)
            self.pos_block[(i + 1, self.height - 2)] = color

        # self.mission = self._gen_mission(self.agent_dir, self.curr_seq)
        self.curr_gripper_pos = (0, 0)
        self.is_grabbing_block = False
        self.curr_verb_step = 0
        self.step_count = 0
        self.mission = ('A ' + ' '.join([f'{c} block,' for c in self.starting_blocks[:-1]]) +
                        f' and a {self.starting_blocks[-1]} all start on a table.')
        if self.curr_split == 'train':
            self.question_block = np.random.choice(self.starting_blocks[:-2])
        elif self.curr_split == 'val':
            self.question_block = self.starting_blocks[-2]
        elif self.curr_split == 'test':
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
                del self.pos_block[(x, y)]
                y -= 1
                block = self.grid.get(x, y)

            new_x, new_y = action
            for block in blocks:
                # Set block to new position
                self.grid.set(new_x, new_y, block)
                self.block_pos[block.color] = (new_x, new_y)
                self.pos_block[(new_x, new_y)] = block.color
                new_y -= 1
            self.curr_gripper_pos = action
        else:
            self.curr_gripper_pos = action

        if self.render_mode == "human":
            self.render()

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

        end_pos_i = np.random.choice(len(end_positions))
        # End position row should be one higher than block it will be stacked on
        end_pos = (end_positions[end_pos_i][0], end_positions[end_pos_i][1] - 1)

        c1, c2 = [self.grid.get(*start_pos).color, self.grid.get(end_pos[0], end_pos[1] + 1).color]

        # Move gripper to start pos, grab, move gripper to end pos, let go
        prev_action = None
        for action in [start_pos, 'grab', end_pos, 'letgo']:
            self.base_step(action)
            if isinstance(action, str):
                self.traj_obss.append(self.get_obs())
                a_pos = self.env_pos_to_human_pos(prev_action)
                action_idx = a_pos[0] * self.size + a_pos[1]
                if self.is_grabbing_block:
                    action_idx += self.size * self.size
                self.traj_actions.append(action_idx)
            prev_action = action

        self.curr_verb_step += 1
        rand_verb_phrase = np.random.choice(ACTION_VERBS)
        rand_verb_phrase = rand_verb_phrase.replace('<c1>', c1).replace('<c2>', c2)
        if self.curr_verb_step == 1:
            self.mission += f' You ' + rand_verb_phrase
        else:
            self.mission += f', then you {rand_verb_phrase}'

        if self.curr_verb_step == self.max_verbs:
            self.mission += f'. The {self.question_block} block is at a height of <mask>.'
            self.answer = INT_TO_WORD[self.size - self.block_pos[self.question_block][1] - 1]
            self.class_distributions[self.curr_split][self.answer] = self.class_distributions[self.curr_split].get(self.answer, 0) + 1
            return {'direction': np.array([]), 'image': np.array([]), 'mission': ''}, 0, True, False, {}
        else:
            return {'direction': np.array([]), 'image': np.array([]), 'mission': ''}, 0, False, False, {}

    # return obs, reward, terminated, truncated, info

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
    parser.add_argument('--base-dir', type=str, default=Path.cwd(),
                        help='Base directory to save dataset to.')

    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)

    env: MiniGridEnv = gym.make(args.env, max_verbs=args.max_verbs, obs_type=args.obs_type, tile_size=args.tile_size)
    metadata = {'obs_type': env.obs_type, 'max_verbs': env.max_verbs, 'grid_size': (env.size, env.size), 'split_sizes': {}}

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    # print('STARTING')
    # env.reset(seed=args.seed)
    # done = False
    # while not done:
    #     _, _, done, _, _ = env.step(None)
    # mission, obss, actions, answer = env.get_trajectory_info()
    #
    # # print(mission)
    # # print(answer)
    # #
    # # exit(0)
    #
    # # env = FullyObsWrapper(env)

    window = Window("minigrid - " + str(env.__class__))

    data_dir = Path(args.base_dir / 'blocks_dataset' / f'{env.obs_type}_{env.max_verbs}verbs')
    data_dir.mkdir(parents=True, exist_ok=True)
    if env.obs_type == 'image':
        obs_shape = (3, env.tile_size * env.size, env.tile_size * env.size)
        obs_type = 'float32'
    elif env.obs_type == 'grid':
        obs_shape = (env.size, env.size, 3)
        obs_type = 'int'
    elif env.obs_type == 'simple':
        obs_shape = (env.size, env.size)
        obs_type = 'int'
    else:
        obs_shape, obs_type = None, None

    memmap_max_obs = 70000

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

                # Code to visualize obs and actions to do sanity checks
                # print(mission)
                # if env.obs_type != 'image':
                #     print(obss[0])
                # for i in range(len(actions)):
                #     prefix = 'empty'
                #     if actions[i] > env.size ** 2:
                #         prefix = 'gripped'
                #         actions[i] = actions[i] - env.size ** 2
                #     x, y = actions[i] // env.size, actions[i] % env.size
                #     print(x, y)
                #     if env.obs_type != 'image':
                #         print(obss[i + 1])
                # print(answer)
                # exit(0)

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
