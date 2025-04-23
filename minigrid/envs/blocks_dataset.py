from __future__ import annotations

import copy
import itertools
import numpy as np
import math
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
                'picks up the <c1> block and places it in column <c2>',
                # 'picks up the <c1> block and puts it on the <c2> block',
                # 'moves the <c1> block onto the <c2> block',
                # 'grabs the <c1> block and places it on the <c2> block',
                # 'uses the <c2> block and stacks the <c1> block on it'
                ]

ALL_COLORS = ['red', 'green', 'blue', 'yellow', 'purple']#, 'cyan', 'orange', 'white']#, 'grey', 'brown']
# C1_COLORS = ['red', 'green', 'blue']

class BlocksDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, max_actions=2, max_blocks=5, obs_type='grid', **kwargs):
        self.size = max_blocks + 2
        self.max_actions = max_actions
        self.max_blocks = max_blocks
        self.obs_type = obs_type
        self.tile_size = 16
        self.render_mode = "human"

        self.set_split("train")

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
        # self.rel_question_fns = (self.rel_q_number_of_blocks_touching, self.rel_q_relative_height, self.rel_q_tower_height)
        # self.abs_question_fns = (self.abs_q_shortest_tower, self.abs_q_tallest_tower, self.abs_q_number_of_towers)
        # self.exact_question_fns = (self.exact_position_question,)

    @staticmethod
    def _gen_mission(starting_blocks: List[str], sequence: List[str], question_block: str):
        pass

    def set_split(self, split):
        self.curr_split = split

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
            img.save(f'example_images/blocks/step_{self.step_count}.png')
        elif self.obs_type == 'grid_one_hot':
            # Do not include walls,
            # Encode color using one-hot encoding (different channel per color)
            obs = np.zeros((self.size - 2, self.size - 2, len(ALL_COLORS) + 1))
            for i, color in enumerate(self.starting_blocks):
                x, y = self.block_pos[color]
                color_idx = COLOR_TO_IDX[str(color)]
                obs[y - 1, x - 1, color_idx] = 1
        elif self.obs_type == 'grid':
            obs = np.zeros((self.size - 2, self.size - 2))
            for i, color in enumerate(self.starting_blocks):
                x, y = self.block_pos[color]
                color_idx = COLOR_TO_IDX[str(color)]
                obs[y - 1, x - 1] = color_idx
        else:
            raise NotImplementedError(f'{self.obs_type} is not a supporter observation type')
        return obs

    @staticmethod
    def permutations(n, r):
        return math.factorial(n) // math.factorial(n - r)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Get starting blocks
        self.starting_blocks = np.random.choice(ALL_COLORS, self.max_blocks, replace=False) # copy.deepcopy(ALL_COLORS) #
        self.block_pos = {}
        # Set up blocks
        for i, color in enumerate(self.starting_blocks):
            self.put_obj(Block(color), i + 1, self.height - 2)
            self.block_pos[color] = (i + 1, self.height - 2)

        # Pick number of actions
        if 'length' in self.curr_split:
            added_actions = self.curr_split.split('+')[-1]
            self.num_actions = self.max_actions + int(added_actions)
        else:
            num_actions_p = np.array([self.permutations(self.max_blocks, i) for i in range(1, self.max_actions + 1)],
                                     dtype=float)
            num_actions_p = num_actions_p / np.sum(num_actions_p)
            self.num_actions = np.random.choice(np.arange(1, self.max_actions + 1), p=num_actions_p)

        self.curr_gripper_pos = (0, 0)
        self.is_grabbing_block = False
        self.curr_step = 0
        self.step_count = 0
        self.init_phrase = (' '.join([f'a {c},' for c in self.starting_blocks[:-1]]) +
                        f' and a {self.starting_blocks[-1]} block start in columns one through five respectively.').capitalize()
        self.action_phrases = []


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
            while isinstance(block, Block) or isinstance(block, GrippedBlock):
                blocks.append(block)
                # Remove block from previous position
                self.grid.set(x, y, None)
                y -= 1
                if y < 0 or y >= self.height:
                    print(block, x, y)
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
        start_block_index = np.random.randint(len(self.starting_blocks))
        start_block = self.starting_blocks[start_block_index]
        start_pos = self.block_pos[start_block]
        # for pos in self.block_pos.values():
        #     # Cant stack block onto any block in the blocks current column
        #     if pos[0] == start_pos[0]:
        #         continue
        #     # Can only stack a block at the top of a current stack
        #     for i, pos2 in enumerate(end_positions):
        #         # Two blocks in the same column, only the highest one can be picked
        #         if pos[0] == pos2[0]:
        #             if pos[1] < pos2[1]:
        #                 end_positions[i] = pos
        #             break
        #     else:
        #         end_positions.append(pos)
        col_probs = np.ones(self.size - 2)
        col_probs[start_pos[0] - 1] = 0
        col_probs = col_probs / np.sum(col_probs)
        end_col = np.random.choice(list(range(1, self.size - 1)), p=col_probs)
        for row in range(self.size-2, -1, -1):
            x = self.grid.get(end_col, row)
            if not isinstance(x, Block):
                end_row = row
                break
        else:
            raise RuntimeError(f'ERROR, no empty row, this should be impossible')

        end_pos = (end_col, end_row)

        c1, c2 = [self.grid.get(*start_pos).color, INT_TO_WORD[end_pos[0]]]#self.grid.get(end_pos[0], end_pos[1] + 1).color]

        # Move gripper to start pos, grab, move gripper to end pos, let go
        for action in [start_pos, 'grab', end_pos, 'letgo']:
            self.base_step(action)

        # Calculate actions
        # end_col - 1 because cols are indexed 1-5 b/c 0 and 6 are walls
        action_idx = int(start_block_index * (self.size - 2) + (end_col - 1))
        self.traj_actions.append(action_idx)

        self.curr_step += 1
        rand_verb_phrase = np.random.choice(ACTION_VERBS)
        rand_verb_phrase = rand_verb_phrase.replace('<c1>', c1).replace('<c2>', c2)
        if self.curr_step == 1:
            self.action_phrases.append(f' The robot {rand_verb_phrase}.')
        else:
            self.action_phrases.append(f' Then the robot {rand_verb_phrase}.')

        self.traj_obss.append(self.get_obs())
        if (self.curr_step == self.num_actions):# or len(end_positions) == 0):
            self.final_state_tallest_tower()

            return {'direction': np.array([]), 'image': np.array([]), 'mission': ''}, 0, True, False, {}
        else:
            return {'direction': np.array([]), 'image': np.array([]), 'mission': ''}, 0, False, False, {}

    def final_state_old(self):
        self.outcome_phrase = ''
        for row in range(self.size):
            blocks = []
            cols = []
            for col in range(self.size):
                x = self.grid.get(col, row)
                if isinstance(x, Block):
                    blocks.append( x.color )
                    cols.append(str(col))

            if len(blocks) == 1:
                self.outcome_phrase += f' At height {row}, there is the {blocks[0]} block in column {cols[0]}.'
                # self.answer += f' Row {row} contains the {", ".join(blocks[0])} blocks.'
            elif len(blocks) > 0:
                self.outcome_phrase += (f' At height {row}, there are the {", ".join(blocks[:-1])}, and {blocks[-1]} blocks'
                                f' in columns {", ".join(cols[:-1])}, and {cols[-1]}.')

    def final_state(self):
        self.outcome_phrase = ' The final block locations are'
        for i, block in enumerate(self.starting_blocks):
            x, y = self.block_pos[block]
            self.outcome_phrase += f' {block} ({x}, {y})'
            self.outcome_phrase += ',' if (i + 1) < len(self.starting_blocks) else '.'

    def final_state_tallest_tower(self):
        col_heights = [self.get_height_of_stack_in_col(c) for c in range(1, self.width - 1)]
        # col_heights = [h for h in col_heights if h > 0]
        tallest_col = np.argmax(col_heights) + 1
        blocks_in_stack = []
        for row in range(self.size-1, 0, -1):
            x = self.grid.get(tallest_col, row)
            if isinstance(x, Block):
                blocks_in_stack.append(x.color)

        block_s = 'block' if len(blocks_in_stack) == 1 else 'blocks'
        self.outcome_phrase = f' The tallest stack is in column {INT_TO_WORD[tallest_col]} and is {INT_TO_WORD[len(blocks_in_stack)]} {block_s} tall. It consists of the '
        self.umap_label = str((tallest_col - 1) * 5 + len(blocks_in_stack)) #INT_TO_WORD[tallest_col]
        #self.umap_label = INT_TO_WORD[len(blocks_in_stack)]
        if len(blocks_in_stack) == 1:
            self.outcome_phrase += f'{blocks_in_stack[0]} block.'
            # self.answer += f' Row {row} contains the {", ".join(blocks[0])} blocks.'
        else:
             self.outcome_phrase += f'{", ".join(blocks_in_stack[:-1])}, and {blocks_in_stack[-1]} blocks.'

    def exact_position_question(self):
        x, y = self.block_pos[self.question_block]
        self.outcome_phrase = f' The {self.question_block} block is now in row {INT_TO_WORD[self.size - 1 - y]} and col {INT_TO_WORD[x]}'

    def get_trajectory_info(self):
        return self.traj_obss, self.traj_actions, self.init_phrase, self.action_phrases, self.outcome_phrase, self.umap_label

if __name__ == "__main__":
        import gymnasium as gym
        import matplotlib.pyplot as plt
        import numpy as np
            
        gym.register(
            id="BlocksDataset-v0",
            entry_point="minigrid.envs:BlocksDataset",
        )

        env: MiniGridEnv = gym.make('BlocksDataset-v0', max_blocks=5, max_actions=5,
                                    obs_type='image', tile_size=32)
        

        print('CREATING VISUAL EXAMPLE')
        ## For examples
        env.reset(seed=42)
        done = False
        while not done:
            _, _, done, _, _ = env.step(None)
        states, actions, init_phrase, action_phrases, outcome_phrase, umap_label = env.env.env.get_trajectory_info()
        print(f"Initial phrase: {init_phrase}")
        combined_action_phrases = ' '.join(action_phrases)
        print(f"Action phrases: {combined_action_phrases}")
        print(f"Outcome phrase: {outcome_phrase}")
        