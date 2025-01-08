from __future__ import annotations

from enum import IntEnum
import itertools
import numpy as np
import os
from PIL import Image
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


class DDActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    turn_around = 2
    stay = 3


USE_HIGH_LEVEL_ACTIONS = True
LL_ACTION_VERBS = {'turn left': [DDActions.left], 'turn right': [DDActions.right], 'go straight': [DDActions.stay],
                   'turn around': [DDActions.right, DDActions.right],
                   'turn 90 degrees clockwise': [DDActions.right],
                   'turn 180 degrees clockwise': [DDActions.right, DDActions.right],
                   'turn 270 degrees clockwise': [DDActions.right, DDActions.right, DDActions.right],

                   'rotate 90 degrees clockwise': [DDActions.right],
                   'rotate 180 degrees clockwise': [DDActions.right, DDActions.right],
                   'rotate 270 degrees clockwise': [DDActions.right, DDActions.right, DDActions.right],

                   'turn 90 degrees counterclockwise': [DDActions.left],
                   'turn 180 degrees counterclockwise': [DDActions.left, DDActions.left],
                   'turn 270 degrees counterclockwise': [DDActions.left, DDActions.left, DDActions.left],

                   'rotate 180 degrees counterclockwise': [DDActions.left, DDActions.left],
                   'rotate 270 degrees counterclockwise': [DDActions.left, DDActions.left, DDActions.left],
                   }

HL_ACTION_VERBS = {'turn left': [DDActions.left], 'turn right': [DDActions.right], 'go straight': [DDActions.stay],
                   'turn around': [DDActions.turn_around],
                   'turn 90 degrees clockwise': [DDActions.right],
                   'turn 180 degrees clockwise': [DDActions.turn_around],
                   'turn 270 degrees clockwise': [DDActions.left],
                   'turn 360 degrees clockwise': [DDActions.stay],

                   # 'rotate 90 degrees clockwise': [DDActions.right],
                   # 'rotate 180 degrees clockwise': [DDActions.turn_around],
                   # 'rotate 270 degrees clockwise': [DDActions.left],

                   'turn 90 degrees counterclockwise': [DDActions.left],
                   'turn 180 degrees counterclockwise': [DDActions.turn_around],
                   'turn 270 degrees counterclockwise': [DDActions.right],
                   'turn 360 degrees counterclockwise': [DDActions.stay],

                   # 'rotate 180 degrees counterclockwise': [DDActions.turn_around],
                   # 'rotate 270 degrees counterclockwise': [DDActions.right],
                   }

ACTION_VERBS = HL_ACTION_VERBS if USE_HIGH_LEVEL_ACTIONS else LL_ACTION_VERBS

DIRECTIONS_IDX_TO_STR = ['east', 'south', 'west', 'north']


class DirectionsDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=3, max_actions=2, obs_type='grid', **kwargs):
        self.size = size
        self.max_actions = max_actions
        self.obs_type = obs_type
        self.tile_size = 16

        train_size, val_size, test_size, length3_size, lengthp1_size = 131072, 1024, 1000, 1000, 1000

        # Base sequences
        base_sequences = []
        for i in range(1, max_actions + 1):
            if i != 3:
                base_sequences += list(itertools.product(ACTION_VERBS.keys(), repeat=i))
        random.shuffle(base_sequences)

        length3_seqs = list(itertools.product(ACTION_VERBS.keys(), repeat=3))
        random.shuffle(length3_seqs)
        length3_seqs = length3_seqs[:length3_size]
        lengthp1_seqs = list(itertools.product(ACTION_VERBS.keys(), repeat=max_actions + 1))
        random.shuffle(lengthp1_seqs)
        lengthp1_seqs = lengthp1_seqs[:lengthp1_size]

        self.splits = {'train': base_sequences[:train_size],
                       'val': base_sequences[train_size:train_size + val_size],
                       'test': base_sequences[train_size + val_size:train_size + val_size + test_size],
                       'length3': length3_seqs, 'length+1': lengthp1_seqs}

        self.set_split('train')

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[[0,1,2,3], [seq for seq in self.splits['train']]],
        )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            highlight=False,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_verbs * 10,
            agent_view_size=size,
            **kwargs,
        )
        self.actions = DDActions

    @staticmethod
    def _gen_mission(starting_dir: str, sequence: str):
        mission = f'The robot is facing {starting_dir}.'
        for i, verb in enumerate(sequence):
            if i == 0:
                mission += f' The robot {verb}.'
            else:
                mission += f' Then the robot {verb}.'
        return mission

    def get_obs(self):
        if self.obs_type == 'simple':
            obs = np.eye(4)[self.agent_dir].tolist()
        elif self.obs_type == 'image':
            obs = self.get_frame(agent_pov=True, highlight=False, tile_size=self.tile_size).transpose(2, 0, 1)
            img = Image.fromarray(obs.transpose(1, 2, 0))
            img.save(f'directions_dataset/step_{self.curr_verb_step + 1}.png')
            # img = Image.fromarray(obs)
            # img.show()
        elif self.obs_type == 'grid':
            # Get rid of object color and object status (only keep object type)
            obs = self.gen_obs()['image'].transpose(2, 0, 1)[0, ..., np.newaxis]

            # Encode object_type using one-hot encoding (different channel per object_type)
            # in core/constants.py, we use world object up to idx 10
            oh_obs = np.zeros((self.size, self.size, 10))
            for x in range(self.size):
                for y in range(self.size):
                    oh_obs[x, y, obs[x, y]] = 1

            obs = oh_obs
        else:
            raise NotImplementedError(f'{self.obs_type} is not a supporter observation type')
        return obs

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Set up visual compass
        self.put_obj(WorldObj.decode(OBJECT_TO_IDX['west'], COLOR_TO_IDX['red'], 0), 0, self.width // 2)
        self.put_obj(WorldObj.decode(OBJECT_TO_IDX['south'], COLOR_TO_IDX['red'], 0), self.height // 2, self.width - 1)
        self.put_obj(WorldObj.decode(OBJECT_TO_IDX['east'], COLOR_TO_IDX['red'], 0), self.height - 1, self.width // 2)
        self.put_obj(WorldObj.decode(OBJECT_TO_IDX['north'], COLOR_TO_IDX['red'], 0), self.height // 2, 0)

        # Get next sequence
        self.curr_seq = self.splits[self.curr_split][self.curr_idx]
        self.curr_idx += 1

        # Place agent in middle with next orientation
        self.place_agent(top=((self.size - 1) // 2, (self.size - 1) // 2), size=(1, 1))

        self.agent_dir = np.random.randint(len(DIRECTIONS_IDX_TO_STR))
        self.mission = self._gen_mission(DIRECTIONS_IDX_TO_STR[self.agent_dir], self.curr_seq)
        self.curr_verb_step = -1
        self.curr_action_step = 0
        self.traj_obss = [self.get_obs()]
        self.curr_verb_step = 0
        self.traj_actions = []

    def base_step(self, action):
        self.step_count += 1

        reward, terminated, truncated = 0, False, False
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.turn_around:
            self.agent_dir = (self.agent_dir + 2) % 4

        # Done action (not used by default)
        elif action == self.actions.stay:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def step(self, _):
        if len(self.curr_seq) == 0:
            action = DDActions.stay
        else:
            curr_verb = self.curr_seq[self.curr_verb_step]
            action = ACTION_VERBS[curr_verb][self.curr_action_step]

        obs, reward, terminated, truncated, info = self.base_step(action)

        self.traj_obss.append(self.get_obs())
        # TODO test action
        self.traj_actions.append(int(action))

        self.curr_action_step += 1
        if len(self.curr_seq) == 0:
            terminated = True
            self.answer = f'The robot is now facing {DIRECTIONS_IDX_TO_STR[self.agent_dir]}.'
        elif self.curr_action_step >= len(ACTION_VERBS[curr_verb]):
            self.curr_action_step = 0
            self.curr_verb_step += 1
            if self.curr_verb_step >= len(self.curr_seq):
                terminated = True
                self.answer = f'The robot is now facing {DIRECTIONS_IDX_TO_STR[self.agent_dir]}.'

        return obs, reward, terminated, truncated, info

    def get_trajectory_info(self):
        return self.mission, self.traj_obss, self.traj_actions, self.answer, self.answer




