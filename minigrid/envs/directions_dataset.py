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
                   # 'turn 270 degrees counterclockwise': [DDActions.right],
                   'turn 360 degrees counterclockwise': [DDActions.stay],

                   # 'rotate 180 degrees counterclockwise': [DDActions.turn_around],
                   # 'rotate 270 degrees counterclockwise': [DDActions.right],
                   }

ACTION_VERBS = HL_ACTION_VERBS if USE_HIGH_LEVEL_ACTIONS else LL_ACTION_VERBS

# COMPOSITIONAL1_VERBS = {
#                 'rotate 90 degrees counterclockwise': [DDActions.left],
#                 'spin 90 degrees counterclockwise': [DDActions.left],
# }

COMPOSITIONAL_VERBS = {
    'turn 270 degrees counterclockwise': [DDActions.right],
}

ALL_VERBS = ACTION_VERBS | COMPOSITIONAL_VERBS

DIRECTIONS_IDX_TO_STR = ['east', 'south', 'west', 'north']


class DirectionsDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=3, max_verbs=2, obs_type='grid', pretrain_version=True, **kwargs):
        self.size = size
        self.max_verbs = max_verbs
        self.obs_type = obs_type
        self.tile_size = 16
        self.pretrain_version = pretrain_version

        # Base sequences
        # base_sequences = []
        # for i in range(max_verbs + 1):
        pretrain_sequences = list(itertools.product((ACTION_VERBS | COMPOSITIONAL_VERBS).keys(), repeat=max_verbs))
        random.shuffle(pretrain_sequences)

        base_sequences = list(itertools.product(ACTION_VERBS.keys(), repeat=max_verbs))
        random.shuffle(base_sequences)

        comp_seqs = [seq for seq in itertools.product((ACTION_VERBS | COMPOSITIONAL_VERBS).keys(), repeat=max_verbs) if
                      any(v in COMPOSITIONAL_VERBS.keys() for v in seq)]
        random.shuffle(comp_seqs)
        comp_seqs = comp_seqs[:250]
        # Length sequences
        # longer_seqs = list(itertools.product(ACTION_VERBS.keys(), repeat=max_verbs + 1))
        # random.shuffle(longer_seqs)
        # longer_seqs = longer_seqs[:10000]
        if pretrain_version:
            pretrain_size, pretrain_val_size, train_size, val_size, test_size = 25000, 125, 50, 250, 2500
        else:
            train_size, val_size, test_size = 25000, 2500, 2500
        self.splits = {'train': base_sequences[:train_size],
                       'val': base_sequences[train_size:train_size + val_size],
                       'test': base_sequences[train_size + val_size:train_size + val_size + test_size],
                       'compositional': comp_seqs}
        if pretrain_version:
            self.splits['pretrain_val'] = pretrain_sequences[:pretrain_val_size]
            self.splits['pretrain'] = pretrain_sequences[pretrain_val_size:pretrain_val_size + pretrain_size]
            self.set_split('pretrain')
        else:
            self.set_split('train')

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[[0, 1, 2, 3], [seq for seq in self.splits['train']], [True, False]],
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

    def set_split(self, split):
        self.curr_split = split
        self.curr_idx = -1
        self.curr_dir = -1
        random.shuffle(self.splits[self.curr_split])

    @staticmethod
    def _gen_mission(starting_dir: int, sequence: str, inc_question: bool=False):
        mission = f'The robot is facing {DIRECTIONS_IDX_TO_STR[starting_dir]}'
        for i, verb in enumerate(sequence):
            if i == 0:
                mission += f'. They {verb}'
            else:
                mission += f', then they {verb}'
        # if inc_question:
        #     mission += '. You are now facing <mask>.'
        # else:
        mission += '. The robot is now facing>'
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
            # Get rid of object color and object status (only keep color)
            obs = self.gen_obs()['image'].transpose(2, 0, 1)[0]
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

        self.mission = self._gen_mission(self.agent_dir, self.curr_seq, inc_question=self.curr_split != 'pretrain')
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
            action = ALL_VERBS[curr_verb][self.curr_action_step]

        obs, reward, terminated, truncated, info = self.base_step(action)

        self.traj_obss.append(self.get_obs())
        self.traj_actions.append(action)

        self.curr_action_step += 1
        if len(self.curr_seq) == 0:
            terminated = True
            self.answer = f'{DIRECTIONS_IDX_TO_STR[self.agent_dir]}'
        elif self.curr_action_step >= len(ALL_VERBS[curr_verb]):
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

    env: MiniGridEnv = gym.make(args.env, max_verbs=args.max_verbs, obs_type=args.obs_type, tile_size=args.tile_size,
                                pretrain_version=args.pretrain)
    metadata = {'obs_type': env.obs_type, 'max_verbs': env.max_verbs, 'split_sizes': {}}

    # env = FullyObsWrapper(env)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    print('STARTING')
    env.reset(seed=args.seed)
    done = False
    while not done:
        _, _, done, _, _ = env.step(None)
    mission, obss, actions, answer = env.get_trajectory_info()
    #
    # print(mission)
    # print(answer)
    #
    # exit(0)

    window = Window("minigrid - " + str(env.__class__))

    dataset_name = f'{env.obs_type}_{env.max_verbs}verbs'
    if args.pretrain:
        dataset_name += '_pretrain'
    data_dir = Path(args.base_dir / 'directions_dataset' / 'data' / dataset_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    if env.obs_type == 'image':
        obs_shape = (3, env.tile_size * env.size, env.tile_size * env.size)
        obs_type = 'float32'
    elif env.obs_type == 'grid':
        obs_shape = (env.size, env.size, 3)
        obs_type = 'int'
    elif env.obs_type == 'simple':
        obs_shape = (4,)
        obs_type = 'int'
    else:
        obs_shape, obs_type = None, None

    memmap_max_obs = 100000

    metadata['memmap_shape'] = (memmap_max_obs, *obs_shape)
    metadata['memmap_type'] = obs_type
    for split in env.splits:  # 'val'
        with open(data_dir / f'{split}_dataset.txt', 'w') as dataset_file:
            offsets = [0]
            curr_observation_file_idx = 1
            curr_observation_memmap = np.memmap(str(data_dir / f'obss_{split}_{curr_observation_file_idx}.memmap'),
                                                dtype=obs_type, mode='w+', shape=(memmap_max_obs, *obs_shape))
            curr_obs_idx = 0
            env.set_split(split)
            num_instances = 4 * len(env.splits[split])  # for each direction
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

    with open(data_dir / f'metadata', 'w') as metadata_file:
        json.dump(metadata, metadata_file)
