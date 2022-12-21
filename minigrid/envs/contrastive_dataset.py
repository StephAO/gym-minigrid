from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, OBJECT_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv


class ContrastiveDataset(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=7, numObjs=1, max_steps: int | None = None, **kwargs):

        self.numObjs = numObjs
        self.size = size
        # Types of objects to be generated

        self.obj_types = NON_BASE_OBJ_NAMES

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
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

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"A {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = self.obj_types

        objs = []
        obj_pos = []

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            obj_type = self._rand_elem(types)
            obj_color = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (obj_type, obj_color) in objs:
                continue

            obj = WorldObj.decode(OBJECT_TO_IDX[obj_type], COLOR_TO_IDX[obj_color], 0)

            pos = self.place_obj(obj)
            objs.append((obj_type, obj_color))
            obj_pos.append(pos)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        objIdx = self._rand_int(0, len(objs))
        self.targetType, self.target_color = objs[objIdx]
        self.target_pos = obj_pos[objIdx]

        self.label = f"A {self.target_color} {self.targetType}"
        self.mission = self.label
        # print(self.mission)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Toggle/pickup action terminates the episode
        if action == self.actions.toggle:
            terminated = True

        # Reward performing the done action next to the target object
        if action == self.actions.done:
            if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info



if __name__ == "__main__":
    import argparse
    import gym
    from minigrid.utils.window import Window
    from PIL import Image

    gym.register(
        id="ContrastiveDataset-v0",
        entry_point="minigrid.envs:ContrastiveDataset",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="ContrastiveDataset-v0"
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
        "--num-instances", type=int, help="size of dataset", default=10000
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

    for i in range(args.num_instances):
        env.reset(seed=args.seed)
        # if hasattr(env, "mission"):
            # print("Mission: %s" % env.mission)
            # window.set_caption(env.mission)

        frame = env.get_frame(agent_pov=args.agent_view, highlight=False)



        img = Image.fromarray(frame)
        img.save(f'contrastive_dataset/{env.mission}.({i}).png')
        #
        # print(type(frame))
        # window.show_img(frame)
        #
        # window.show(block=False)

        # input('hit enter')



