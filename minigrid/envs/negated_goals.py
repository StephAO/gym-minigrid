from numpy import random
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid


class NegatedEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
            self,
            size=6,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            num_distractors=1,
            mode="TRAIN",
            mission_type="EITHER",
            training_type='all',  # One of: 1set, 2set, all
            use_color=True,
            types=
            {'things': ['key', 'box', 'ball', 'tree', 'cup', 'tool', 'building', 'crate', 'chair', 'flower'],
             'shapes': ['square', 'circle', 'oval', 'line', 'rectangle', 'diamond', 'ring', 'cross', 'star', 'arrow']},
            colors=['red', 'green', 'blue', 'purple', 'yellow', 'grey', 'white', 'cyan', 'brown', 'orange']
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_distractors = num_distractors

        self.training_type = training_type
        self.types = types
        self.use_color = use_color
        if self.use_color:
            self.colors = colors

        self.set_mode(mode, mission_type)
        self.setup_splits()

        self.base_templates = [
            # "The target is <not><the><desc>. The target is the<mask>.",
            "The target is <not><the><desc>.",
            "The <desc><obj> is <not>the target.",
            "The object to pick up is <not><the><desc>.",
            "The object that is <not><the><desc> must be picked up.",
            "Pick up the object that is <not><the><desc>.",
            "Get the object that is <not><the><desc>.",
            "<not><the><desc>.",
            "Navigate to the object that is <not><desc>",
            "Find the object that is <not><desc>",
            "The object that is <not><desc> is the goal",
        ]
        # Thought to better resemble cloze task: "The goal is not blue"

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.base_templates, self.colors, self.types['things'] + self.types['shapes'],
                                  self.colors, self.types['things'] + self.types['shapes'], [True, False]],
        )

        self.vocabulary = None

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=size * size + 5,  # max 2 turns at start + 1 middle turn, 1 end turn, 1 pick up
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def get_oracle_goal(self):
        return self.target_cell

    def get_vocabulary(self):
        if self.vocabulary:
            return self.vocabulary
        self.vocabulary = set()
        for template in self.base_templates:
            template = template.replace("<not><desc>", "")
            for word in template.split(" "):
                if word:
                    self.vocabulary.add(word)
        return self.vocabulary

    def set_mode(self, mode, mission_type):
        if mode not in ["TRAIN", "EVAL"]:
            raise ValueError("Unexpected value for mode")
        if mission_type not in ["DIRECT", "NEGATED", "EITHER", "TEST"]:
            raise ValueError("Unexpected value for mission_type")
        self.mode = mode
        self.mission_type = mission_type

    def setup_splits(self):
        half_len = len(self.types['shapes']) // 2
        if self.training_type == '1set':
            target_types_set_1 = self.types['shapes'][:half_len]
            target_types_set_2 = self.types['shapes'][half_len:]
            distra_types = self.types['shapes']
        elif self.training_type == '2set':
            target_types_set_1 = (self.types['shapes'][:half_len], self.types['things'][:half_len])
            target_types_set_2 = (self.types['shapes'][half_len:], self.types['things'][half_len:])
            distra_types = (self.types['shapes'], self.types['things'])
        elif self.training_type == 'all':
            target_types_set_1 = self.types['shapes'][:half_len] + self.types['things'][:half_len]
            target_types_set_2 = self.types['shapes'][half_len:] + self.types['things'][half_len:]
            distra_types = self.types['shapes'] + self.types['things']
        self.distra_types = distra_types

        if self.use_color:
            half_len = len(self.colors) // 2
            target_color_set_1 = self.colors[:half_len]
            target_color_set_2 = self.colors[half_len:]
            self.distr_colors = self.colors

        if self.mode == 'TRAIN':
            self.dir_target_types = target_types_set_1
            self.neg_target_types = target_types_set_2
            if self.use_color:
                self.dir_target_colors = target_color_set_1
                self.neg_target_colors = target_color_set_2
        elif self.mode == 'EVAL':
            self.dir_target_types = target_types_set_2
            self.neg_target_types = target_types_set_1
            if self.use_color:
                self.dir_target_colors = target_color_set_2
                self.neg_target_colors = target_color_set_1

    @staticmethod
    def _gen_mission(base_template: str, obj_color: str, obj_type: str, dist_color: str, dist_type: str, negated: bool):
        # Generate with necessary language output
        mission = base_template.replace("<not>", "not " if negated else "")
        if random.random() < 0.5:  # use color
            color = dist_color if negated else obj_color
            mission = mission.replace("<desc>", color)
            mission = mission.replace("<obj>", " object")
            mission = mission.replace("<the>", "")
        else:  # use object
            obj_type = dist_type if negated else obj_type
            mission = mission.replace("<the>", "the ")
            mission = mission.replace("<desc>", obj_type)
            mission = mission.replace("<obj>", "")
        # target_desc = f' {self.target_color} {self.target_type}'
        return mission


    def new_mission(self, negated: bool):
        target_types = self.neg_target_types if negated else self.dir_target_types
        target_colors = self.neg_target_colors if negated else self.dir_target_colors
        distra_types = self.distra_types
        distra_colors = self.distr_colors

        if self.training_type == '2set':
            set_idx = self._rand_int(0, 2)
            target_types = target_types[set_idx]
            distra_types = distra_types[set_idx]

        # 1. Choose a target type and color
        self.target_type = self._rand_elem(target_types)
        self.target_color = self._rand_elem(target_colors)
        target_obj = WorldObj.decode(OBJECT_TO_IDX[self.target_type], COLOR_TO_IDX[self.target_color], 0)
        self.target_cell = self.place_obj(target_obj)

        # 2. Create a distraction objects that cannot share the same tuple
        type_idx = distra_types.index(self.target_type)
        distractor_type_opts = distra_types[:type_idx] + distra_types[type_idx + 1:]
        color_idx = distra_colors.index(self.target_color)
        distractor_color_opts = distra_colors[:color_idx] + distra_colors[color_idx + 1:]
        dist_color = self._rand_elem(distractor_color_opts)
        dist_type = self._rand_elem(distractor_type_opts)
        dist = WorldObj.decode(OBJECT_TO_IDX[dist_type], COLOR_TO_IDX[dist_color], 0)
        self.place_obj(dist)

        template = self._rand_elem(self.base_templates)
        self.mission = self._gen_mission(template, self.target_color, self.target_type, dist_color, dist_type, negated)
        self.label = self.mission #.replace("<mask>", target_desc)

        return self.mission

    def object_test(self):
        for i, type in enumerate(self.types):
            if i == 0:
                obj = WorldObj.decode(OBJECT_TO_IDX[type], COLOR_TO_IDX['green'], 0)
                self.target_type = type
                self.target_color = 'green'
                self.target_cell = self.place_obj(obj)
            else:
                obj = WorldObj.decode(OBJECT_TO_IDX[type], COLOR_TO_IDX['red'], 0)
                self.place_obj(obj)

        # self.label = f' {self.target_color} {self.target_type} red square'

        template = self._rand_elem(self.base_templates)
        self.mission = self._gen_mission(template, None, None, self.target_color, self.target_type, True)
        self.label = self.mission
        return self.mission

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Randomize the player start position and orientation
        self.place_agent()

        if self.mission_type == "TEST":
            self.mission = self.object_test()
        else:
            negated = (self.mission_type == "EITHER" and self._rand_bool()) or self.mission_type == "NEGATED"
            self.mission = self.new_mission(negated)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if self.carrying.color == self.target_color and \
                    self.carrying.type == self.target_type:
                reward = 1
                terminated = True
            else:
                reward = -1
                terminated = True

        return obs, reward, terminated, False, info


class NegatedSimple(NegatedEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)




if __name__ == "__main__":
    base_templates = [
        "The target is <not><the><desc>.",
        "The target is the object that is <not><the><desc>.",
        "The <desc><obj> is <not>the target.",
        "The object to pick up is <not><the><desc>.",
        "The object that is <not><the><desc> must be picked up.",
        "Pick up the object that is <not><the><desc>.",
        "<not><the><desc>.",
        # "Navigate to the object that is <not><desc>",
        # "Find the object that is <not><desc>",
        # "The object that is <not><desc> is the goal",
    ]

    for template in base_templates:
        # COLOR
        mission = template.replace("<not>", "not ")
        mission = mission.replace("<desc>", 'red')
        mission = mission.replace("<obj>", " object")
        mission = mission.replace("<the>", "")
        mission = mission.capitalize()
        print(mission)

        mission = template.replace("<not>", "not ")
        mission = mission.replace("<the>", "the ")
        mission = mission.replace("<desc>", 'ball')
        mission = mission.replace("<obj>", "")
        mission = mission.capitalize()
        print(mission)