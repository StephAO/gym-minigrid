from numpy import mat
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=6,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        num_distractors=1,
        split = 1.0,
        mode = "TRAIN",
        mission_type = "EITHER",
        types = ('key', 'ball', 'box', 'square', 'crate', 'circle'),
        colors = ('red', 'green', 'blue', 'purple', 'yellow', 'grey', 'white', 'cyan', 'brown', 'orange')
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_distractors = num_distractors

        self.types = types
        self.colors = colors

        self.type_dict = {
            'key': Key,
            'ball': Ball,
            'box': Box,
            'crate': Crate,
            'square': Square,
            'circle': Circle
        }

        self.base_templates = [
            "The target is <not><desc>."
            "The target is object that is <not><desc>.",
            "Pick up the <not><desc><obj>.",
            "Pick up the object that is <not><desc>.",
            "<not><desc>.",
            "The <desc><obj> is <not>the target."
            # "Navigate to the object that is <not><desc>",
            # "Find the object that is <not><desc>",
            # "The object that is <not><desc> is the goal",
        ]
        # Thought to better resemble cloze task: "The goal is not blue"

        self.vocabulary = None

        self.split = split
        self.set_mode(mode, mission_type)

        super().__init__(
            grid_size=size,
            max_steps=size*size + 5, # max 2 turns at start + 1 middle turn, 1 end turn, 1 pick up
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
        if mission_type not in ["DIRECT", "NEGATED", "EITHER"]:
            raise ValueError("Unexpected value for mission_type")

        self.mode = mode
        self.mission_type = mission_type

    def compute_indices(self, length):
        if self.mode == "TRAIN":
            return 0, math.floor(self.split * length)
        else:
            return math.floor(self.split * length), length

    def direct_mission(self):
        # 1. Choose a target type and color
        # 2. Create N distraction objects that cannot share the same tuple
        type_idx = self._rand_int(*self.compute_indices(len(self.types)))
        # Updated the random selection to limit the indices accoring to the mode
        # Train: 0 -> len * split(exclusive)
        # Eval: len * split -> len(exclusive)
        color_idx = self._rand_int(*self.compute_indices(len(self.colors)))
        self.target_type = self.types[type_idx]
        self.target_color = self.colors[color_idx]
        target_obj = self.type_dict[self.target_type](self.target_color)
        self.target_cell = self.place_obj(target_obj)

        distractor_type_opts = self.types[:type_idx] + self.types[type_idx + 1:]
        distractor_color_opts = self.colors[:color_idx] + self.colors[color_idx + 1:]

        for d in range(self.num_distractors):
            obj_type = self._rand_elem(distractor_type_opts)
            obj_color = self._rand_elem(distractor_color_opts)

            obj = self.type_dict[obj_type](obj_color)

            self.place_obj(obj)

        template = self._rand_elem(self.base_templates)
        mission = template.replace("<not>", "")
        if False: # self._rand_bool(): # use color
            mission = mission.replace("<desc>", self.target_color)
            mission = mission.replace("<obj>", " object")
            mission = mission.replace("<the>", "")
        else: # use object
            mission = mission.replace("<the>", "the ")
            mission = mission.replace("<desc>", self.target_type)
            mission = mission.replace("<obj>", "")
        return mission.capitalize()

    def negated_mission(self):
        # 1. Choose a target type and color
        # 2. Choose a negated description
        # 3. Create N distraction objects that share the negated description
        type_idx = self._rand_int(0, len(self.types))
        # Updated the random selection to limit the indices accoring to the mode
        # Train: 0 -> len * split(exclusive)
        # Eval: len * split -> len(exclusive)
        color_idx = self._rand_int(*self.compute_indices(len(self.colors)))
        self.target_type = self.types[type_idx]
        self.target_color = self.colors[color_idx]
        target_obj = self.type_dict[self.target_type](self.target_color)
        self.target_cell = self.place_obj(target_obj)

        distractor_type_opts = self.types[:type_idx] + self.types[type_idx + 1:]
        distractor_color_opts = self.colors[:color_idx] + self.colors[color_idx + 1:]

        dist_color = self._rand_elem(distractor_color_opts)

        for d in range(self.num_distractors):
            obj_type = self._rand_elem(distractor_type_opts)
            obj = self.type_dict[obj_type](dist_color)
            self.place_obj(obj)

        template = self._rand_elem(self.base_templates)
        mission = template.replace("<not>", "not ")
        if False: #self._rand_bool():  # use color
            mission = mission.replace("<desc>", self.target_color)
            mission = mission.replace("<obj>", " object")
            mission = mission.replace("<the>", "")
        else:  # use object
            mission = mission.replace("<the>", "the ")
            mission = mission.replace("<desc>", self.target_type)
            mission = mission.replace("<obj>", "")
        return mission.capitalize()


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Randomize the player start position and orientation
        self.place_agent()

        if (self.mission_type == "EITHER" and self._rand_bool()) or self.mission_type == "DIRECT":
            self.mission = self.direct_mission()
        else: # mission_type == "NEGATED" or rand bool returned False
            self.mission = self.negated_mission()

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.target_color and \
                    self.carrying.type == self.target_type:
                reward = 1
                done = True
            else:
                reward = -1
                done = True

        return obs, reward, done, info

class NegatedSimple(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)

register(
    id='MiniGrid-Negated-Simple-v0',
    entry_point='gym_minigrid.envs:NegatedSimple'
)

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
