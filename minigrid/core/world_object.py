from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    point_in_oval
)

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal(color)
        elif obj_type == 'lava':
            v = Lava(color)

        # Shapes
        elif obj_type == 'circle':
            v = Circle(color)
        elif obj_type == 'square':
            v = Square(color)
        elif obj_type == 'oval':
            v = Oval(color)
        elif obj_type == 'line':
            v = Line(color)
        elif obj_type == 'rectangle':
            v = Rectangle(color)
        elif obj_type == 'diamond':
            v = Diamond(color)
        elif obj_type == 'ring':
            v = Ring(color)
        elif obj_type == 'star':
            v = Star(color)
        elif obj_type == 'cross':
            v = Cross(color)
        elif obj_type == 'arrow':
            v = Arrow(color)

        # Things
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'tree':
            v = Tree(color)
        elif obj_type == 'cup':
            v = Cup(color)
        elif obj_type == 'tool':
            v = Tool(color)
        elif obj_type == 'building':
            v = Building(color)
        elif obj_type == 'crate':
            v = Crate(color)
        elif obj_type == 'chair':
            v = Chair(color)
        elif obj_type == 'flower':
            v = Flower(color)
        elif obj_type == 'crate':
            v = Crate(color)
        elif obj_type == 'north':
            v = North(color)
        elif obj_type == 'east':
            v = East(color)
        elif obj_type == 'south':
            v = South(color)
        elif obj_type == 'west':
            v = West(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """
    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

# Shapes
class Square(WorldObj):
    def __init__(self, color):
        super(Square, self).__init__('square', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Outline
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), c)

class Circle(WorldObj):
    def __init__(self, color='blue'):
        super(Circle, self).__init__('circle', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Oval(WorldObj):
    def __init__(self, color='blue'):
        super(Oval, self).__init__('oval', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_oval(0.5, 0.5, 0.4, 0.2), COLORS[self.color])

class Line(WorldObj):
    def __init__(self, color='blue'):
        super(Line, self).__init__('line', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.1, 0.9, 0.45, 0.55), COLORS[self.color])

class Rectangle(WorldObj):
    def __init__(self, color='blue'):
        super(Rectangle, self).__init__('rectangle', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.3, 0.7, 0.1, 0.9), COLORS[self.color])

class Diamond(WorldObj):
    def __init__(self, color='blue'):
        super(Diamond, self).__init__('diamond', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_triangle( (0.5, 0.25), (0.5, 0.75), (0.85, 0.5) ), COLORS[self.color])
        fill_coords(img, point_in_triangle( (0.5, 0.25), (0.5, 0.75), (0.15, 0.5) ), COLORS[self.color])

class Ring(WorldObj):
    def __init__(self, color='blue'):
        super(Ring, self).__init__('ring', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.15), (0,0,0))

class Star(WorldObj):
    def __init__(self, color='blue'):
        super(Star, self).__init__('star', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_triangle((0.15, 0.3), (0.85, 0.3), (0.5, 0.9)), COLORS[self.color])
        fill_coords(img, point_in_triangle((0.15, 0.7), (0.85, 0.7), (0.5, 0.1)), COLORS[self.color])

class Cross(WorldObj):
    def __init__(self, color='blue'):
        super(Cross, self).__init__('cross', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.4, 0.6, 0.1, 0.9), COLORS[self.color])
        fill_coords(img, point_in_rect(0.1, 0.9, 0.4, 0.6), COLORS[self.color])

class Arrow(WorldObj):
    def __init__(self, color='blue'):
        super(Arrow, self).__init__('arrow', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.1, 0.6, 0.4, 0.6), COLORS[self.color])
        fill_coords(img, point_in_triangle((0.6, 0.25), (0.9, 0.5), (0.6, 0.75)), COLORS[self.color])


# Things
class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
        # Cross to differentiate from circle
        fill_coords(img, point_in_rect(0.19, 0.81, 0.45, 0.55), (0, 0, 0))
        fill_coords(img, point_in_rect(0.45, 0.55, 0.19, 0.81), (0, 0, 0))

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class Tree(WorldObj):
    def __init__(self, color):
        super(Tree, self).__init__('tree', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Trunk
        fill_coords(img, point_in_rect(0.4, 0.6, 0.8, 0.9), c)
        # Leaves
        fill_coords(img, point_in_triangle((0.1, 0.8), (0.9, 0.8), (0.5, 0.5)), c)
        fill_coords(img, point_in_triangle((0.2, 0.6), (0.8, 0.6), (0.5, 0.3)), c)
        fill_coords(img, point_in_triangle((0.3, 0.4), (0.7, 0.4), (0.5, 0.1)), c)

class Cup(WorldObj):
    def __init__(self, color):
        super(Cup, self).__init__('cup', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Handle
        fill_coords(img, point_in_circle(0.7, 0.5, 0.2), c)
        fill_coords(img, point_in_circle(0.7, 0.5, 0.1), (0, 0, 0))
        # Body
        fill_coords(img, point_in_rect(0.15, 0.7, 0.2, 0.8), c)

class Tool(WorldObj):
    def __init__(self, color):
        super(Tool, self).__init__('tool', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Handle
        fill_coords(img, point_in_rect(0.45, 0.55, 0.15, 0.85), COLORS[self.color])
        # Head
        fill_coords(img, point_in_rect(0.25, 0.75, 0.15, 0.45), COLORS[self.color])

class Building(WorldObj):
    def __init__(self, color):
        super(Building, self).__init__('building', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Main house
        fill_coords(img, point_in_rect(0.2, 0.8, 0.5, 0.8), COLORS[self.color])
        # Door
        fill_coords(img, point_in_rect(0.45, 0.55, 0.6, 0.8), (0, 0, 0))
        # Roof
        fill_coords(img, point_in_triangle((0.1, 0.5), (0.9, 0.5), (0.5, 0.1)), COLORS[self.color])


class Crate(WorldObj):
    def __init__(self, color):
        super(Crate, self).__init__('crate', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Outline
        fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), c)
        # Slats
        fill_coords(img, point_in_rect(0.15, 0.25, 0.15, 0.85), (0, 0, 0))
        fill_coords(img, point_in_rect(0.30, 0.40, 0.15, 0.85), (0, 0, 0))
        fill_coords(img, point_in_rect(0.45, 0.55, 0.15, 0.85), (0, 0, 0))
        fill_coords(img, point_in_rect(0.60, 0.70, 0.15, 0.85), (0, 0, 0))
        fill_coords(img, point_in_rect(0.75, 0.85, 0.15, 0.85), (0, 0, 0))

class Chair(WorldObj):
    def __init__(self, color):
        super(Chair, self).__init__('chair', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Back
        fill_coords(img, point_in_rect(0.2, 0.3, 0.15, 0.85), c)
        # Seat
        fill_coords(img, point_in_rect(0.2, 0.8, 0.45, 0.55), c)
        # Leg
        fill_coords(img, point_in_rect(0.7, 0.8, 0.5, 0.85), c)

class Flower(WorldObj):
    def __init__(self, color):
        super(Flower, self).__init__('flower', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Stem
        fill_coords(img, point_in_rect(0.47, 0.53, 0.5, 0.85), c)
        # Flower center
        fill_coords(img, point_in_circle(0.5, 0.3, 0.05), c)
        # Petals
        fill_coords(img, point_in_circle(0.66, 0.3, 0.07), c)
        fill_coords(img, point_in_circle(0.58, 0.16, 0.07), c)
        fill_coords(img, point_in_circle(0.42, 0.16, 0.07), c)
        fill_coords(img, point_in_circle(0.34, 0.3, 0.07), c)
        fill_coords(img, point_in_circle(0.42, 0.44, 0.07), c)
        fill_coords(img, point_in_circle(0.58, 0.44, 0.07), c)


class North(WorldObj):
    def __init__(self, color):
        super(North, self).__init__('north', color)

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        # Left Vertical
        fill_coords(img, point_in_rect(0.2, 0.3, 0.2, 0.8), c)
        # Right Vertical
        fill_coords(img, point_in_rect(0.7, 0.8, 0.2, 0.8), c)
        # Diagonal
        fill_coords(img, point_in_rect(0.6, 0.7, 0.65, 0.8), c)
        fill_coords(img, point_in_rect(0.5, 0.6, 0.5, 0.65), c)
        fill_coords(img, point_in_rect(0.4, 0.5, 0.35, 0.5), c)
        fill_coords(img, point_in_rect(0.3, 0.4, 0.2, 0.35), c)

class East(WorldObj):
    def __init__(self, color):
        super(East, self).__init__('east', color)

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        # Left Vertical
        fill_coords(img, point_in_rect(0.2, 0.3, 0.2, 0.8), c)
        # Top Horizontal
        fill_coords(img, point_in_rect(0.2, 0.8, 0.7, 0.8), c)
        # Middle Horizontal
        fill_coords(img, point_in_rect(0.2, 0.5, 0.45, 0.55), c)
        # Bottom Horizontal
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.3), c)

class South(WorldObj):
    def __init__(self, color):
        super(South, self).__init__('south', color)

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        # Top Horizontal
        fill_coords(img, point_in_rect(0.2, 0.8, 0.7, 0.8), c)
        # Left Vertical
        fill_coords(img, point_in_rect(0.2, 0.3, 0.3, 0.55), c)
        # Middle Horizontal
        fill_coords(img, point_in_rect(0.2, 0.8, 0.45, 0.55), c)
        # Right Vertical
        fill_coords(img, point_in_rect(0.7, 0.8, 0.45, 0.7), c)
        # Bottom Horizontal
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.3), c)

class West(WorldObj):
    def __init__(self, color):
        super(West, self).__init__('west', color)

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        # Left Vertical
        fill_coords(img, point_in_rect(0.2, 0.3, 0.2, 0.8), c)
        # Right Vertical
        fill_coords(img, point_in_rect(0.7, 0.8, 0.2, 0.8), c)
        # Diagonals
        fill_coords(img, point_in_rect(0.30, 0.38, 0.6, 0.75), c)
        fill_coords(img, point_in_rect(0.38, 0.46, 0.5, 0.65), c)
        fill_coords(img, point_in_rect(0.46, 0.54, 0.4, 0.55), c)
        fill_coords(img, point_in_rect(0.54, 0.62, 0.5, 0.65), c)
        fill_coords(img, point_in_rect(0.62, 0.70, 0.6, 0.75), c)





if __name__ == "__main__":
    import gym_minigrid.window

    window = gym_minigrid.window.Window('gym_minigrid')
    window.show(block=False)
    tile_size = 32
    subdivs = 3
    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    obj = Crate('red')
    obj.render(img)

    window.show_img(img)
    from time import sleep
    sleep(5)