from __future__ import annotations

import numpy as np

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'white' : np.array([255, 255, 255]),
    'cyan' : np.array([0, 255, 255]),
    'brown' : np.array([139, 69, 19]),
    'orange' : np.array([255, 99, 71])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 1,
    'green' : 2,
    'blue'  : 3,
    'purple': 4,
    'yellow': 5,
    'grey'  : 6,
    'white' : 7,
    'cyan'  : 8,
    'brown' : 9,
    'orange': 10
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    # Base objects
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,

    'block'         : 5,
    'north'         : 6,
    'east'          : 7,
    'south'         : 8,
    'west'          : 9,

    'agent'         : 10,
    'goal'          : 31,
    'lava'          : 32,
    'gripped_block' : 33,

    # Shapes
    'square'        : 11,
    'circle'        : 12,
    'oval'          : 13,
    'line'          : 14,
    'rectangle'     : 15,
    'diamond'       : 16,
    'ring'          : 17,
    'cross'         : 18,
    'star'          : 19,
    'arrow'         : 20,
    # Things
    'key'           : 21,
    'ball'          : 22,
    'box'           : 23,
    'tree'          : 24,
    'cup'           : 25,
    'tool'          : 26,
    'building'      : 27,
    'crate'         : 28,
    'chair'         : 29,
    'flower'        : 30,

}

OBJECT_NAMES = sorted(list(OBJECT_TO_IDX.keys()))
NON_BASE_OBJ_NAMES = [o for o in OBJECT_NAMES if o not in ['unseen','empty','wall','floor','door','goal','lava','agent']]

# TODO : shapes: 'line', 'rectangle', 'diamond', 'cross', 'star'
# TODO : object: 'tool', 'tree',  'flower', 'cup', '
# REDO : 'ball', 'box'


IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)), # EAST
    # Down (positive Y)
    np.array((0, 1)), # SOUTH
    # Pointing left (negative X)
    np.array((-1, 0)), # WEST
    # Up (negative Y)
    np.array((0, -1)), # NORTH
]

