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
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'white' : 6,
    'cyan'  : 7,
    'brown' : 8,
    'orange': 9
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

    'goal'          : 5,
    'lava'          : 6,
    'agent'         : 7,
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
    'flower'        : 30
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
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

