from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.unstack import UnStack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.nut_disassembly import NutDisAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.pick_place_right import PickPlaceRight
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.door_close import Door_Close
from robosuite.environments.manipulation.old_door import Old_Door
from robosuite.environments.manipulation.old_door_close import Old_Door_Close
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_peg_removal import TwoArmPegRemoval
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.3.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
