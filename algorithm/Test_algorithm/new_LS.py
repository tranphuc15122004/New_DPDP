import sys
import math
from typing import Dict , List
import copy
import time
from algorithm.Object import Node, Vehicle, OrderItem
from algorithm.algorithm_config import *
from algorithm.engine import *
from algorithm.local_search import * 

def Improve_by_relocate_delay_order(vehicleid_to_plan: Dict[str , List[Node]], id_to_vehicle: Dict[str , Vehicle] , route_map: Dict[tuple , tuple] , is_limited : bool = False ):
    pass