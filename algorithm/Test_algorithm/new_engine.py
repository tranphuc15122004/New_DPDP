import copy
from datetime import datetime
import json
import math
import os
import random
import re
import sys
from algorithm.algorithm_config import *
from typing import Dict , List, Optional, Tuple
from algorithm.Object import *
from algorithm.engine import *

def new_dispatch_new_orders(vehicleid_to_plan: Dict[str , list[Node]] ,  id_to_factory:Dict[str , Factory] , route_map: Dict[tuple , tuple] ,  id_to_vehicle: Dict[str , Vehicle] , id_to_unlocated_items:Dict[str , OrderItem], new_order_itemIDs: list[str]):
    if new_order_itemIDs:
        orderId_to_Item : Dict[str , list[OrderItem]] = {}
        for new_order_item in new_order_itemIDs:
            new_item = id_to_unlocated_items.get(new_order_item)
            orderID  = new_item.order_id
            if orderID not in orderId_to_Item:
                orderId_to_Item[orderID] = []
            orderId_to_Item.get(orderID).append(new_item)

        # Kiểm tra xem đơn ở các xe có bị lặp hay không
        for orderID in orderId_to_Item.keys():
            # Xóa duplicate items trong cùng một order
            items = orderId_to_Item[orderID]
            seen_ids = set()
            unique_items = []
            for item in items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    unique_items.append(item)
            orderId_to_Item[orderID] = unique_items
            
        
        for vehicle in id_to_vehicle.values():
            capacity = vehicle.board_capacity
            break
        
        for orderID , orderID_items in orderId_to_Item.items():
            order_demand = 0
            for item in orderID_items:
                order_demand += item.demand
            
            if order_demand > capacity:
                tmp_demand = 0
                tmp_itemList: list[OrderItem] = []
                
                for item in orderID_items:
                    tmp_itemList = [item]
                    
                    node_list: list[Node] = create_Pickup_Delivery_nodes(tmp_itemList , id_to_factory)
                    isExhausive = False
                    route_node_list : List[Node] = []
                    
                    if node_list:
                        isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(node_list , id_to_vehicle , vehicleid_to_plan , route_map)
                    
                    route_node_list = vehicleid_to_plan.get(bestInsertVehicleID , [])

                    if isExhausive:
                        route_node_list = bestNodeList[:]
                    else:
                        if route_node_list is None:
                            route_node_list = []
                        
                        new_order_pickup_node = node_list[0]
                        new_order_delivery_node = node_list[1]
                        
                        route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                        route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                    vehicleid_to_plan[bestInsertVehicleID] = route_node_list
            else:
                node_list: list[Node] = create_Pickup_Delivery_nodes(copy.deepcopy(orderID_items) , id_to_factory)
                
                isExhausive = False
                if node_list:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(node_list , id_to_vehicle , vehicleid_to_plan , route_map)
                route_node_list : List[Node] = vehicleid_to_plan.get(bestInsertVehicleID , [])
                if isExhausive:
                    route_node_list = bestNodeList[:]
                else:
                    if route_node_list is None:
                        route_node_list = []
                    
                    new_order_pickup_node = node_list[0]
                    new_order_delivery_node = node_list[1]
                    
                    route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                    route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                vehicleid_to_plan[bestInsertVehicleID] = route_node_list