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
    all_exhautive = True
    
    if new_order_itemIDs:
        orderId_to_Item : Dict[str , list[OrderItem]] = {}
        for new_order_item in new_order_itemIDs:
            new_item = id_to_unlocated_items.get(new_order_item)
            orderID  = new_item.order_id
            if orderID not in orderId_to_Item:
                orderId_to_Item[orderID] = []
            orderId_to_Item.get(orderID).append(new_item)
        
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
                    if (tmp_demand + item.demand) > capacity:
                        for plan in vehicleid_to_plan.values():
                            if len(plan) >= 6: all_exhautive = False
                        
                        
                        node_list: list[Node] = create_Pickup_Delivery_nodes(copy.deepcopy(tmp_itemList) , id_to_factory)
                        isExhausive = False
                        route_node_list : List[Node] = []
                        
                        if node_list:
                            isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(node_list , id_to_vehicle , vehicleid_to_plan , route_map)
                        
                        route_node_list = vehicleid_to_plan.get(bestInsertVehicleID , [])

                        if isExhausive:
                            route_node_list = bestNodeList[:]
                        else:
                            all_exhautive = False
                            if route_node_list is None:
                                route_node_list = []
                            
                            new_order_pickup_node = node_list[0]
                            new_order_delivery_node = node_list[1]
                            
                            route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                            route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                        vehicleid_to_plan[bestInsertVehicleID] = route_node_list
                        
                        tmp_itemList.clear()
                        tmp_demand = 0
                    tmp_itemList.append(item)
                    tmp_demand += item.demand 

                if len(tmp_itemList) > 0:
                    for plan in vehicleid_to_plan.values():
                        if len(plan) >= 6: all_exhautive = False
                    
                    node_list: list[Node] = create_Pickup_Delivery_nodes(copy.deepcopy(tmp_itemList) , id_to_factory)
                    isExhausive = False
                    
                    if node_list:
                        isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList =  dispatch_nodePair(node_list , id_to_vehicle , vehicleid_to_plan, route_map)
                    route_node_list : List[Node] = vehicleid_to_plan.get(bestInsertVehicleID , [])
                    
                    if isExhausive:
                        route_node_list = bestNodeList[:]
                    else:
                        all_exhautive = False
                        
                        if route_node_list is None:
                            route_node_list = []
                        
                        new_order_pickup_node = node_list[0]
                        new_order_delivery_node = node_list[1]
                        
                        route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                        route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                    vehicleid_to_plan[bestInsertVehicleID] = route_node_list
            else:
                for plan in vehicleid_to_plan.values():
                    if len(plan) >= 6: all_exhautive = False
                
                node_list: list[Node] = create_Pickup_Delivery_nodes(copy.deepcopy(orderID_items) , id_to_factory)
                
                isExhausive = False
                if node_list:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(node_list , id_to_vehicle , vehicleid_to_plan , route_map)
                route_node_list : List[Node] = vehicleid_to_plan.get(bestInsertVehicleID , [])
                if isExhausive:
                    route_node_list = bestNodeList[:]
                else:
                    all_exhautive = False
                    
                    if route_node_list is None:
                        route_node_list = []
                    
                    new_order_pickup_node = node_list[0]
                    new_order_delivery_node = node_list[1]
                    
                    route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                    route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                vehicleid_to_plan[bestInsertVehicleID] = route_node_list
    
    return all_exhautive            

def generate_random_chromosome(initial_vehicleid_to_plan : Dict[str , List[Node]],  route_map: Dict[Tuple, Tuple], id_to_vehicle: Dict[str, Vehicle], Unongoing_super_nodes : Dict[int , Dict[str, Node]]  ,Base_vehicleid_to_plan : Dict[str , List[Node]] , quantity : int):
    ls_node_pair_num = len(Unongoing_super_nodes)
    if ls_node_pair_num == 0:
        return None , None

    #Quan the
    population : List[Chromosome] = []
    number_of_node = 0
    for plan in initial_vehicleid_to_plan.values():
        number_of_node += len(plan)
    
    pdg_Map : Dict[str , List[Node]] = {}
    
    # tao Dict cac super node
    for idx, pdg in Unongoing_super_nodes.items():
        pickup_node = None
        delivery_node = None
        node_list: List[Node] = []
        pos_i = 0
        pos_j = 0
        d_num = len(pdg) // 2
        index = 0

        if pdg:
            vehicleID = ''
            for v_and_pos_str, node in (pdg.items()):
                vehicleID = v_and_pos_str.split(",")[0]
                if index % 2 == 0:
                    pos_i = int(v_and_pos_str.split(",")[1])
                    pickup_node = node
                    node_list.insert(0, pickup_node)
                    index += 1
                else:
                    pos_j = int(v_and_pos_str.split(",")[1])
                    delivery_node = node
                    node_list.append(delivery_node)
                    index += 1
                    pos_j = int(pos_j - d_num + 1)
            
            k : str = f"{vehicleID},{int(pos_i)}+{int(pos_j)}"
            pdg_Map[k] = node_list
    if len(pdg_Map) < 2:
        return None , None
    
    # Tao quan the
    while len(population) < quantity:
        temp_route: Dict[str , List[Node]] = {}
        for vehicleID , plan in Base_vehicleid_to_plan.items():
            temp_route[vehicleID] = []
            for node in plan:
                temp_route[vehicleID].append(node)
        
        # Chen ngau nhien cac super node vao cac lo trinh cua cac xe 
        for DPG in pdg_Map.values():
            # Khai bao cac bien lien quan
            # chen vao sau cac tuyen duong
            if random.uniform(0 , 1) <= 0.25:
                isExhausive = False
                route_node_list : List[Node] = []
                selected_vehicleID = random.choice(list(Base_vehicleid_to_plan.keys()))
                if DPG:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(DPG , id_to_vehicle , temp_route , route_map , selected_vehicleID)
                
                route_node_list = temp_route.get(bestInsertVehicleID , [])

                if isExhausive:
                    route_node_list = bestNodeList[:]
                else:
                    if route_node_list is None:
                        route_node_list = []
                    
                    new_order_pickup_node = DPG[0]
                    new_order_delivery_node = DPG[1]
                    
                    route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                    route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                temp_route[bestInsertVehicleID] = route_node_list
            else:
                if random.uniform(0 , 1) <= 0.5:
                    selected_vehicleID = random.choice(list(id_to_vehicle.keys()))
                    selected_vehicle = id_to_vehicle[selected_vehicleID]
                    
                    temp_route[selected_vehicleID].extend(DPG)
                else:
                    random_dispatch_nodePair(DPG , id_to_vehicle , temp_route)
                
        # Da tao xong mot ca the moi
        if len(temp_route) == len(id_to_vehicle):
            temp = 0
            for vehicle_route in temp_route.values():
                temp += len(vehicle_route)
            if temp == number_of_node:
                population.append(Chromosome(temp_route , route_map , id_to_vehicle ))
    population.append(Chromosome(initial_vehicleid_to_plan , route_map , id_to_vehicle))
    return population , pdg_Map 

def calculate_diversity(population: List[Chromosome]) -> float:
    """Tính độ đa dạng của quần thể dựa trên sự khác biệt về route"""
    if len(population) < 2:
        return 1.0
    
    total_distance = 0
    count = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = calculate_chromosome_distance(population[i], population[j])
            total_distance += distance
            count += 1
    
    return total_distance / count if count > 0 else 0.0

def calculate_chromosome_distance(c1: Chromosome, c2: Chromosome) -> float:
    """Tính khoảng cách giữa 2 chromosome dựa trên route structure"""
    distance = 0
    total_positions = 0
    
    # So sánh route của từng vehicle
    for vehicle_id in c1.solution.keys():
        route1 = c1.solution.get(vehicle_id, [])
        route2 = c2.solution.get(vehicle_id, [])
        
        max_len = max(len(route1), len(route2))
        total_positions += max_len
        
        # Đếm số vị trí khác nhau
        for i in range(max_len):
            node1_id = route1[i].id if i < len(route1) else None
            node2_id = route2[i].id if i < len(route2) else None
            
            if node1_id != node2_id:
                distance += 1
    
    # Normalize distance
    return distance / total_positions if total_positions > 0 else 0.0

def calculate_fitness_diversity(population: List[Chromosome]) -> float:
    """Tính độ đa dạng dựa trên fitness values"""
    if len(population) < 2:
        return 1.0
    
    fitness_values = [c.fitness for c in population]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    
    # Tính độ lệch chuẩn
    variance = sum((f - avg_fitness) ** 2 for f in fitness_values) / len(fitness_values)
    std_dev = math.sqrt(variance)
    
    # Normalize bằng average fitness
    return std_dev / avg_fitness if avg_fitness > 0 else 0.0

def generate_single_random_chromosome(Base_vehicleid_to_plan: Dict[str, List[Node]], 
                                    route_map: Dict[Tuple, Tuple],
                                    id_to_vehicle: Dict[str, Vehicle],
                                    PDG_map: Dict[str, List[Node]]) -> Chromosome:
    """Tạo một cá thể ngẫu nhiên"""
    temp_route: Dict[str, List[Node]] = {}
    
    # Copy base route
    for vehicleID, plan in Base_vehicleid_to_plan.items():
        temp_route[vehicleID] = [node for node in plan]
    
    # Random dispatch các PDG
    for DPG in PDG_map.values():
        if random.uniform(0 , 1) < 0.5:
            isExhausive = False
            route_node_list : List[Node] = []
            if DPG:
                isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(DPG , id_to_vehicle , temp_route , route_map)
            
            route_node_list = temp_route.get(bestInsertVehicleID , [])

            if isExhausive:
                route_node_list = bestNodeList[:]
            else:
                if route_node_list is None:
                    route_node_list = []
                
                new_order_pickup_node = DPG[0]
                new_order_delivery_node = DPG[1]
                
                route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
            temp_route[bestInsertVehicleID] = route_node_list
        else:
            if random.uniform(0 , 1) < 0.5:
                # Random vehicle
                selected_vehicleID = random.choice(list(id_to_vehicle.keys()))
                temp_route[selected_vehicleID].extend(DPG)
            else:
                # Random dispatch
                random_dispatch_nodePair(DPG, id_to_vehicle, temp_route)
        
    return Chromosome(temp_route, route_map, id_to_vehicle)

def remove_similar_individuals(population: List[Chromosome], threshold: float = 0.1) -> List[Chromosome]:
    """Loại bỏ các cá thể quá giống nhau"""
    unique_population = []
    
    for individual in population:
        is_unique = True
        
        for unique_individual in unique_population:
            distance = calculate_chromosome_distance(individual, unique_individual)
            if distance < threshold:
                # Giữ cá thể có fitness tốt hơn
                if individual.fitness < unique_individual.fitness:
                    unique_population.remove(unique_individual)
                    unique_population.append(individual)
                is_unique = False
                break
        
        if is_unique:
            unique_population.append(individual)
    
    return unique_population