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
import algorithm.algorithm_config as config
from algorithm.local_search import *
from algorithm.Test_algorithm.new_LS import *

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


def new_generate_random_chromosome(initial_vehicleid_to_plan : Dict[str , List[Node]],  route_map: Dict[Tuple, Tuple], id_to_vehicle: Dict[str, Vehicle], Unongoing_super_nodes : Dict[int , Dict[str, Node]]  ,Base_vehicleid_to_plan : Dict[str , List[Node]] , quantity : int):
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
    
    while len(population) < quantity:
        new_individual = disturbance_opt(initial_vehicleid_to_plan , id_to_vehicle , route_map , 0.5)
        if new_individual:
            population.append(new_individual)
    
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
            unique_population.append(copy.deepcopy(individual))
    
    return unique_population


def get_adaptive_order(indivisual: Chromosome ,methods : Dict , mode= 1) -> List[str]:
    """Tạo thứ tự adaptive dựa trên improved_LS_map của cá thể"""
    
    # Lấy lịch sử cải thiện từ cá thể
    improvement_history = indivisual.improved_LS_map.copy()
    
    # Tính score cho từng phương pháp (với một chút randomness)
    method_scores = {}
    total_improvements = sum(improvement_history.values())
    
    for method_name in methods.keys():
        # Base score từ lịch sử cải thiện
        improvement_count = improvement_history.get(method_name, 0)
        
        if total_improvements > 0:
            # Success rate của method này
            success_rate = improvement_count / total_improvements
            method_scores[method_name] = success_rate * random.uniform(0.8 , 1.2)
        else:
            # Nếu chưa có lịch sử, dùng random weights
            method_scores[method_name] = random.uniform(0.5, 1.0)
    
    # Sắp xếp theo score giảm dần (method tốt nhất trước)
    if mode == 1:
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1])
        
    # Trả về danh sách tên methods theo thứ tự ưu tiên
    ordered_methods = [method[0] for method in sorted_methods]
    return ordered_methods

def randon_1_LS(indivisual: Chromosome , is_limited = False , mode = 0):
    if config.is_timeout():
        return False
    
    i = 1
    
    # Dictionary các phương pháp Local Search
    methods = {
        'PDPairExchange': lambda: new_inter_couple_exchange(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'BlockExchange': lambda: new_block_exchange(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'BlockRelocate': lambda: new_block_relocate(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'mPDG': lambda: new_multi_pd_group_relocate(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        '2opt': lambda: improve_ci_path_by_2_opt(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited)
    }
    
    # Counter cho từng phương pháp
    counters = {name: 0 for name in methods.keys()}
    
    # Lấy thứ tự adaptive
    if mode == 0:
        chosen_method = random.choices(config.LS_METHODS)[0]
    else:
        chosen_method = get_adaptive_order(indivisual , methods , mode=mode)[0]
    i = 0
    begin_LS_time = time.time()
    while i < 1:
        if config.is_timeout():
            break
        
        if methods[chosen_method]():
            i +=1 
            counters[chosen_method] += 1
            continue
        else: 
            break
    #indivisual.fitness = indivisual.evaluate_fitness()
    for method_name in methods.keys():
        indivisual.improved_LS_map[method_name] += counters[method_name]
    total_ls_time = time.time() - begin_LS_time
    print(f"LS: {chosen_method} | Count: {i} | TotalTime:{total_ls_time:.3f}s | Cost:{total_cost(indivisual.id_to_vehicle, indivisual.route_map, indivisual.solution):.2f}", file=sys.stderr)


def new_crossover(parent1: Chromosome , parent2: Chromosome , PDG_map : Dict[str , List[Node]] , is_limited = False):
    begin_func_time = time.time()
    # Cac super node
    new_PDG_map : Dict[str , List[Node]] = {}
    for key , value in PDG_map.items():
        key = f'{len(value[0].pickup_item_list)}_{value[0].pickup_item_list[0].id}'
        new_PDG_map[key] = value
    
    # Khởi tạo lời giải con là rỗng -> điều kiện dừng của vòng lặp sẽ là kiểm tra child đã được thêm tất cả các tuyền đường từ cha và mẹ
    child_solution_1 :Dict[str, List[Node]] = {vehicleID:[] for vehicleID in parent1.id_to_vehicle.keys()}
    child_solution_2 :Dict[str, List[Node]] = {vehicleID:[] for vehicleID in parent1.id_to_vehicle.keys()}
    
    check_valid_1 : Dict[str , int]= {key : 0 for key in new_PDG_map.keys()}
    check_valid_2 : Dict[str , int]= {key : 0 for key in new_PDG_map.keys()}
    
    # thêm các tuyến tuyến đường một cách ngẫu nhiên cho 2 lời giải con
    for vehicleID in parent1.id_to_vehicle.keys():
        if random.uniform(0 , 1) < 0.5:
            for node in parent1.solution[vehicleID]:
                child_solution_1[vehicleID].append(node)
            for node in parent2.solution[vehicleID]:
                child_solution_2[vehicleID].append(node)
        else:
            for node in parent2.solution[vehicleID]:
                child_solution_1[vehicleID].append(node)
            for node in parent1.solution[vehicleID]:
                child_solution_2[vehicleID].append(node)
    
    #Kiểm tra các cặp node còn thiếu
    # Lưu các nút thừa trong tuyến đường hiện tại
    for vehicleID in parent1.id_to_vehicle.keys():
        redundant = []
        del_index = []
        # Duyệt ngược danh sách để tìm và xóa nút thừa    
        for i in range(len(child_solution_1[vehicleID]) - 1, -1, -1):  
            node = child_solution_1[vehicleID][i]
            
            if node.pickup_item_list:
                if redundant and node.pickup_item_list[0].id == redundant[-1]:
                    redundant.pop()  # Loại bỏ phần tử tương ứng trong danh sách `redundant`
                    del_index.append(i)
            else:
                key = f'{len(node.delivery_item_list)}_{node.delivery_item_list[-1].id}'
                
                if key in new_PDG_map:
                    check_valid_1[key] += 1
                    
                    # nếu tìm được một super node thừa
                    if check_valid_1[key] > 1:
                        first_itemID_of_redundant_supernode = key.split('_')[-1]
                        redundant.append(first_itemID_of_redundant_supernode)
                        #print(f"Redundant nodes: {redundant}" , file= sys.stderr)
                        # Xóa node giao của super node thừa
                        del_index.append(i)
                        #print('Đã xóa 1 super node thừa' , file= sys.stderr)
        for i in del_index:
            child_solution_1[vehicleID].pop(i)
        
    #xóa các cặp node thừa cho xe 2
    for vehicleID in parent2.id_to_vehicle.keys():
        redundant = []
        del_index = []
        # Duyệt ngược danh sách để tìm và xóa nút thừa    
        for i in range(len(child_solution_2[vehicleID]) - 1, -1, -1):  
            node = child_solution_2[vehicleID][i]
            
            if node.pickup_item_list:
                if redundant and node.pickup_item_list[0].id == redundant[-1]:
                    redundant.pop()  # Loại bỏ phần tử tương ứng trong danh sách `redundant`
                    del_index.append(i)
            else:
                key = f'{len(node.delivery_item_list)}_{node.delivery_item_list[-1].id}'
                
                if key in new_PDG_map:
                    check_valid_2[key] += 1
                    
                    # nếu tìm được một super node thừa
                    if check_valid_2[key] > 1:
                        first_itemID_of_redundant_supernode = key.split('_')[-1]
                        redundant.append(first_itemID_of_redundant_supernode)
                        #print(f"Redundant nodes: {redundant}" , file= sys.stderr)

                        # Xóa node giao của super node thừa
                        del_index.append(i)
                        #print('Đã xóa 1 super node thừa' , file= sys.stderr)
        for i in del_index:
            child_solution_2[vehicleID].pop(i)
    
    #kiểm tra xem tổng số các node có bằng với số các node yêu cầu không
    node_num = 0
    for k, v in parent1.solution.items():
        node_num += len(v)
    child1_node_num = 0
    child2_node_num = 0
    for k, v in child_solution_1.items():
        child1_node_num += len(v)
    for key, value in check_valid_1.items():
        if value == 0:
            child1_node_num += 2
    
    for k, v in child_solution_2.items():
        child2_node_num += len(v)
    for key, value in check_valid_2.items():
        if value == 0:
            child2_node_num += 2    
    
    if child1_node_num != node_num or child2_node_num != node_num:
        return None , None
    
    print('Thoi gian thuc hien crossover: ' , time.time() - begin_func_time  , file=sys.stderr)
    
    #Tối ưu các lời giải relaxation con
    sorted_child_solution_1 = sorted(child_solution_1.items() ,  key=lambda x: int(x[0].split('_')[1]))
    child_solution_1.clear()
    child_solution_1.update(sorted_child_solution_1)
    child_1 = Chromosome(child_solution_1 , parent1.route_map , parent1.id_to_vehicle)
    
    sorted_child_solution_2 = sorted(child_solution_2.items() ,  key=lambda x: int(x[0].split('_')[1]))
    child_solution_2.clear()
    child_solution_2.update(sorted_child_solution_2)
    child_2 = Chromosome(child_solution_2 , parent2.route_map , parent2.id_to_vehicle)
    
    randon_1_LS(child_1 , is_limited , 0)
    randon_1_LS(child_2 , is_limited , 0)
    
    # Kiem tra lai và thêm các node còn thiếu solution 1        
    for key, value in check_valid_1.items():
        if value == 0:
            if random.uniform(0 , 1) < 1:
                # truong hop bi thieu 1 super node thi gan theo chien luoc CI vao solution hien tai
                selected_vehicleID = random.choice(list(parent1.id_to_vehicle.keys()))
                node_list = new_PDG_map[key]
                isExhausive = False
                route_node_list : List[Node] = []
                
                if node_list:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = new_dispatch_nodePair(node_list , parent1.id_to_vehicle , child_solution_1 , parent1.route_map ,selected_vehicleID)
                    
                route_node_list = child_solution_1.get(bestInsertVehicleID , [])
                if isExhausive:
                    route_node_list = bestNodeList[:]
                else:
                    if route_node_list is None:
                        route_node_list = []
                    
                    new_order_pickup_node = node_list[0]
                    new_order_delivery_node = node_list[1]
                    
                    route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                    route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                child_solution_1[bestInsertVehicleID] = route_node_list
            else:
                node_list = new_PDG_map[key]
                random_dispatch_nodePair(node_list, parent1.id_to_vehicle, child_solution_1)
            
            
    # Kiem tra lai và thêm các node còn thiếu solution 2      
    for key, value in check_valid_2.items():
        if value == 0:
            if random.uniform(0 , 1) < 1:
                # truong hop bi thieu 1 super node thi gan theo chien luoc CI vao solution hien tai
                selected_vehicleID = random.choice(list(parent1.id_to_vehicle.keys()))
                
                node_list = new_PDG_map[key]
                isExhausive = False
                route_node_list : List[Node] = []
                
                if node_list:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = new_dispatch_nodePair(node_list , parent2.id_to_vehicle , child_solution_2 , parent2.route_map , selected_vehicleID )
                    
                route_node_list = child_solution_2.get(bestInsertVehicleID , [])

                if isExhausive:
                    route_node_list = bestNodeList[:]
                else:
                    if route_node_list is None:
                        route_node_list = []
                    
                    new_order_pickup_node = node_list[0]
                    new_order_delivery_node = node_list[1]
                    
                    route_node_list.insert(bestInsertPosI, new_order_pickup_node)
                    route_node_list.insert(bestInsertPosJ, new_order_delivery_node)
                child_solution_2[bestInsertVehicleID] = route_node_list
            else:
                node_list = new_PDG_map[key]
                random_dispatch_nodePair(node_list, parent2.id_to_vehicle, child_solution_2)
    
    print('Thoi gian thuc hien crossover: ' , time.time() - begin_func_time  , file=sys.stderr)
    print(child_1.fitness , file=sys.stderr)
    print(child_2.fitness , file=sys.stderr)
    return child_1 , child_2

def new_dispatch_nodePair(node_list: list[Node]  , id_to_vehicle: Dict[str , Vehicle] , vehicleid_to_plan: Dict[str, list[Node]], route_map: Dict[tuple , tuple]  , selected_vehicle: str= None , mode = 'total' ):
    bestInsertVehicleID: str = ''
    bestInsertPosI: int = 0
    bestInsertPosJ: int = 1
    bestNodeList : list[Node] = []
    isExhausive  = False
    new_pickup_node = node_list[0]
    new_delivery_node = node_list[1]
    minCostDelta = math.inf

    for vehicleID , vehicle in id_to_vehicle.items():
        if selected_vehicle is not None and vehicleID != selected_vehicle:
            continue
        
        vehicle_plan = vehicleid_to_plan[vehicleID]
        
        node_list_size = len(vehicle_plan) if vehicle_plan else 0

        insert_pos = 0 
        model_nodes_num = node_list_size + 2
        first_merge_node_num = 0

        if vehicle.des:
            if new_pickup_node.id != vehicle.des.id:
                insert_pos = 1
            
            if vehicle_plan is not None and vehicle_plan:
                for node in vehicle_plan:
                    if vehicle.des.id != node.id:
                        break
                    first_merge_node_num += 1

        model_nodes_num -= first_merge_node_num
        cp_route_node_list : List[Node] = [] # Một copy của một kế hoạch hiện có 
        if vehicle_plan:
            for node in vehicle_plan:
                cp_route_node_list.append(node)
        
        
        for i in range(insert_pos, node_list_size + 1):
            if vehicle_plan is not None:
                tempRouteNodeList = copy.deepcopy(vehicle_plan)
            else:
                tempRouteNodeList = []

            tempRouteNodeList.insert(i, new_pickup_node)

            for j in range(i + 1, node_list_size + 2):
                if j != i + 1 and tempRouteNodeList[j - 1].pickup_item_list:
                    for k in range(j, node_list_size + 2):
                        if tempRouteNodeList[k].delivery_item_list:
                            if tempRouteNodeList[j - 1].pickup_item_list[0].id == tempRouteNodeList[k].delivery_item_list[- 1].id:
                                j = k + 1
                                break

                elif tempRouteNodeList[j - 1].delivery_item_list :
                    is_terminal = True
                    for k in range(j - 2, -1, -1):
                        if tempRouteNodeList[k].pickup_item_list:
                            if tempRouteNodeList[j - 1].delivery_item_list[- 1].id == tempRouteNodeList[k].pickup_item_list[0].id:
                                if k < i:
                                    is_terminal = True
                                    break
                                elif k > i:
                                    is_terminal = False
                                    break
                    if is_terminal:
                        break

                tempRouteNodeList.insert(j, new_delivery_node)

                costValue = cost_of_a_route(tempRouteNodeList, vehicle, id_to_vehicle , route_map , vehicleid_to_plan, mode)
                if costValue < minCostDelta:
                    minCostDelta = costValue
                    bestInsertPosI = i
                    bestInsertPosJ = j
                    bestInsertVehicleID = vehicleID
                    isExhausive = False

                tempRouteNodeList.pop(j)
    return isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList
