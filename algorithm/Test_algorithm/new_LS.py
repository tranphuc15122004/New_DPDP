import sys
import math
import random
from typing import Dict , List
import copy
import time
from algorithm.Object import Node, Vehicle, OrderItem , Chromosome
from algorithm.algorithm_config import *
from algorithm.engine import *
from algorithm.local_search import * 

def Improve_by_relocate_delay_order(vehicleid_to_plan: Dict[str , List[Node]], id_to_vehicle: Dict[str , Vehicle] , route_map: Dict[tuple , tuple] , is_limited : bool = False ):
    is_improved = False
    cp_vehicle_id2_planned_route : Dict [str , List [Node]]= {}
    for key , value in vehicleid_to_plan.items():
        cp_vehicle_id2_planned_route[key] = []
        for node in value:
            cp_vehicle_id2_planned_route[key].append(node)
    #cp_vehicle_id2_planned_route = copy.deepcopy(vehicleid_to_plan)
    dis_order_super_node,  _ = get_UnongoingSuperNode(vehicleid_to_plan , id_to_vehicle)
    ls_node_pair_num = len(dis_order_super_node)
    if ls_node_pair_num == 0:
        return False
    formal_super_node : Dict[int, Dict[str , List[Node]]]= {}
    new_formal_super_node : Dict[int, Dict[str , List[Node]]]= {}
    new_cost_delta = [math.inf] * ls_node_pair_num
    
    for idx, pdg in dis_order_super_node.items():
        if not pdg or len(pdg) == 0:
            continue
        
        pickup_node : Node = None ; delivery_node : Node = None
        node_list = []
        index = 0
        d_num = len(pdg) // 2
        vehicle_id = None
        pos_i, pos_j = 0, 0
        
        if pdg: 
            # Them toan cac super node vao nodelist
            for v_and_pos_str , node in pdg.items():
                if index % 2 == 0:
                    vehicle_id, pos_i = v_and_pos_str.split(",")
                    pos_i = int(pos_i)
                    pickup_node =  node
                    node_list.insert(0, pickup_node)
                else:
                    pos_j = int(v_and_pos_str.split(",")[1])
                    delivery_node = node
                    node_list.append(delivery_node)
                    pos_j = pos_j - d_num + 1
                index += 1
            
            k = f"{vehicle_id},{pos_i}+{pos_j}"
            pdg_hash_map : Dict[str , List[Node]] = {k: node_list}
            formal_super_node[idx] = pdg_hash_map
            new_formal_super_node[idx] = pdg_hash_map
        
        
        route_node_list : List[Node] = cp_vehicle_id2_planned_route.get(vehicle_id , [])
        vehicle = id_to_vehicle.get(vehicle_id)
        
        cost_after_insertion = single_vehicle_cost(route_node_list , vehicle , route_map )
        
        del route_node_list[pos_i : pos_i + d_num]
        del route_node_list[pos_j - d_num : pos_j]
        cp_vehicle_id2_planned_route[vehicle_id] = route_node_list
        
        cost_before_insertion = single_vehicle_cost(route_node_list , vehicle , route_map )
        curr_cost_detal = cost_after_insertion - cost_before_insertion
        
        min_cost_delta, best_insert_pos_i, best_insert_pos_j, best_insert_vehicle_id = dispatch_order_to_best(node_list , cp_vehicle_id2_planned_route , id_to_vehicle , route_map)
        
        if min_cost_delta < curr_cost_detal:
            new_cost_delta[idx] = min_cost_delta
            pdg_hash_map : Dict[str , List[Node]] = {}
            k = f"{best_insert_vehicle_id},{best_insert_pos_i}+{best_insert_pos_j}"
            pdg_hash_map[k] = node_list
            new_formal_super_node[idx] = pdg_hash_map
        
        route_node_list[pos_i:pos_i] = node_list[0 : len(node_list) // 2]
        route_node_list[pos_j:pos_j] = node_list[len(node_list) // 2 : len(node_list)]

    cost_delta_temp : List[float] = new_cost_delta[:]
    sort_index = sorted(range(ls_node_pair_num), key=lambda k: cost_delta_temp[k])
    mask = [False] * len(id_to_vehicle)
    orgin_cost = -1.0
    final_cost = -1.0
    is_improved = False
    for i in range(ls_node_pair_num):
        if new_cost_delta[i] != math.inf:
            before_super_node_map = formal_super_node[sort_index[i]]
            new_super_node_map = new_formal_super_node[sort_index[i]]

            before_key = next(iter(before_super_node_map))
            before_vid, before_pos = before_key.split(',')
            before_post_i, before_post_j = map(int, before_pos.split('+'))
            before_dpg = before_super_node_map[before_key]
            d_num = len(before_dpg) // 2
            before_vehicle_idx = int(before_vid.split('_')[1]) - 1

            new_key = next(iter(new_super_node_map))
            new_vid, new_pos = new_key.split(',')
            new_post_i, new_post_j = map(int, new_pos.split('+'))
            new_dpg = new_super_node_map[new_key]
            new_vehicle_idx = int(new_vid.split('_')[1]) - 1

            if not mask[before_vehicle_idx] and not mask[new_vehicle_idx]:
                before_route_node_list = copy.deepcopy(vehicleid_to_plan.get(before_vid, []))
                before_vehicle = id_to_vehicle[before_vid]
                cost0 = cost_of_a_route(before_route_node_list, before_vehicle , id_to_vehicle , route_map , vehicleid_to_plan )
                if orgin_cost < 0:
                    orgin_cost = cost0

                del before_route_node_list[before_post_i:before_post_i + d_num]
                del before_route_node_list[before_post_j - d_num:before_post_j]
                vehicleid_to_plan[before_vid] = before_route_node_list

                new_route_node_list = copy.deepcopy(vehicleid_to_plan.get(new_vid, []))
                new_vehicle = id_to_vehicle[new_vid]

                new_route_node_list[new_post_i:new_post_i] = new_dpg[:d_num]
                new_route_node_list[new_post_j:new_post_j] = new_dpg[d_num:]
                cost1 = cost_of_a_route(new_route_node_list, new_vehicle , id_to_vehicle , route_map , vehicleid_to_plan)

                if cost0 <= cost1:
                    before_route_node_list[before_post_i:before_post_i] = before_dpg[:d_num]
                    before_route_node_list[before_post_j:before_post_j] = before_dpg[d_num:]
                    vehicleid_to_plan[before_vid] = before_route_node_list
                else:
                    final_cost = cost1
                    mask[before_vehicle_idx] = True
                    mask[new_vehicle_idx] = True
                    is_improved = True
                    vehicleid_to_plan[new_vid] = new_route_node_list
                    
                    if is_limited:
                        break
    
    return is_improved

def disturbance_opt(vehicleid_to_plan: Dict[str , List[Node]], id_to_vehicle: Dict[str , Vehicle] , route_map: Dict[tuple , tuple]):

    new_vehicle_to_plan : Dict[str , List[Node]] = {}
    for VID , plan in vehicleid_to_plan.items():
        new_vehicle_to_plan[VID] = []
        for node in plan:
            new_vehicle_to_plan[VID].append(copy.deepcopy(node))
            
    dis_order_super_node,  _ = get_UnongoingSuperNode(vehicleid_to_plan , id_to_vehicle)
    ls_node_pair_num = len(dis_order_super_node)
    if ls_node_pair_num == 0:
        return False
    
    pdg_Map : Dict[str , List[Node]] = {}
    
    for idx, pdg in dis_order_super_node.items():
        pickup_node = None
        delivery_node = None
        node_list: List[Node] = []
        pos_i = 0
        pos_j = 0
        d_num = len(pdg) // 2
        index = 0

        if pdg:
            for v_and_pos_str, node in pdg.items():
                if index % 2 == 0:
                    vehicleID = v_and_pos_str.split(",")[0]
                    pos_i = int(v_and_pos_str.split(",")[1])
                    pickup_node = node
                    node_list.insert(0, pickup_node)
                    index += 1
                else:
                    pos_j = int(v_and_pos_str.split(",")[1])
                    delivery_node = node
                    node_list.append(delivery_node)
                    index += 1
                    pos_j = pos_j - d_num + 1

            k : str = f"{vehicleID},{int(pos_i)}+{int(pos_j)}"
            pdg_Map[k] = node_list
            
    if len(pdg_Map) < 2:
        return False
    
    # khoảng 30% cơ hội một cặp node sẽ được gán lại
    # Đánh dấu các cặp node sẽ được gán
    num_pairs_to_relocate = max(1, int(len(pdg_Map) * 0.3))
    pairs_to_relocate = random.sample(list(pdg_Map.keys()), num_pairs_to_relocate)
    
    # Lưu trữ các cặp node sẽ được gán lại
    relocated_pairs : Dict[str, List[Node]] = {}
    
    # Nhóm các cặp node cần xóa theo xe
    vehicle_removal_info : Dict[str, List[tuple]] = {}
    
    for key in pairs_to_relocate:
        relocated_pairs[key] = pdg_Map[key]
        
        # Lấy thông tin vị trí và xe
        vehicle_pos_info = key.split(',')
        vehicle_id = vehicle_pos_info[0]
        positions = vehicle_pos_info[1].split('+')
        pos_i = int(positions[0])
        pos_j = int(positions[1])
        
        node_list = pdg_Map[key]
        d_num = len(node_list) // 2
        
        # Thêm thông tin xóa vào dictionary theo xe
        if vehicle_id not in vehicle_removal_info:
            vehicle_removal_info[vehicle_id] = []
        vehicle_removal_info[vehicle_id].append((pos_i, pos_i + d_num, pos_j, pos_j + d_num))
    
    # Xây dựng lời giải sau khi bỏ những cặp sẽ được gán lại
    for vehicle_id, removal_list in vehicle_removal_info.items():
        route_node_list = new_vehicle_to_plan.get(vehicle_id, [])
        
        # Thu thập tất cả các chỉ số cần xóa
        indices_to_remove = set()
        for pos_i_start, pos_i_end, pos_j_start, pos_j_end in removal_list:
            # Thêm chỉ số pickup nodes
            for idx in range(pos_i_start, pos_i_end):
                if idx < len(route_node_list):
                    indices_to_remove.add(idx)
            # Thêm chỉ số delivery nodes
            for idx in range(pos_j_start, pos_j_end):
                if idx < len(route_node_list):
                    indices_to_remove.add(idx)
        
        # Sắp xếp chỉ số theo thứ tự giảm dần để xóa từ cuối lên đầu
        sorted_indices = sorted(indices_to_remove, reverse=True)
        
        # Xóa các node theo thứ tự từ cuối lên đầu
        for idx in sorted_indices:
            if idx < len(route_node_list):
                del route_node_list[idx]
        
        new_vehicle_to_plan[vehicle_id] = route_node_list
    
    # Gán lại các cặp node đã đánh dấu một cách ngẫu nhiên vào tuyến đường
    for key, node_list in relocated_pairs.items():
        # Sử dụng random_dispatch_nodePair để gán ngẫu nhiên cặp node
        random_dispatch_nodePair(node_list, id_to_vehicle, new_vehicle_to_plan)

    return Chromosome(new_vehicle_to_plan , route_map , id_to_vehicle)