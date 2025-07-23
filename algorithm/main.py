import sys
import time
from typing import Dict , List
from algorithm.In_and_Out import *
from algorithm.Object import *
from algorithm.engine import *
from algorithm.Main_algorithm.GA import GA ,  GA1LS , GA5LS
from algorithm.Main_algorithm.ACO import ACO
from algorithm.Test_algorithm.new_engine import *
from algorithm.Test_algorithm.new_GA import new_GA
from algorithm.Test_algorithm.new_GA2 import new_algo_2
import algorithm.algorithm_config as Config


input_directory = r'algorithm\data_interaction'

def main():
    Config.BEGIN_TIME = time.time()
    id_to_factory , route_map ,  id_to_vehicle , id_to_unlocated_items ,  id_to_ongoing_items , id_to_allorder = Input()
    deal_old_solution_file(id_to_vehicle)

    vehicleid_to_plan: Dict[str , List[Node]]= {}
    vehicleid_to_destination : Dict[str , Node] = {}

    new_order_itemIDs : List[str] = []
    new_order_itemIDs = restore_scene_with_single_node(vehicleid_to_plan , id_to_ongoing_items, id_to_unlocated_items  , id_to_vehicle , id_to_factory ,id_to_allorder)

    new_order_itemIDs = [item for item in new_order_itemIDs if item]
    
    #Thuat toan
    print()
    
    if over24hours(id_to_vehicle , new_order_itemIDs):
        redispatch_process(id_to_vehicle , route_map , vehicleid_to_plan , id_to_factory , id_to_unlocated_items)
    else:
        new_dispatch_new_orders(vehicleid_to_plan , id_to_factory , route_map , id_to_vehicle , id_to_unlocated_items , new_order_itemIDs)
    
    Unongoing_super_nodes , Base_vehicleid_to_plan= get_UnongoingSuperNode(vehicleid_to_plan , id_to_vehicle)
    
    best_chromosome : Chromosome = new_algo_2(vehicleid_to_plan , route_map , id_to_vehicle , Unongoing_super_nodes , Base_vehicleid_to_plan)
    if best_chromosome is None or best_chromosome.fitness > total_cost(id_to_vehicle , route_map , vehicleid_to_plan):
        best_chromosome = Chromosome(vehicleid_to_plan , route_map , id_to_vehicle)

    print()
    print('The solution initialized by Cheapest Insertion:')
    print(get_route_after(vehicleid_to_plan , {}))
    print(f'The fitness value before EA: {total_cost(id_to_vehicle , route_map , vehicleid_to_plan)}')
    print(get_route_after(best_chromosome.solution , {}))
    print(f'The fitness value after EA: {best_chromosome.fitness}')
    
    #Ket thuc thuat toan
    
    used_time = time.time() - Config.BEGIN_TIME
    update_solution_json(id_to_ongoing_items , id_to_unlocated_items , id_to_vehicle , best_chromosome.solution , vehicleid_to_destination , route_map , used_time)
    
    merge_node(id_to_vehicle , best_chromosome.solution)
    
    get_output_solution(id_to_vehicle , best_chromosome.solution , vehicleid_to_destination)
    
    print( f"Destination: {vehicleid_to_destination}", file = sys.stderr)
    write_destination_json_to_file(vehicleid_to_destination   , input_directory)
    
    
    print(f"Route: {best_chromosome.solution}", file = sys.stderr)
    write_route_json_to_file(best_chromosome.solution  , input_directory) 
    

if __name__ == '__main__':
    main()