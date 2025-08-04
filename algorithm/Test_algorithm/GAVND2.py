from typing import Dict , List , Tuple 
from algorithm.Object import *
import algorithm.algorithm_config as config 
import random
from algorithm.engine import *
from algorithm.Test_algorithm.new_LS import *
from algorithm.Test_algorithm.new_engine import *


def GAVND_2(initial_vehicleid_to_plan: Dict[str, List[Node]], route_map: Dict[Tuple, Tuple], 
            id_to_vehicle: Dict[str, Vehicle], Unongoing_super_nodes: Dict[int, Dict[str, Node]], 
            Base_vehicleid_to_plan: Dict[str, List[Node]]) -> Chromosome:
    
    population, PDG_map = generate_random_chromosome(initial_vehicleid_to_plan, route_map, id_to_vehicle, Unongoing_super_nodes, Base_vehicleid_to_plan, config.POPULATION_SIZE)

    if population is None:
        print('Cant initialize the population')
        return None
    
    best_solution: Chromosome = initial_vehicleid_to_plan
    stagnant_generations = 0
    population.sort(key=lambda x: x.fitness)
    best_solution = population[0]
    min_diversity = 0.25
    # Elite size
    elite_size = max(2, config.POPULATION_SIZE // 5)
    
    for gen in range(config.NUMBER_OF_GENERATION):
        # Kiểm tra timeout
        if config.is_timeout():
            print(f"TimeOut!! Elapsed: {elapsed_time:.1f}s")
            break
        
        diversity = calculate_diversity(population)
        if diversity < min_diversity :
            population = maintain_diversity(population, id_to_vehicle, route_map, Base_vehicleid_to_plan, PDG_map)
        else:
            new_population = []
            
            # Elitism - giữ lại elite
            population.sort(key=lambda x: x.fitness)
            new_population = population[:elite_size]

            # Tạo con
            while len(new_population) < config.POPULATION_SIZE:
                parent1, parent2 = select_parents(population)
                child1 , child2 = new_crossover(parent1 , parent2 , PDG_map)
                if child1:
                    new_population.append(child1)
                if child2:
                    new_population.append(child2)

            population = new_population
            for c in population:
                if random.random() < config.MUTATION_RATE:
                    new_mutation(c ,False , 1)
        
        # Sắp xếp lại quần thể
        population.sort(key=lambda x: x.fitness)

        # Cập nhật best solution
        if best_solution is None or population[0].fitness < best_solution.fitness:
            best_solution = population[0]
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        
        diversity = calculate_diversity(population)
        fitness_diversity = calculate_fitness_diversity(population)
        
        print(f'Generation {gen+1}: Best = {best_solution.fitness:.2f}, '
            f'Worst = {population[-1].fitness:.2f}, '
            f'Avg = {sum([c.fitness for c in population]) / len(population):.2f}, '
            f'Diversity = {diversity:.3f}, '
            f'FitDiv = {fitness_diversity:.3f}')

        # Điều kiện dừng
        if stagnant_generations >= 7:
            print("Stopping early due to lack of improvement.")
            break

        # Time check - Điều chỉnh
        gen_end_time = time.time()
        
        # Tính tổng thời gian đã sử dụng
        elapsed_time = gen_end_time - config.BEGIN_TIME
        
        # Ước tính thời gian cần cho generation tiếp theo
        avg_gen_time = elapsed_time / (gen + 1)
        estimated_next_gen_time = avg_gen_time
        
        # Kiểm tra timeout
        if elapsed_time  > config.ALGO_TIME_LIMIT:
            print(f"TimeOut!! Elapsed: {elapsed_time:.1f}s, Estimated next gen: {estimated_next_gen_time:.1f}s")
            break

    final_time = time.time()
    total_runtime = final_time - config.BEGIN_TIME
    print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.1f} minutes) , Estimated each gen: {estimated_next_gen_time:.1f}s" )
    return best_solution


def select_parents(population: List[Chromosome]) -> Tuple[Chromosome, Chromosome]:
    if config.is_timeout():
        return None , None
    
    def tournament_selection():
        # Tăng tournament size để tăng selective pressure
        tournament_size = max(3, len(population) // 5)  # Tăng từ //10 lên //5
        candidates = random.sample(population, tournament_size)
        return min(candidates, key=lambda x: x.fitness)
    
    def roulette_wheel_selection():
        # Thêm roulette wheel selection để tăng đa dạng
        fitness_values = [1 / (c.fitness + 1) for c in population]  # Inverse fitness
        total_fitness = sum(fitness_values)
        r = random.uniform(0, total_fitness)
        cumulative = 0
        for i, fitness in enumerate(fitness_values):
            cumulative += fitness
            if cumulative >= r:
                return population[i]
        return population[-1]
    
    # Kết hợp cả 2 phương pháp
    if random.random() < 0.75:
        return tournament_selection(), tournament_selection()
    else:
        return roulette_wheel_selection(), roulette_wheel_selection()

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
            # Thêm một chút exploration (random factor)
            exploration_factor = random.uniform(0.8, 1.2)
            method_scores[method_name] = success_rate * 1
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

def new_mutation(indivisual: Chromosome, is_limited=True , mode = 1):
    if config.is_timeout():
        return False
    
    i = 1
    
    # Dictionary các phương pháp Local Search
    methods = {
        'PDPairExchange': lambda: inter_couple_exchange(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'BlockExchange': lambda: block_exchange(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'BlockRelocate': lambda: block_relocate(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'mPDG': lambda: multi_pd_group_relocate(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        '2opt': lambda: improve_ci_path_by_2_opt(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited)
    }
    
    # Counter cho từng phương pháp
    counters = {name: 0 for name in methods.keys()}
    
    # Lấy thứ tự adaptive
    method_names = get_adaptive_order(indivisual , methods , mode=mode)
    
    while i < config.LS_MAX:
        if config.is_timeout():
            break
        
        if methods[method_names[0]]():
            i += 1
            counters[method_names[0]] += 1
            continue
        
        elif methods[method_names[1]]():
            i += 1
            counters[method_names[1]] += 1
            continue
        
        elif methods[method_names[2]]():
            i += 1
            counters[method_names[2]] += 1
            continue
        
        elif methods[method_names[3]]():
            i += 1
            counters[method_names[3]] += 1
            continue
        
        elif methods[method_names[4]]():
            i += 1
            counters[method_names[4]] += 1
            continue
        else: 
            break
    indivisual.fitness = indivisual.evaluate_fitness()
    for method_name in methods.keys():
        indivisual.improved_LS_map[method_name] += counters[method_name]
    print(f"PDPairExchange:{counters['PDPairExchange']}; BlockExchange:{counters['BlockExchange']}; BlockRelocate:{counters['BlockRelocate']}; mPDG:{counters['mPDG']}; 2opt:{counters['2opt']}; cost:{total_cost(indivisual.id_to_vehicle, indivisual.route_map, indivisual.solution):.2f}", file=sys.stderr)

def maintain_diversity(population: List[Chromosome], 
                    id_to_vehicle: Dict[str, Vehicle],
                    route_map: Dict[Tuple, Tuple],
                    Base_vehicleid_to_plan: Dict[str, List[Node]],
                    PDG_map: Dict[str, List[Node]],) -> List[Chromosome]:
    # Kiểm tra timeout ngay từ đầu
    if config.is_timeout():
        print("Timeout detected in maintain_diversity", file=sys.stderr)
        return population  # ✅ Return population hiện tại

    print("Low diversity detected, applying diversity maintenance...")
    
    # Sắp xếp theo fitness
    population.sort(key=lambda x: x.fitness)
    
    # Giữ lại elite (30% tốt nhất)
    elite_size = max(2, len(population) // 4)
    elite = population[:elite_size]
    
    # Loại bỏ các cá thể quá giống nhau
    unique_population = remove_similar_individuals(elite, threshold=0.1)
    for c in unique_population:
        new_mutation(c , False , 1)
    
    # Tạo cá thể mới để bù đắp
    new_individuals = []
    needed = config.POPULATION_SIZE - len(unique_population)
    
    for i in range(needed):
        if config.is_timeout():
            print(f"Timeout during creating new individuals, created {i}/{needed}", file=sys.stderr)
            break
        
        # Tạo cá thể mới bằng nhiều phương pháp khác nhau
        if random.random() < 0.5:
            # Tạo tương đối tốt
            new_individual = generate_single_random_chromosome(Base_vehicleid_to_plan, route_map, id_to_vehicle, PDG_map)
        else:
            # Random crossover từ elite
            if random.random() < 0.5 and len(unique_population) > 2:
                parent1, parent2 = random.sample(elite, 2)
                new_individual = parent1.crossover(parent2, PDG_map)
            else:
                new_individual =  copy.deepcopy(random.choice(unique_population))
            
            new_mutation(new_individual, is_limited=False , mode= 2)
        
        new_individuals.append(new_individual)
    
    return unique_population + new_individuals


def new_crossover(parent1: Chromosome , parent2: Chromosome , PDG_map : Dict[str , List[Node]]):
    
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
                        print(f"Redundant nodes: {redundant}" , file= sys.stderr)

                        # Xóa node giao của super node thừa
                        del_index.append(i)
                        print('Đã xóa 1 super node thừa' , file= sys.stderr)
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
                        print(f"Redundant nodes: {redundant}" , file= sys.stderr)

                        # Xóa node giao của super node thừa
                        del_index.append(i)
                        print('Đã xóa 1 super node thừa' , file= sys.stderr)
        for i in del_index:
            child_solution_2[vehicleID].pop(i)
    
    #kiểm tra xem tổng số các node có bằng với số các node yêu cầu không
    """ node_num = 0
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
        return None , None """
    
    #Tối ưu các lời giải relaxation con
    
    
    """ new_mutation(child_1 , False , 1)
    new_mutation(child_2 , False , 1) """
    
    # Kiem tra lai và thêm các node còn thiếu solution 1        
    for key, value in check_valid_1.items():
        if value == 0:
            if random.uniform(0 , 1) < 0.5:
                # truong hop bi thieu 1 super node thi gan theo chien luoc CI vao solution hien tai
                node_list = new_PDG_map[key]
                isExhausive = False
                route_node_list : List[Node] = []
                
                if node_list:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(node_list , parent1.id_to_vehicle , child_solution_1 , parent1.route_map)
                    
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
                if random.uniform(0 , 1) < 0.25:
                    # Random vehicle
                    selected_vehicleID = random.choice(list(parent1.id_to_vehicle.keys()))
                    child_solution_1[selected_vehicleID].extend(node_list)
                else:
                    # Random dispatch
                    random_dispatch_nodePair(node_list, parent1.id_to_vehicle, child_solution_1)
            
            print('Cập nhật super node còn thiếu' , file= sys.stderr)
            
    # Kiem tra lai và thêm các node còn thiếu solution 2      
    for key, value in check_valid_2.items():
        if value == 0:
            if random.uniform(0 , 1) < 0.5:
                # truong hop bi thieu 1 super node thi gan theo chien luoc CI vao solution hien tai
                node_list = new_PDG_map[key]
                isExhausive = False
                route_node_list : List[Node] = []
                
                if node_list:
                    isExhausive , bestInsertVehicleID, bestInsertPosI, bestInsertPosJ , bestNodeList = dispatch_nodePair(node_list , parent2.id_to_vehicle , child_solution_2 , parent2.route_map)
                    
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
                if random.uniform(0 , 1) < 0.25:
                    # Random vehicle
                    selected_vehicleID = random.choice(list(parent2.id_to_vehicle.keys()))
                    child_solution_2[selected_vehicleID].extend(node_list)
                else:
                    # Random dispatch
                    random_dispatch_nodePair(node_list, parent2.id_to_vehicle, child_solution_2)
            
            print('Cập nhật super node còn thiếu' , file= sys.stderr)
            
    sorted_child_solution_1 = sorted(child_solution_1.items() ,  key=lambda x: int(x[0].split('_')[1]))
    child_solution_1.clear()
    child_solution_1.update(sorted_child_solution_1)
    child_1 = Chromosome(child_solution_1 , parent1.route_map , parent1.id_to_vehicle)
    
    sorted_child_solution_2 = sorted(child_solution_2.items() ,  key=lambda x: int(x[0].split('_')[1]))
    child_solution_2.clear()
    child_solution_2.update(sorted_child_solution_2)
    child_2 = Chromosome(child_solution_2 , parent2.route_map , parent2.id_to_vehicle)
    
    child_1.fitness = child_1.evaluate_fitness()
    child_2.fitness = child_2.evaluate_fitness()
    return child_1 , child_2