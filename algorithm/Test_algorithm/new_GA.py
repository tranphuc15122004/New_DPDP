from typing import Dict , List , Tuple 
from algorithm.Object import *
import algorithm.algorithm_config as config 
import random
from algorithm.engine import *
from algorithm.Test_algorithm.new_LS import *
from algorithm.Test_algorithm.new_engine import *
from algorithm.Test_algorithm.new_LS import *


def new_GA(initial_vehicleid_to_plan: Dict[str, List[Node]], route_map: Dict[Tuple, Tuple], 
            id_to_vehicle: Dict[str, Vehicle], Unongoing_super_nodes: Dict[int, Dict[str, Node]], 
            Base_vehicleid_to_plan: Dict[str, List[Node]]) -> Chromosome:
    
    population, PDG_map = generate_random_chromosome(initial_vehicleid_to_plan, route_map, id_to_vehicle, Unongoing_super_nodes, Base_vehicleid_to_plan, config.POPULATION_SIZE)

    if population is None:
        print('Cant initialize the population')
        return None
    best_solution: Chromosome = None
    stagnant_generations = 0
    population.sort(key=lambda x: x.fitness)
    best_solution = population[0]
    
    # Elite size
    elite_size = max(2, config.POPULATION_SIZE // 5)
    
    for gen in range(config.NUMBER_OF_GENERATION):
        gen_start_time = time.time()  # Bắt đầu generation
        
        new_population = []
        
        # Elitism - giữ lại elite
        population.sort(key=lambda x: x.fitness)
        new_population = population[:elite_size]

        # Tạo con
        while len(new_population) < config.POPULATION_SIZE:
            parent1, parent2 = select_parents(population)
            child = parent1.crossover(parent2, PDG_map)
            new_population.append(child)

        population = new_population
        
        for c in population:
            new_mutation(c , True)
            #c.mutate(True, False)
        
        # Duy trì đa dạng mỗi thế hệ
        population = maintain_diversity(population, id_to_vehicle, route_map, Base_vehicleid_to_plan, PDG_map)
        
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
        if stagnant_generations >= 6:
            print("Stopping early due to lack of improvement.")
            break

        # Time check - Điều chỉnh
        gen_end_time = time.time()
        current_gen_duration = gen_end_time - gen_start_time
        
        # Tính thời gian của generation đầu tiên
        if gen == 0:
            first_gen_time = current_gen_duration
        
        # Tính tổng thời gian đã sử dụng
        elapsed_time = gen_end_time - config.BEGIN_TIME
        
        # Ước tính thời gian cần cho generation tiếp theo
        avg_gen_time = elapsed_time / (gen + 1)
        estimated_next_gen_time = avg_gen_time
        
        # Kiểm tra timeout
        if elapsed_time + estimated_next_gen_time > config.ALGO_TIME_LIMIT:
            print(f"TimeOut!! Elapsed: {elapsed_time:.1f}s, Estimated next gen: {estimated_next_gen_time:.1f}s")
            break

    final_time = time.time()
    total_runtime = final_time - config.BEGIN_TIME
    print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.1f} minutes)" )
    return best_solution

def select_parents(population: List[Chromosome]) -> Tuple[Chromosome, Chromosome]:
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

def new_mutation(indivisual: Chromosome, is_limited=False):
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
    
    # Lấy danh sách tên phương pháp và trộn ngẫu nhiên cho mỗi iteration
    method_names = list(methods.keys())
    random.shuffle(method_names)
    
    while i < config.LS_MAX:
        is_improved = False
        
        if methods[method_names[0]]():
            i += 1
            counters[method_names[0]] += 1
            is_improved = True
            continue
        
        if methods[method_names[1]]():
            i += 1
            counters[method_names[1]] += 1
            is_improved = True
            continue
        
        if methods[method_names[2]]():
            i += 1
            counters[method_names[2]] += 1
            is_improved = True
            continue
        
        if methods[method_names[3]]():
            i += 1
            counters[method_names[3]] += 1
            is_improved = True
            continue
        
        if methods[method_names[4]]():
            i += 1
            counters[method_names[4]] += 1
            is_improved = True
            continue
        
        if is_improved:
            i += 0
        else:
            break
    indivisual.fitness = indivisual.evaluate_fitness()
    print(f"PDPairExchange:{counters['PDPairExchange']}; BlockExchange:{counters['BlockExchange']}; BlockRelocate:{counters['BlockRelocate']}; mPDG:{counters['mPDG']}; 2opt:{counters['2opt']}; cost:{total_cost(indivisual.id_to_vehicle, indivisual.route_map, indivisual.solution):.2f}", file=sys.stderr)
        
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

def maintain_diversity(population: List[Chromosome], 
                    id_to_vehicle: Dict[str, Vehicle],
                    route_map: Dict[Tuple, Tuple],
                    Base_vehicleid_to_plan: Dict[str, List[Node]],
                    PDG_map: Dict[str, List[Node]],
                    min_diversity: float = 0.15 ) -> List[Chromosome]:
    """Duy trì độ đa dạng của quần thể"""
    
    diversity = calculate_diversity(population)
    fitness_diversity = calculate_fitness_diversity(population)
    
    print(f"Current diversity: {diversity:.3f}, Fitness diversity: {fitness_diversity:.3f}", file=sys.stderr)
    
    # Nếu độ đa dạng quá thấp
    if diversity < min_diversity or fitness_diversity < min_diversity:
        print("Low diversity detected, applying diversity maintenance...")
        
        # Sắp xếp theo fitness
        population.sort(key=lambda x: x.fitness)
        
        # Giữ lại elite (30% tốt nhất)
        elite_size = max(2, len(population) // 2)
        elite = population[:elite_size]
        
        # Loại bỏ các cá thể quá giống nhau
        unique_population = remove_similar_individuals(elite, threshold=0.1)
        
        # Tạo cá thể mới để bù đắp
        new_individuals = []
        needed = config.POPULATION_SIZE - len(unique_population)
        
        for _ in range(needed):
            # Tạo cá thể mới bằng nhiều phương pháp khác nhau
            if random.random() < 0.4:
                # Tạo hoàn toàn ngẫu nhiên
                new_individual = generate_single_random_chromosome(Base_vehicleid_to_plan, route_map, id_to_vehicle, PDG_map)
            else:
                # Random crossover từ elite
                if random.random() < 0.5:    
                    parent1, parent2 = random.sample(elite, 2)
                    new_individual = parent1.crossover(parent2, PDG_map)
                    intensive_mutation(new_individual, is_limited=True)
                else:
                    new_individual =  copy.deepcopy(random.choice(elite))
                    #new_individual = disturbance_opt(new_individual.solution , id_to_vehicle , route_map)
                    intensive_mutation(new_individual, is_limited=True)
            
            new_individuals.append(new_individual)
        
        return unique_population + new_individuals
    return population

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

def intensive_mutation(individual: Chromosome , is_limited = True):
    """Mutation mạnh để tăng đa dạng"""
    # Áp dụng mutation nhiều lần
    for _ in range(random.randint(2, 5)):
        new_mutation(individual, is_limited = is_limited)
    # Recalculate fitness
    individual.fitness = individual.evaluate_fitness()

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
        if random.random() < 0.5:
            # Random vehicle
            selected_vehicleID = random.choice(list(id_to_vehicle.keys()))
            temp_route[selected_vehicleID].extend(DPG)
        else:
            # Random dispatch
            random_dispatch_nodePair(DPG, id_to_vehicle, temp_route)
    
    return Chromosome(temp_route, route_map, id_to_vehicle)