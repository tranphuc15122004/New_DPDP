from typing import Dict , List , Tuple 
from algorithm.Object import *
import algorithm.algorithm_config as config 
import random
import time
import copy
import math
from algorithm.engine import *
from algorithm.Test_algorithm.new_LS import *
from algorithm.Test_algorithm.new_engine import *
from algorithm.Test_algorithm.new_GA import generate_random_chromosome , calculate_diversity , calculate_fitness_diversity, calculate_chromosome_distance, new_mutation


def new_algo_2(initial_vehicleid_to_plan: Dict[str, List[Node]], route_map: Dict[Tuple, Tuple], 
            id_to_vehicle: Dict[str, Vehicle], Unongoing_super_nodes: Dict[int, Dict[str, Node]], 
            Base_vehicleid_to_plan: Dict[str, List[Node]]) -> Chromosome:
    
    population, PDG_map = generate_random_chromosome(initial_vehicleid_to_plan, route_map, id_to_vehicle, Unongoing_super_nodes, Base_vehicleid_to_plan, config.POPULATION_SIZE)

    if population is None:
        print('Cant initialize the population')
        return None
    
    best_solution: Chromosome = None
    stagnant_generations = 0
    convergence_phase = False  # Theo dõi giai đoạn hội tụ
    
    population.sort(key=lambda x: x.fitness)
    best_solution = population[0]
    
    # Dynamic elite size based on diversity
    base_elite_size = max(2, config.POPULATION_SIZE // 8)  # Giảm elite size để tăng diversity
    
    for gen in range(config.NUMBER_OF_GENERATION):
        gen_start_time = time.time()
        
        # Tính diversity trước mọi thứ để điều chỉnh strategy
        diversity = calculate_diversity_fast(population)
        
        # Xác định giai đoạn của thuật toán - adjusted cho 20 individuals
        if diversity < 0.25 and not convergence_phase:  # Tăng từ 0.2 lên 0.25
            convergence_phase = True
            print(f"Entering convergence phase at generation {gen+1}")
        
        # Điều chỉnh parameters dựa trên diversity
        current_elite_size, selection_strategy, mutation_strategy = adjust_parameters(
            diversity, convergence_phase, base_elite_size, gen, stagnant_generations
        )
        
        # Evolution process với strategy động
        new_population = evolve_population(
            population, current_elite_size, selection_strategy, 
            mutation_strategy, PDG_map, diversity
        )
        
        population = new_population
        population.sort(key=lambda x: x.fitness)
        
        # Diversity injection nếu cần thiết - adjusted threshold
        if diversity < 0.20:  # Tăng từ 0.15 lên 0.20
            population = inject_diversity(population, id_to_vehicle, route_map, 
                                        Base_vehicleid_to_plan, PDG_map, 
                                        injection_ratio=0.25)  # Giảm từ 0.3 xuống 0.25 (inject 5 cá thể)
            print(f"Diversity injection at generation {gen+1}")
        
        # Update best solution
        population.sort(key=lambda x: x.fitness)
        
        if best_solution is None or population[0].fitness < best_solution.fitness:
            best_solution = population[0]
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        
        # Early stopping với điều kiện diversity
        if should_stop(stagnant_generations, diversity, convergence_phase):
            print(f"Stopping: stagnant={stagnant_generations}, diversity={diversity:.3f}")
            break
        
        # Time check
        gen_end_time = time.time()
        current_gen_duration = gen_end_time - gen_start_time
        elapsed_time = gen_end_time - config.BEGIN_TIME
        avg_gen_time = elapsed_time / (gen + 1)
        estimated_next_gen_time = avg_gen_time
        
        if elapsed_time + estimated_next_gen_time > config.ALGO_TIME_LIMIT:
            print(f"TimeOut!! Elapsed: {elapsed_time:.1f}s, Estimated next gen: {estimated_next_gen_time:.1f}s")
            break
        
        # Enhanced logging với VND coverage cho 20 individuals
        total_mutated = estimate_mutated_individuals(mutation_strategy, len(population) - current_elite_size)
        
        print(f'Gen {gen+1}: Best={best_solution.fitness:.2f}, '
              f'Worst={population[-1].fitness:.2f}, '
              f'Avg={sum(c.fitness for c in population)/len(population):.2f}, '
              f'Div={diversity:.3f}, VND={total_mutated}/{len(population)-current_elite_size}, '
              f'Elite={current_elite_size}, Phase={"Conv" if convergence_phase else "Expl"}, '
              f'Sel={selection_strategy[:4]}, Mut={mutation_strategy[:4]}')

    final_time = time.time()
    total_runtime = final_time - config.BEGIN_TIME
    print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.1f} minutes)")
    return best_solution


def calculate_diversity_fast(population: List[Chromosome]) -> float:
    """Tính diversity nhanh bằng sampling - optimized cho 20 individuals"""
    if len(population) < 2:
        return 1.0
    
    # Với 20 cá thể, sample toàn bộ thay vì 20
    sample_size = len(population)  # Không cần sampling với population nhỏ
    sample = population
    
    total_distance = 0
    count = 0
    
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            distance = calculate_chromosome_distance(sample[i], sample[j])
            total_distance += distance
            count += 1
    
    return total_distance / count if count > 0 else 0.0


def adjust_parameters(diversity: float, convergence_phase: bool, base_elite_size: int, 
                     generation: int, stagnant_generations: int) -> Tuple[int, str, str]:
    """Điều chỉnh parameters dựa trên trạng thái quần thể - optimized cho 20 individuals"""
    
    # Với 20 cá thể, elite size phải nhỏ để có đủ cá thể cho VND mutation
    if diversity > 0.4:  # High diversity
        elite_size = 2  # Chỉ giữ 2 elite (90% cho VND)
        selection_strategy = "tournament_strong"
        mutation_strategy = "conservative"
    elif diversity > 0.25:  # Medium diversity
        elite_size = 2  # Giữ 2 elite (90% cho VND)
        selection_strategy = "mixed"
        mutation_strategy = "adaptive"
    elif diversity > 0.15:  # Low diversity
        elite_size = 1  # Chỉ 1 elite (95% cho VND)
        selection_strategy = "diversity_focused"
        mutation_strategy = "aggressive"
    else:  # Very low diversity
        elite_size = 1  # Chỉ 1 elite (95% cho VND)
        selection_strategy = "random_heavy"
        mutation_strategy = "disruptive"
    
    # Với 20 generations, stagnation threshold phải thấp hơn
    if stagnant_generations > 3:  # Giảm từ 5 xuống 3
        elite_size = 1
        selection_strategy = "diversity_focused"
        mutation_strategy = "disruptive"
    
    return elite_size, selection_strategy, mutation_strategy


def evolve_population(population: List[Chromosome], elite_size: int, 
                     selection_strategy: str, mutation_strategy: str,
                     PDG_map: Dict[str, List[Node]], diversity: float) -> List[Chromosome]:
    """Evolution process với strategy động"""
    
    population.sort(key=lambda x: x.fitness)
    new_population = population[:elite_size]
    
    # Selection và crossover
    while len(new_population) < config.POPULATION_SIZE:
        parent1, parent2 = dynamic_select_parents(population, selection_strategy, diversity)
        
        # Crossover với probability điều chỉnh
        crossover_prob = get_crossover_probability(diversity)
        if random.random() < crossover_prob:
            child = parent1.crossover(parent2, PDG_map)
        else:
            child = copy.deepcopy(random.choice([parent1, parent2]))
        
        new_population.append(child)
    
    # Mutation với strategy động
    apply_mutation_strategy(new_population, mutation_strategy, diversity)
    
    return new_population


def dynamic_select_parents(population: List[Chromosome], strategy: str, 
                          diversity: float) -> Tuple[Chromosome, Chromosome]:
    """Selection với nhiều strategies - optimized cho 20 individuals"""
    
    if strategy == "tournament_strong":
        tournament_size = min(6, len(population) // 3)  # Max 6 instead của len//3
        return tournament_selection(population, tournament_size), tournament_selection(population, tournament_size)
    
    elif strategy == "diversity_focused":
        parent1 = tournament_selection(population, 3)
        # Với 20 cá thể, threshold distance có thể thấp hơn
        candidates = [c for c in population if calculate_chromosome_distance(parent1, c) > 0.25]  # Giảm từ 0.3
        if candidates:
            parent2 = random.choice(candidates)
        else:
            parent2 = roulette_wheel_selection(population)
        return parent1, parent2
    
    elif strategy == "random_heavy":
        # Tăng random ratio cho small population
        if random.random() < 0.8:  # Tăng từ 0.7 lên 0.8
            return random.choice(population), random.choice(population)
        else:
            return tournament_selection(population, 2), tournament_selection(population, 2)
    
    elif strategy == "mixed":
        # Kết hợp phù hợp với 20 cá thể
        methods = [
            lambda: tournament_selection(population, 3),
            lambda: roulette_wheel_selection(population),
            lambda: random.choice(population[:10])  # Top half = 10 cá thể
        ]
        return random.choice(methods)(), random.choice(methods)()
    
    else:  # Default
        return tournament_selection(population, 3), tournament_selection(population, 3)


def tournament_selection(population: List[Chromosome], tournament_size: int) -> Chromosome:
    """Tournament selection"""
    candidates = random.sample(population, min(tournament_size, len(population)))
    return min(candidates, key=lambda x: x.fitness)


def roulette_wheel_selection(population: List[Chromosome]) -> Chromosome:
    """Roulette wheel selection"""
    fitness_values = [1 / (c.fitness + 1) for c in population]
    total_fitness = sum(fitness_values)
    r = random.uniform(0, total_fitness)
    cumulative = 0
    for i, fitness in enumerate(fitness_values):
        cumulative += fitness
        if cumulative >= r:
            return population[i]
    return population[-1]


def get_crossover_probability(diversity: float) -> float:
    """Điều chỉnh crossover probability"""
    if diversity > 0.3:
        return 0.9  # High crossover when diverse
    elif diversity > 0.2:
        return 0.7
    else:
        return 0.5  # Reduce crossover when converged


def apply_mutation_strategy(population_subset: List[Chromosome], strategy: str, diversity: float):
    """Áp dụng mutation strategy phù hợp với VND - optimized cho 20 individuals"""
    
    if strategy == "conservative":
        # Với 18-19 cá thể non-elite, mutate 80% để có 14-15 cá thể VND
        mutation_rate = 0.8
        for c in population_subset:
            if random.random() < mutation_rate:
                c.mutate(False, True)  # Limited VND
    
    elif strategy == "adaptive":
        # Adaptive rate cao hơn cho small population
        base_rate = 0.85
        adaptive_rate = min(0.95, base_rate + (0.4 - diversity))  # Tăng khi diversity thấp
        for c in population_subset:
            if random.random() < adaptive_rate:
                c.mutate(True, False)
    
    elif strategy == "aggressive":
        # Hầu hết cá thể được VND
        mutation_rate = 0.9
        for c in population_subset:
            if random.random() < mutation_rate:
                c.mutate(True, False)
                # Multiple VND với probability cao
                if random.random() < 0.6:
                    c.mutate(True, False)
    
    elif strategy == "disruptive":
        # Tất cả non-elite được intensive VND
        mutation_rate = 1.0
        for c in population_subset:
            # Guaranteed VND cho mọi cá thể non-elite
            c.mutate(True, False)
            # 70% chance cho second VND pass
            if random.random() < 0.7:
                new_mutation(c, is_limited=False)
            # 40% chance cho third VND pass
            if random.random() < 0.4:
                new_mutation(c, is_limited=False)


def inject_diversity(population: List[Chromosome], id_to_vehicle: Dict[str, Vehicle],
                    route_map: Dict[Tuple, Tuple], Base_vehicleid_to_plan: Dict[str, List[Node]],
                    PDG_map: Dict[str, List[Node]], injection_ratio: float = 0.25) -> List[Chromosome]:
    """Inject diversity một cách có kiểm soát - optimized cho 20 individuals"""
    
    population.sort(key=lambda x: x.fitness)
    keep_size = int(len(population) * (1 - injection_ratio))  # Giữ 15, inject 5
    kept_population = population[:keep_size]
    
    # Tạo 5 cá thể mới đa dạng với VND
    new_individuals = []
    needed = len(population) - keep_size  # = 5
    
    for i in range(needed):
        if i == 0:
            # Random chromosome với immediate VND
            new_individual = generate_single_random_chromosome(Base_vehicleid_to_plan, route_map, id_to_vehicle, PDG_map)
            new_individual.mutate(True, False)
        elif i == 1:
            # Disruptive mutation from best
            new_individual = copy.deepcopy(kept_population[0])
            new_mutation(new_individual, is_limited=False)
            new_individual.mutate(True, False)
        elif i == 2:
            # Crossover + VND
            if len(kept_population) >= 2:
                parent1 = kept_population[0]
                parent2 = random.choice(kept_population[1:5])
                new_individual = parent1.crossover(parent2, PDG_map)
                new_individual.mutate(True, False)
            else:
                new_individual = generate_single_random_chromosome(Base_vehicleid_to_plan, route_map, id_to_vehicle, PDG_map)
        elif i == 3:
            # Elite clone với heavy VND
            new_individual = copy.deepcopy(random.choice(kept_population[:3]))
            new_individual.mutate(True, False)
            if random.random() < 0.7:
                new_mutation(new_individual, is_limited=False)
        else:  # i == 4
            # Distant pair crossover
            if len(kept_population) >= 2:
                distant_pair = find_most_distant_pair(kept_population)
                new_individual = distant_pair[0].crossover(distant_pair[1], PDG_map)
                new_individual.mutate(True, False)
            else:
                new_individual = generate_single_random_chromosome(Base_vehicleid_to_plan, route_map, id_to_vehicle, PDG_map)
        
        new_individuals.append(new_individual)
    
    return kept_population + new_individuals

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
    
    #thêm đánh giá fittness 
    res = Chromosome(temp_route, route_map, id_to_vehicle)
    res.fitness = res.evaluate_fitness()
    return res


def find_most_distant_pair(population: List[Chromosome]) -> Tuple[Chromosome, Chromosome]:
    """Tìm cặp cá thể xa nhau nhất"""
    max_distance = 0
    best_pair = (population[0], population[1] if len(population) > 1 else population[0])
    
    sample_size = min(10, len(population))
    sample = random.sample(population, sample_size)
    
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            distance = calculate_chromosome_distance(sample[i], sample[j])
            if distance > max_distance:
                max_distance = distance
                best_pair = (sample[i], sample[j])
    
    return best_pair


def should_stop(stagnant_generations: int, diversity: float, convergence_phase: bool) -> bool:
    """Điều kiện dừng thông minh - optimized cho 20 generations"""
    if convergence_phase:
        return stagnant_generations >= 10  # Giảm từ 15 xuống 6
    else:
        return stagnant_generations >= 7  # Giảm từ 8 xuống 4


def estimate_mutated_individuals(strategy: str, non_elite_count: int) -> int:
    """Ước tính số cá thể được mutate"""
    rates = {
        "conservative": 0.8,
        "adaptive": 0.85, 
        "aggressive": 0.9,
        "disruptive": 1.0
    }
    return int(non_elite_count * rates.get(strategy, 0.5))