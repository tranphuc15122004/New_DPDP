from typing import Dict , List , Tuple 
from algorithm.Object import *
import algorithm.algorithm_config as config 
import random
import time
from algorithm.engine import *
from algorithm.Test_algorithm.new_LS import *
from collections import deque
import hashlib
from algorithm.Test_algorithm.new_engine import *
from algorithm.Test_algorithm.new_LS import *


def GAVND_4(initial_vehicleid_to_plan: Dict[str, List[Node]], route_map: Dict[Tuple, Tuple], 
            id_to_vehicle: Dict[str, Vehicle], Unongoing_super_nodes: Dict[int, Dict[str, Node]], 
            Base_vehicleid_to_plan: Dict[str, List[Node]]) -> Chromosome:
    
    population, PDG_map = new_generate_random_chromosome(initial_vehicleid_to_plan, route_map, id_to_vehicle, Unongoing_super_nodes, Base_vehicleid_to_plan, config.POPULATION_SIZE)

    if population is None:
        print('Cant initialize the population')
        return None
    best_solution: Chromosome = None
    stagnant_generations = 0
    population.sort(key=lambda x: x.fitness)
    best_solution = population[0]
    # Elite size
    
    # Global Tabu list across generations (store SHA1 hex of route signatures)
    global_tabu = deque()
    cant_improved_list = []

    for gen in range(config.NUMBER_OF_GENERATION):
        # Kiểm tra timeout
        begin_gen_time = time.time()
        if config.is_timeout():
            elapsed_time = time.time() - config.BEGIN_TIME
            print(f"TimeOut!! Elapsed: {elapsed_time:.1f}s")
            break
        
        elite_count = max(1, int(config.POPULATION_SIZE * 0.3))
        if elite_count > len(population):
            elite_count = len(population)
        new_population = population[:elite_count]
        
        # Tạo con (có giới hạn số lần thử để tránh vòng lặp vô hạn ở test nhỏ)
        target_size =  config.POPULATION_SIZE
        attempt = 0
        max_attempt = config.POPULATION_SIZE
        while len(new_population) < target_size and attempt < max_attempt:
            attempt += 1
            parent1, parent2 = select_parents(population)
            if not parent1 or not parent2:
                break
            child = new_crossver2(parent1 , parent2 , Base_vehicleid_to_plan , PDG_map)
            if child is None:
                continue
            if config.USE_TABU:
                sig_str = get_route_after(child.solution, {})
                h_sig = hashlib.sha1(sig_str.encode('utf-8')).hexdigest()
                if h_sig in global_tabu and not (config.TABU_ACCEPT_BETTER and child.fitness < best_solution.fitness):
                    continue
                global_tabu.append(h_sig)
                if len(global_tabu) > config.TABU_LIST_SIZE:
                    global_tabu.popleft()
            new_population.append(child)

        population = new_population[:config.POPULATION_SIZE]

        # Fallback: nếu không đủ số lượng (do tabu chặn hết), nhân bản elite
        if len(population) < target_size:
            population.sort(key=lambda x: x.fitness)
            needed = target_size - len(population)
            base = population[:max(1, min(len(population), 5))]
            for k in range(needed):
                population.append(copy.deepcopy(base[k % len(base)]))
        
        if config.is_timeout():
            break
        
        population.sort(key=lambda x: x.fitness)
        population = population[:config.POPULATION_SIZE]
        if population[0].fitness < best_solution.fitness: config.IMPROVED_IN_CROSS += 1
        
        population.sort(key=lambda x: x.fitness)
        for c in range( int(len(population) * config.MUTATION_RATE) ):
            sig_str = get_route_after(population[c].solution , {})
            h_sig = hashlib.sha1(sig_str.encode('utf-8')).hexdigest()
            if population[c].cant_improved == False and h_sig not in cant_improved_list:
                adaptive_LS_stategy(population[c] , True, mode=1, tabu_list=global_tabu, global_best=best_solution , global_population = population)
                #randon_1_LS(population[c] , False)
            if population[c].cant_improved:
                cant_improved_list.append(h_sig)
                population.pop(c)
                if best_solution is None or population[c].fitness < best_solution.fitness:
                    best_solution = copy.deepcopy(population[c])
        
        population.sort(key=lambda x: x.fitness)

        if population[0].fitness < best_solution.fitness: config.IMPROVED_IN_MUTATION += 1
        
        
        # Cập nhật best solution
        if best_solution is None or population[0].fitness < best_solution.fitness:
            best_solution = copy.deepcopy(population[0])
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        
        avg = sum([c.fitness for c in population]) / len(population)
        
        print(f'Generation {gen+1}: Best = {best_solution.fitness:.2f}, '
            f'Worst = {population[-1].fitness:.2f}, '
            f'Avg = {avg:.2f}, '
            f'Time: {time.time() - begin_gen_time}')

        # Điều kiện dừng
        if stagnant_generations >= 5 or avg  == population[0].fitness:
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
    print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.1f} minutes)" )
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
    if random.random() < 1:
        return tournament_selection(), tournament_selection()
    else:
        return roulette_wheel_selection(), roulette_wheel_selection()


def adaptive_LS_stategy(indivisual: Chromosome, is_limited=True , mode = 1, tabu_list: deque | None = None, global_best: Chromosome | None = None , global_population : List[Chromosome] | None = None):
    if config.is_timeout():
        return False
    
    i = 1
    
    # Dictionary các phương pháp Local Search
    methods = {
        'PDPairExchange': lambda: new_inter_couple_exchange(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'BlockExchange': lambda: new_block_exchange(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'BlockRelocate': lambda: new_block_relocate(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited),
        'mPDG': lambda: new_multi_pd_group_relocate(indivisual.solution, indivisual.id_to_vehicle, indivisual.route_map, is_limited)
    }
    
    # Counter cho từng phương pháp
    counters = {name: 0 for name in methods.keys()}
    
    # Lấy thứ tự adaptive
    method_names = get_adaptive_order(indivisual , methods , mode=mode)
    
    #  Track local search timings per method
    ls_timings = {
        'PDPairExchange': 0.0,
        'BlockExchange': 0.0,
        'BlockRelocate': 0.0,
        'mPDG': 0.0,
    }
    
    # Pre-compute current route signature for tabu logic
    from algorithm.engine import get_route_after
    current_signature_str = get_route_after(indivisual.solution, {})
    h_current_signature = hashlib.sha1(current_signature_str.encode('utf-8')).hexdigest()
    if config.USE_TABU_IN_LS and tabu_list is not None and h_current_signature not in tabu_list:
        tabu_list.append(h_current_signature)
        if len(tabu_list) > config.TABU_LIST_SIZE:
            tabu_list.popleft()

    while i < config.LS_MAX:
        if config.is_timeout():
            break
        
        ls_start = time.time()
        if methods[method_names[0]]():
            """ tmp_chromosome = copy.deepcopy(indivisual)
            tmp_chromosome.improved_LS_map =  {method: 0 for method in config.LS_METHODS}
            global_population.append(tmp_chromosome) """
            
            if config.USE_TABU_IN_LS and tabu_list is not None:
                signature_str = get_route_after(indivisual.solution, {})
                h_signature = hashlib.sha1(signature_str.encode('utf-8')).hexdigest()
                # Tabu check: skip if already seen and not improving global best
                if h_signature in tabu_list and not (config.TABU_ACCEPT_BETTER and global_best and indivisual.fitness < global_best.fitness):
                    # revert counting this improvement (treat as non-improving move)
                    ls_timings[method_names[0]] += time.time() - ls_start
                    # undo effect by not incrementing counters/i and continue to next method
                else:
                    tabu_list.append(h_signature)
                    if len(tabu_list) > config.TABU_LIST_SIZE:
                        tabu_list.popleft()
                    ls_timings[method_names[0]] += time.time() - ls_start
                    i += 1
                    counters[method_names[0]] += 1
                    continue
            else:
                ls_timings[method_names[0]] += time.time() - ls_start
                i += 1
                counters[method_names[0]] += 1
                continue
        
        if config.is_timeout():
            break
        
        ls_start = time.time()
        if methods[method_names[1]]():
            """ tmp_chromosome = copy.deepcopy(indivisual)
            tmp_chromosome.improved_LS_map =  {method: 0 for method in config.LS_METHODS}
            global_population.append(tmp_chromosome) """
            
            if config.USE_TABU_IN_LS and tabu_list is not None:
                signature_str = get_route_after(indivisual.solution, {})
                h_signature = hashlib.sha1(signature_str.encode('utf-8')).hexdigest()
                if h_signature in tabu_list and not (config.TABU_ACCEPT_BETTER and global_best and indivisual.fitness < global_best.fitness):
                    ls_timings[method_names[1]] += time.time() - ls_start
                else:
                    tabu_list.append(h_signature)
                    if len(tabu_list) > config.TABU_LIST_SIZE:
                        tabu_list.popleft()
                    ls_timings[method_names[1]] += time.time() - ls_start
                    i += 1
                    counters[method_names[1]] += 1
                    continue
            else:
                ls_timings[method_names[1]] += time.time() - ls_start
                i += 1
                counters[method_names[1]] += 1
                continue

        ls_timings[method_names[1]] += time.time() - ls_start
        
        if config.is_timeout():
            break
        
        ls_start = time.time()
        if methods[method_names[2]]():
            if config.USE_TABU_IN_LS and tabu_list is not None:
                """ tmp_chromosome = copy.deepcopy(indivisual)
                tmp_chromosome.improved_LS_map =  {method: 0 for method in config.LS_METHODS}
                global_population.append(tmp_chromosome) """
                
                signature_str = get_route_after(indivisual.solution, {})
                h_signature = hashlib.sha1(signature_str.encode('utf-8')).hexdigest()
                if h_signature in tabu_list and not (config.TABU_ACCEPT_BETTER and global_best and indivisual.fitness < global_best.fitness):
                    ls_timings[method_names[2]] += time.time() - ls_start
                else:
                    tabu_list.append(h_signature)
                    if len(tabu_list) > config.TABU_LIST_SIZE:
                        tabu_list.popleft()
                    ls_timings[method_names[2]] += time.time() - ls_start
                    i += 1
                    counters[method_names[2]] += 1
                    continue
            else:
                ls_timings[method_names[2]] += time.time() - ls_start
                i += 1
                counters[method_names[2]] += 1
                continue

        ls_timings[method_names[2]] += time.time() - ls_start
        
        if config.is_timeout():
            break
        
        ls_start = time.time()
        if methods[method_names[3]]():
            """ tmp_chromosome = copy.deepcopy(indivisual)
            tmp_chromosome.improved_LS_map =  {method: 0 for method in config.LS_METHODS}
            global_population.append(tmp_chromosome) """
            
            if config.USE_TABU_IN_LS and tabu_list is not None:
                signature_str = get_route_after(indivisual.solution, {})
                h_signature = hashlib.sha1(signature_str.encode('utf-8')).hexdigest()
                if h_signature in tabu_list and not (config.TABU_ACCEPT_BETTER and global_best and indivisual.fitness < global_best.fitness):
                    ls_timings[method_names[3]] += time.time() - ls_start
                else:
                    tabu_list.append(h_signature)
                    if len(tabu_list) > config.TABU_LIST_SIZE:
                        tabu_list.popleft()
                    ls_timings[method_names[3]] += time.time() - ls_start
                    i += 1
                    counters[method_names[3]] += 1
                    continue
            else:
                ls_timings[method_names[3]] += time.time() - ls_start
                i += 1
                counters[method_names[3]] += 1
                continue

        ls_timings[method_names[3]] += time.time() - ls_start
        
        if config.is_timeout():
            break
        
        indivisual.cant_improved = True
        break

    for method_name in methods.keys():
        indivisual.improved_LS_map[method_name] += counters[method_name]
    
    #  Enhanced logging with detailed timing information
    total_ls_time = sum(ls_timings.values())
    timing_details = " | ".join([f"{name}:{counters[name]}({ls_timings[name]:.3f}s)" for name in method_names])
    print(f"LS: {timing_details} | TotalTime:{total_ls_time:.3f}s | Cost:{total_cost(indivisual.id_to_vehicle, indivisual.route_map, indivisual.solution):.2f}", file=sys.stderr)
