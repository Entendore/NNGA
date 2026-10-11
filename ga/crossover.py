import random
from typing import List, Tuple, Union

Individual = List[Union[int, float, bool]]

def single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same length")
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same length")
    a, b = sorted(random.sample(range(1, len(parent1)), 2))
    offspring1 = parent1[:a] + parent2[a:b] + parent1[b:]
    offspring2 = parent2[:a] + parent1[a:b] + parent2[b:]
    return offspring1, offspring2

def uniform_crossover(parent1: Individual, parent2: Individual, swap_prob: float = 0.5) -> Tuple[Individual, Individual]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same length")
    offspring1, offspring2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < swap_prob:
            offspring1.append(g2)
            offspring2.append(g1)
        else:
            offspring1.append(g1)
            offspring2.append(g2)
    return offspring1, offspring2

def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    # Order crossover for permutations
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same length")

    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    
    def ox(p1, p2):
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        fill_pos = (b + 1) % size
        p2_pos = (b + 1) % size

        while None in child:
            if p2[p2_pos] not in child:
                child[fill_pos] = p2[p2_pos]
                fill_pos = (fill_pos + 1) % size
            p2_pos = (p2_pos + 1) % size
        return child

    return ox(parent1, parent2), ox(parent2, parent1)
