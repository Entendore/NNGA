import random
from typing import List, Tuple, Union

GeneType = Union[int, float, bool]
Individual = List[GeneType]
Population = List[Individual]

def initialize_float_population(
    population_size: int,
    chromosome_length: int,
    gene_bounds: Tuple[float, float] = (0.0, 1.0)
) -> Population:
    low, high = gene_bounds
    return [
        [random.uniform(low, high) for _ in range(chromosome_length)]
        for _ in range(population_size)
    ]

def initialize_int_population(
    population_size: int,
    chromosome_length: int,
    gene_bounds: Tuple[int, int] = (0, 100)
) -> Population:
    low, high = gene_bounds
    return [
        [random.randint(low, high) for _ in range(chromosome_length)]
        for _ in range(population_size)
    ]

def initialize_binary_population(
    population_size: int,
    chromosome_length: int
) -> Population:
    return [
        [random.choice([False, True]) for _ in range(chromosome_length)]
        for _ in range(population_size)
    ]

def initialize_permutation_population(
    population_size: int,
    chromosome_length: int
) -> Population:
    base = list(range(chromosome_length))
    population = []
    for _ in range(population_size):
        individual = base[:]
        random.shuffle(individual)
        population.append(individual)
    return population

def get_population_stats(population: Population, fitness_fn) -> dict:
    fitness_values = [fitness_fn(ind) for ind in population]
    return {
        "max": max(fitness_values),
        "min": min(fitness_values),
        "average": sum(fitness_values) / len(fitness_values),
        "best_individual": population[fitness_values.index(max(fitness_values))]
    }
