import random
from typing import List, Callable, Tuple


def tournament_selection(
    population: Population,
    fitness_fn: Callable[[Individual], float],
    k: int = 3,
    maximize: bool = True
) -> Individual:
    selected = random.sample(population, k)
    scored = [(ind, fitness_fn(ind)) for ind in selected]
    scored.sort(key=lambda x: x[1], reverse=maximize)
    return scored[0][0]


def roulette_wheel_selection(
    population: Population,
    fitness_fn: Callable[[Individual], float],
    maximize: bool = True
) -> Individual:
    fitness_values = [fitness_fn(ind) for ind in population]
    if not maximize:
        fitness_values = [1 / (f + 1e-6) for f in fitness_values]

    total = sum(fitness_values)
    probabilities = [f / total for f in fitness_values]
    return random.choices(population, weights=probabilities, k=1)[0]


def rank_selection(
    population: Population,
    fitness_fn: Callable[[Individual], float],
    maximize: bool = True
) -> Individual:
    scored = sorted(population, key=fitness_fn, reverse=maximize)
    n = len(scored)
    ranks = [n - i for i in range(n)]  # High rank = better
    return random.choices(scored, weights=ranks, k=1)[0]


def stochastic_universal_sampling(
    population: Population,
    fitness_fn: Callable[[Individual], float],
    num_select: int = 1,
    maximize: bool = True
) -> List[Individual]:
    fitness_values = [fitness_fn(ind) for ind in population]
    if not maximize:
        fitness_values = [1 / (f + 1e-6) for f in fitness_values]

    total_fitness = sum(fitness_values)
    distance = total_fitness / num_select
    start_point = random.uniform(0, distance)
    points = [start_point + i * distance for i in range(num_select)]

    selected = []
    fitness_accum = 0
    index = 0
    for point in points:
        while fitness_accum < point and index < len(population):
            fitness_accum += fitness_values[index]
            index += 1
        selected.append(population[index - 1])
    return selected
