import random
import numpy as np
from typing import List, Tuple, Union

Individual = List[Union[float, int, bool]]

def uniform_mutation(
    individual: Individual,
    gene_bounds: Tuple[float, float],
    mutation_rate: float,
    gene_type: type = float
) -> Individual:
    low, high = gene_bounds
    return [
        gene_type(random.uniform(low, high)) if random.random() < mutation_rate else gene
        for gene in individual
    ]

def gaussian_mutation(
    individual: Individual,
    mutation_rate: float,
    mean: float = 0.0,
    stddev: float = 0.1,
    gene_bounds: Tuple[float, float] = (float('-inf'), float('inf')),
) -> Individual:
    low, high = gene_bounds
    return [
        min(max(gene + random.gauss(mean, stddev), low), high) if random.random() < mutation_rate else gene
        for gene in individual
    ]

def bit_flip_mutation(
    individual: List[bool],
    mutation_rate: float
) -> List[bool]:
    return [not gene if random.random() < mutation_rate else gene for gene in individual]

def swap_mutation(
    individual: Individual,
    mutation_rate: float
) -> Individual:
    mutated = individual[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def scramble_mutation(
    individual: Individual,
    mutation_rate: float
) -> Individual:
    mutated = individual[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(mutated)), 2))
        subset = mutated[i:j+1]
        random.shuffle(subset)
        mutated[i:j+1] = subset
    return mutated
