import numpy as np
import copy as cp
import random as rand
import sys

# Population and Chromosome Constants
POPULATION_SIZE = 100
CHROMOSOME_SIZE = 30
MIN_ALLELE_VALUE = -5.12
MAX_ALLELE_VALUE = 5.12
MAX_NUM_OF_GENERATIONS = 50000
BEST_FITNESS = 0
K_SELECTED_CHROMOSOMES = 50

# None Crossover and Mutation
NONE = 0

# Crossover Flags
SINGLE_POINT_CROSSOVER = 1
TWO_POINTS_CROSSOVER = 2
SINGLE_ARITHMETIC_CROSSOVER = 3
UNIFORM_CROSSOVER = 4
RING_CROSSOVER = 5
AVERAGE_CROSSOVER = 6
FLAT_CROSSOVER = 7

# Mutation Flags
UNIFORM_MUTATION = 1
SWAP_MUTATION = 2
SCRAMBLE_MUTATION = 3
INVERSION_MUTATION = 4
GAUSSIAN_MUTATION = 5
ONE_RANDOM_SIGN_MUTATION = 6
ONE_RANDOM_MEAN_MUTATION = 7

g_crossover = NONE
g_mutation = NONE

class Chromosome:
    def __init__(self, generation):
        self.generation = generation
        self.genes = []
        for _ in range(CHROMOSOME_SIZE):
            allele = np.random.uniform(MIN_ALLELE_VALUE, MAX_ALLELE_VALUE)
            self.genes.append(allele)
        self.fitness = fitness_function(self)

    def get_genes(self):
        return self.genes

    def set_genes(self, genes):
        self.genes = []
        self.genes = cp.deepcopy(genes)
        return None

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
        return None

    def get_generation(self):
        return self.generation

class Population:
    def __init__(self, size, generation):
        self.size = size
        self.chromosomes = []
        self.total_fitness = 0
        for _ in range(self.size):
            chromosome = Chromosome(generation)
            self.chromosomes.append(chromosome)
            self.total_fitness = self.total_fitness + chromosome.get_fitness()

    def get_chromosomes(self):
        return self.chromosomes

    def set_chromosomes(self, chromosomes):
        self.chromosomes = []
        self.chromosomes = cp.deepcopy(chromosomes)
        return None

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size
        return None

    def get_total_fitness(self):
        return self.total_fitness

    def set_total_fitness(self, total_fitness):
        self.total_fitness = total_fitness
        return None

    def compute_total_fitness(self):
        total_fitness = 0
        for it in range(self.size):
            total_fitness += self.chromosomes[it].get_fitness()
        return total_fitness

class GeneticOperators:
    # Selection
    def tournament_population_selection(self, population):
        J_SELECTED_CHROMOSOMES = np.random.randint(K_SELECTED_CHROMOSOMES / 4, K_SELECTED_CHROMOSOMES + 1)

        tournament_population = Population(J_SELECTED_CHROMOSOMES, 0)
        slected_chromosomes_indices = np.random.randint(0, population.get_size(), size = K_SELECTED_CHROMOSOMES)

        for it in range(K_SELECTED_CHROMOSOMES):
            tournament_population.get_chromosomes().append(population.get_chromosomes()[slected_chromosomes_indices[it]])

        tournament_population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)
        tournament_population.set_chromosomes(tournament_population.get_chromosomes()[0 : J_SELECTED_CHROMOSOMES])

        return tournament_population

    # Crossover
    def single_point_crossover(self, first_chromosome, second_chromosome, generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            cutting_point = rand.randrange(CHROMOSOME_SIZE)

            for it in range(CHROMOSOME_SIZE):
                if it <= cutting_point:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def average_crossover(self, first_chromosome, second_chromosome, generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            for it in range(CHROMOSOME_SIZE):
                first_offspring.get_genes()[it] = (first_chromosome.get_genes()[it]+second_chromosome.get_genes()[it]) / 2
                second_offspring.get_genes()[it] = (first_chromosome.get_genes()[it]+second_chromosome.get_genes()[CHROMOSOME_SIZE - it - 1]) / 2

            return (first_offspring, second_offspring)
        else:
            return None

    def flat_crossover(self, first_chromosome, second_chromosome, generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            for it in range(CHROMOSOME_SIZE):
                if first_chromosome.get_genes()[it] > second_chromosome.get_genes()[it]:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]


            return (first_offspring, second_offspring)
        else:
            return None

    def two_points_crossover(self, first_chromosome, second_chromosome, generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            first_cutting_point = rand.randrange(1, CHROMOSOME_SIZE - 3)
            second_cutting_point = rand.randrange(first_cutting_point + 2, CHROMOSOME_SIZE - 1)

            for it in range(CHROMOSOME_SIZE):
                if (it <= first_cutting_point) or (it >= second_cutting_point):
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def uniform_crossover(self, first_chromosome, second_chromosome, generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            for it in range(CHROMOSOME_SIZE):
                rand_value = rand.random()
                if rand_value < 0.5:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def single_arithmetic_crossover(self,first_chromosome, second_chromosome,generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            alpha = rand.random()
            rand_gene = rand.randrange(CHROMOSOME_SIZE)

            for it in range(CHROMOSOME_SIZE):
                if it == rand_gene:
                    first_offspring.get_genes()[it] = (alpha * first_chromosome.get_genes()[it] + (1 - alpha) * second_chromosome.get_genes()[it])
                    second_offspring.get_genes()[it] = (alpha * second_chromosome.get_genes()[it] + (1 - alpha) * first_chromosome.get_genes()[it])
                else:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def ring_crossover(self, first_chromosome, second_chromosome,generation):
        if first_chromosome.get_generation() == second_chromosome.get_generation():
            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            cutting_point = rand.randrange(CHROMOSOME_SIZE)

            for it in range(CHROMOSOME_SIZE):
                if it < cutting_point:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                else:
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            for it in range(CHROMOSOME_SIZE):
                if it < CHROMOSOME_SIZE - cutting_point:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    # Mutation
    def uniform_mutation(self, chromosome):
        rand_gene = rand.randrange(CHROMOSOME_SIZE)
        chromosome.get_genes()[rand_gene] = np.random.uniform(MIN_ALLELE_VALUE, MAX_ALLELE_VALUE)
        return chromosome

    def swap_mutation(self, chromosome):
        first_rand_gene = rand.randrange(CHROMOSOME_SIZE)
        second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        while first_rand_gene == second_rand_gene:
            second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        aux = chromosome.get_genes()[first_rand_gene]
        chromosome.get_genes()[first_rand_gene] = chromosome.get_genes()[second_rand_gene]
        chromosome.get_genes()[second_rand_gene] = aux

        return chromosome

    def scramble_mutation(self, chromosome):
        first_rand_gene = rand.randrange(CHROMOSOME_SIZE)
        second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        while second_rand_gene == first_rand_gene:
            second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        if second_rand_gene < first_rand_gene:
            aux = first_rand_gene
            first_rand_gene = second_rand_gene
            second_rand_gene = aux

        genes = chromosome.get_genes()[first_rand_gene : second_rand_gene]
        rand.shuffle(genes)

        for it in range(first_rand_gene, second_rand_gene):
            chromosome.get_genes()[it] = genes[it - first_rand_gene]

        return chromosome

    def inversion_mutation(self, chromosome):
        first_rand_gene = rand.randrange(CHROMOSOME_SIZE)
        second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        while first_rand_gene == second_rand_gene:
            second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        if first_rand_gene > second_rand_gene:
            aux = first_rand_gene
            second_rand_gene = first_rand_gene
            second_rand_gene = aux

        genes = chromosome.get_genes()[first_rand_gene : second_rand_gene]
        genes = genes[::-1]

        for it in range(first_rand_gene, second_rand_gene):
            chromosome.get_genes()[it] = genes[it - first_rand_gene]

        return chromosome

    def one_random_sign_mutation(self, chromosome):
        rand_gene = rand.randrange(CHROMOSOME_SIZE)
        chromosome.get_genes()[rand_gene] = -1 * chromosome.get_genes()[rand_gene]
        return chromosome

    def one_random_mean_mutation(self, chromosome):
        rand_gene = rand.randrange(CHROMOSOME_SIZE)
        _sum = 0
        for it in range(CHROMOSOME_SIZE):
            _sum = _sum + chromosome.get_genes()[it]
        chromosome.get_genes()[rand_gene] = _sum / CHROMOSOME_SIZE
        return chromosome

    def gaussian_mutation(self, chromosome):
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        random_gene = rand.randrange(CHROMOSOME_SIZE)
        chromosome.get_genes()[random_gene] = chromosome.get_genes()[random_gene] + s[0]

        while chromosome.get_genes()[random_gene] > MAX_ALLELE_VALUE:
            s = np.random.normal(mu, sigma, 1)
            chromosome.get_genes()[random_gene] = chromosome.get_genes()[random_gene] + s[0]

        return chromosome

def evolve(population, generation):
    new_population = Population(0, generation)
    genetic_operators = GeneticOperators()
    tournament_population = genetic_operators.tournament_population_selection(population)

    new_chromosomes = []
    new_size = 0

    for first_it in range(0, tournament_population.get_size()):
        for second_it in range(first_it + 1, tournament_population.get_size()):
            offsprings = create_offsprings(
                            genetic_operators,
                            tournament_population.get_chromosomes()[first_it],
                            tournament_population.get_chromosomes()[second_it],
                            generation)

            if offsprings != None:
                offsprings[0].set_fitness(fitness_function(offsprings[0]))
                offsprings[1].set_fitness(fitness_function(offsprings[1]))
                new_chromosomes.append(offsprings[0])
                new_chromosomes.append(offsprings[1])
                new_size += 2

            if new_size == POPULATION_SIZE:
                break

        if new_size == POPULATION_SIZE:
            break

    if new_size == 0:
        print("Err @ Population None!")

    new_population.set_chromosomes(new_chromosomes)
    new_population.set_size(new_size)
    new_population.set_total_fitness(new_population.compute_total_fitness())

    return new_population

def create_offsprings(genetic_operators, first_chromosome, second_chromosome, generation):
    offsprings = ()

    if g_crossover == SINGLE_POINT_CROSSOVER:
        offsprings = genetic_operators.single_point_crossover(first_chromosome, second_chromosome, generation)
    elif g_crossover == TWO_POINTS_CROSSOVER:
        offsprings = genetic_operators.two_points_crossover(first_chromosome, second_chromosome, generation)
    elif g_crossover == SINGLE_ARITHMETIC_CROSSOVER:
        offsprings = genetic_operators.single_arithmetic_crossover(first_chromosome, second_chromosome, generation)
    elif g_crossover == UNIFORM_CROSSOVER:
        offsprings = genetic_operators.uniform_crossover(first_chromosome, second_chromosome, generation)
    elif g_crossover == RING_CROSSOVER:
        offsprings = genetic_operators.ring_crossover(first_chromosome, second_chromosome, generation)
    elif g_crossover == AVERAGE_CROSSOVER:
        offsprings = genetic_operators.average_crossover(first_chromosome, second_chromosome, generation)
    elif g_crossover == FLAT_CROSSOVER:
        offsprings = genetic_operators.flat_crossover(first_chromosome, second_chromosome, generation)
    else:
        print("Err @ Invalid Crossover!")

    if offsprings != None:
        first_offspring = offsprings[0]
        second_offspring = offsprings[1]

        if g_mutation == UNIFORM_MUTATION:
            first_offspring = genetic_operators.uniform_mutation(first_offspring)
            second_offspring == genetic_operators.uniform_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        elif g_mutation == SWAP_MUTATION:
            first_offspring = genetic_operators.swap_mutation(first_offspring)
            second_offspring == genetic_operators.swap_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        elif g_mutation == SCRAMBLE_MUTATION:
            first_offspring = genetic_operators.scramble_mutation(first_offspring)
            second_offspring = genetic_operators.scramble_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        elif g_mutation == INVERSION_MUTATION:
            first_offspring = genetic_operators.inversion_mutation(first_offspring)
            second_offspring = genetic_operators.inversion_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        elif g_mutation == GAUSSIAN_MUTATION:
            first_offspring = genetic_operators.gaussian_mutation(first_offspring)
            second_offspring = genetic_operators.gaussian_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        elif g_mutation == ONE_RANDOM_SIGN_MUTATION:
            first_offspring = genetic_operators.one_random_sign_mutation(first_offspring)
            second_offspring = genetic_operators.one_random_sign_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        elif g_mutation == ONE_RANDOM_MEAN_MUTATION:
            first_offspring = genetic_operators.one_random_mean_mutation(first_offspring)
            second_offspring = genetic_operators.one_random_mean_mutation(second_offspring)
            offsprings = (first_offspring, second_offspring)
        else:
            print("Err @ Invalid Mutation!")

    return offsprings

def fitness_function(chromosome):
    fitness = 1 / benchmark_function(chromosome.get_genes())
    penalty = penalty_function(chromosome, fitness)
    return fitness + penalty

def penalty_function(chromosome, fitness):
    penalty = 0
    CONSTANT_PENALTY = 0.1

    if (fitness <= -0.5) or (fitness >= 0.5):
        penalty = fitness * CONSTANT_PENALTY

    return penalty

def benchmark_function(_list):

    # De Jong function
    _sum = 0

    list_len = len(_list)

    for it in range(list_len):
        _sum += pow(_list[it], 2)

    if _sum == 0:
        _sum = 0.000000000000001

    return _sum

def main():
    LOGS_FILE_PATH = "logs.txt"

    with open(LOGS_FILE_PATH, "w+") as fd:
        population = Population(POPULATION_SIZE, 0)
        most_fittest_chromosome = Chromosome(0)

        population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)
        most_fittest_chromosome = cp.deepcopy(population.get_chromosomes()[0])

        fd.write(
            "Generation #{0} | With Size: {1} | And Best Chromosome Benchmark: {3} | Global Best Chromosome Benchmark: {4}\n".format(
                0,
                population.get_size(),
                population.get_chromosomes()[0].get_genes(),
                benchmark_function(population.get_chromosomes()[0].get_genes()),
                benchmark_function(most_fittest_chromosome.get_genes())
            )
        )

        generation = 1
        while (generation < MAX_NUM_OF_GENERATIONS):
            population = evolve(population, generation)
            population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)

            if population.get_chromosomes()[0].get_fitness() > most_fittest_chromosome.get_fitness():
                most_fittest_chromosome = cp.deepcopy(population.get_chromosomes()[0])

            fd.write(
                "Generation #{0} | With Size: {1} | And Best Chromosome Benchmark: {3} | Global Best Chromosome Benchmark: {4}\n".format(
                    generation,
                    population.get_size(),
                    population.get_chromosomes()[0].get_genes(),
                    benchmark_function(population.get_chromosomes()[0].get_genes()),
                    benchmark_function(most_fittest_chromosome.get_genes())
                )
            )

            generation += 1

    return None

if __name__ == '__main__':
    EXPECTED_CMD_LINE_ARGS = 3
    SCRIPT_INDEX = 0
    CROSSOVER_INDEX = 1
    MUTATION_INDEX = 2

    if len(sys.argv) == EXPECTED_CMD_LINE_ARGS:
        g_crossover = int(sys.argv[CROSSOVER_INDEX])
        g_mutation = int(sys.argv[MUTATION_INDEX])

        main()
    else:
        print("Err @ Invalid Command Line Arguments!")
        print("Correct Syntax: python [script] [crossover] [mutation]")
        print("[script]:", sys.argv[SCRIPT_INDEX])
        print("[crossover]: SINGLE_POINT_CROSSOVER: 1 | TWO_POINTS_CROSSOVER: 2 | SINGLE_ARITHMETIC_CROSSOVER: 3 | UNIFORM_CROSSOVER: 4 | RING_CROSSOVER: 5 | AVERAGE_CROSSOVER: 6 | FLAT_CROSSOVER: 7")
        print("[mutation]: UNIFORM_MUTATION: 1 | SWAP_MUTATION: 2 | SCRAMBLE_MUTATION: 3 | INVERSION_MUTATION: 4 | GAUSSIAN_MUTATION: 5 | ONE_RANDOM_SIGN_MUTATION: 6 | ONE_RANDOM_MEAN_MUTATION: 7")
