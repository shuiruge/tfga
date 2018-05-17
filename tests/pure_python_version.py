"""Pure python implementation of the abstract base class of genetic
algorithms."""

import abc
import numpy as np



class Generation(object):
    """Auxillary class."""

    def __init__(self, population, fitnesses=None):
        """
        Args:
            population: Numpy array with shape `[n_individuals] + `
                `individual_shape`.
            fitnesses: Numpy array with shape `[n_individuals]`, or
                `None` as the initialized state, optional.
        """
        self.population = population
        self.fitnesses = fitnesses


class BaseGeneticAlgorithm(abc.ABC):

    def __init__(self,
                 n_individuals,
                 crossover_probability,
                 mutation_probability,
                 verbose=False):
        self.n_individuals = n_individuals
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.verbose = verbose

        # Initialize
        self.n_generations = 1

    def train(self):
        generation = self.initialize_generation()
        while not self.shall_stop(generation):
            generation = self.create_next_generation(generation)
            self.n_generations += 1
            if self.verbose:
                msg = 'Generation {0}, with the best fitness {1}'
                best_fitness = np.max(generation.fitnesses)
                print(msg.format(self.n_generations, best_fitness))
        return generation

    def create_next_generation(self, generation):
        """
        Args:
            generation: An instance of `Generation`.

        Returns:
            An instance of `Generation`.
        """
        # Population
        next_generation = self.select(generation)
        next_generation = self.crossover(next_generation)
        next_generation = self.mutate(next_generation)
        # Fitness
        next_fitnesses = self.get_fitnesses(next_generation.population)
        next_generation.fitnesses = next_fitnesses
        return next_generation

    def select(self, generation):
        """
        Args:
            generation: An instance of `Generation`.
        Returns:
            An instance of `Generation`.
        """
        probabilities = softmax(generation.fitnesses)
        n_selected = int((1-self.crossover_probability) * self.n_individuals)
        permutation = np.random.choice(
            self.n_individuals,
            size=n_selected,
            p=probabilities,
            replace=True)
        population = generation.population[permutation]
        fitnesses = generation.fitnesses[permutation]
        return Generation(population, fitnesses)
        
    def crossover(self, generation):
        """
        Args:
            generation: An instance of `Generation`.
        Returns:
            An instance of `Generation`.
        """
        n_individuals = generation.population.shape[0]
        if n_individuals != (1-self.crossover_probability)*self.n_individuals:
            raise ValueError('...')

        # Select individuals for crossover
        n_selected = 2 * int(self.crossover_probability*self.n_individuals/2)
        probabilities = softmax(generation.fitnesses)
        permutation = np.random.choice(
            n_individuals,
            size=n_selected,
            p=probabilities)
        selected = generation.population[permutation]

        # Get pairs as the shape `[2, n_selected/2] + individual_shape`
        individual_shape = list(selected.shape[1:])
        pairs = np.reshape(selected, [2, n_selected//2]+individual_shape)

        # Make crossover
        crossovered = np.array(
            [self._crossover(a, b) for a, b in zip(pairs[0], pairs[1])])
        crossovered = np.reshape(crossovered, [n_selected]+individual_shape)
        population = np.concatenate((generation.population, np.array(crossovered)),
                                    axis=0)
        return Generation(population)

    @abc.abstractmethod
    def _crossover(self, individual_a, individual_b):
        """
        Args:
            individual_a: Numpy array with the shape `individual_shape`.
            individual_b: Numpy array with the shape `individual_shape`.
        Returns:
            Tuple of two numpy arraies, both with the shape
            `individual_shape`.
        """
        pass

    def mutate(self, generation):
        """
        Args:
            generation: An instance of `Generation`.
        Returns:
            An instance of `Generation`.
        """
        new_population = []
        for individual in generation.population:
            if np.random.random() < self.mutation_probability:
                new_population.append(self._mutate(individual))
            else:
                new_population.append(individual)
        new_population = np.array(new_population)
        return Generation(new_population, generation.fitnesses)

    @abc.abstractmethod
    def _mutate(self, individual):
        """
        Args:
            individual: Numpy array with the shape `individual_shape`.
        Returns:
            Numpy array with the shape `individual_shape`.
        """
        pass

    def get_fitnesses(self, population):
        """
        Args:
            population: Numpy array with shape `[n_individuals] + `
                `individual_shape`.
        Returns:
            Numpy array with shape `[n_individuals]`.
        """
        return np.array([self.get_fitness(_) for _ in population])

    @abc.abstractmethod
    def get_fitness(self, individual):
        pass

    def initialize_generation(self):
        """Returns an instance of `Generation`."""
        population = self.initialize_population()
        fitnesses = self.get_fitnesses(population)
        return Generation(population, fitnesses)

    @abc.abstractmethod
    def initialize_population(self):
        pass

    @abc.abstractmethod
    def shall_stop(self, generation):
        pass


def softmax(x):
    """
    Args:
        x: Numpy array.
    Returns:
        Numpy array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':


    import time


    class SimpleGeneticAlgorithm(BaseGeneticAlgorithm):

        def __init__(self,
                     n_individuals,
                     crossover_probability,
                     mutation_probability,
                     **kwargs):
            super().__init__(n_individuals,
                             crossover_probability,
                             mutation_probability,
                             **kwargs)

        def get_fitness(self, x):
            fitness = np.sin(2 * np.pi * x)
            if x.shape:
                return np.mean(fitness)
            else:
                return fitness

        def _mutate(self, x):
            noise_scale = 0.2
            new_x = x + np.random.normal() * noise_scale
            condition = np.logical_and(new_x > np.zeros_like(x),
                                       new_x < np.ones_like(x))
            return np.where(condition, new_x, np.random.random(size=x.shape))

        def _crossover(self, x1, x2):
            return (x1 + (x2 - x1) * np.random.random(),
                    x1 + (x2 - x1) * np.random.random())

        def initialize_population(self):
            population = [
                np.random.random([100]) for _ in range(self.n_individuals)
            ]
            population = np.array(population, dtype='float32')
            print('Initialized population with shape', population.shape)
            return population

        def shall_stop(self, generation):
            max_fitness = np.max(generation.fitnesses)
            if max_fitness > 1 - 1e-2:
                return True
            else:
                return False

    def main():
        time_start = time.time()

        sga = SimpleGeneticAlgorithm(10**2, 0.3, 0.1, verbose=True)
        final_generation = sga.train()
        best_fitness = np.max(final_generation.fitnesses)

        time_elipsed = time.time() - time_start

        msg = ('After {0} generations, {1} seconds elipsed, '
               'the best fitness is gained as {2}')
        msg = msg.format(sga.n_generations, time_elipsed, best_fitness)
        print(msg)

    main()

