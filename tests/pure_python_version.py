"""Pure python implementation of the abstract base class of genetic
algorithms."""

import abc
import numpy as np



class Generation(object):
    """Auxillary class."""

    def __init__(self, population, log_fitnesses=None):
        """
        Args:
            population: Numpy array with shape `[n_individuals] + `
                `individual_shape`.
            log_fitnesses: Numpy array with shape `[n_individuals]`, or
                `None` as the initialized state, optional.
        """
        self.population = population
        self.log_fitnesses = log_fitnesses


class BaseGeneticAlgorithm(abc.ABC):

    def __init__(self,
                 n_individuals,
                 crossover_probability,
                 mutation_probability):
        self.n_individuals = n_individuals
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self._probabilities = None  # as cache.

    def train(self):
        generation = self.initialize_generation()
        while not self.shall_shop(generation):
            generation = self.create_next_generation(generation)
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
        next_log_fitnesses = self.get_log_fitnesses(next_generation.population)
        next_generation.log_fitnesses = next_log_fitnesses
        return next_generation

    def select(self, generation):
        """
        Args:
            generation: An instance of `Generation`.
        Returns:
            An instance of `Generation`.
        """
        self._probabilities = softmax(generation.log_fitnesses)
        n_selected = (1 - self.crossover_probability) * self.n_individuals
        permutation = np.random.choice(
            np.arrange(self.n_individuals, dtype='int32'),
            size=n_selected,
            p=self._probabilities,
            replace=False)
        population = generation.population[permutation]
        log_fitnesses = generation.log_fitnesses[permutation]
        return Generation(population, log_fitnesses)
        
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
        permutation = np.random.choice(
            np.arrange(n_individuals, dtype='int32'),
            size=n_selected,
            p=self._probabilities)
        selected = generation.population[permutation]

        # Get pairs as the shape `[2, n_selected/2] + individual_shape`
        individual_shape = selected.shape[1:]
        pairs = selected.reshape([2, n_selected/2]+individual_shape)

        # Make crossover
        crossovered = [self._crossover(a, b) for a, b in zip(pairs[0], pairs[1])]
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
          Numpy array with the shape `individual_shape`.
        """
        pass

    def mutate(self, generation):
        """
        Args:
            generation: An instance of `Generation`.
        Returns:
            An instance of `Generation`.
        """
        new_population = np.array([
            self._mutate(individual) for individual in generation.population
        ])
        return Generation(new_population, generation.log_fitnesses)

    @abc.abstractmethod
    def _mutate(self, individual):
        """
        Args:
            individual: Numpy array with the shape `individual_shape`.
        Returns:
          Numpy array with the shape `individual_shape`.
        """
        pass

    @abc.abstractmethod
    def get_log_fitnesses(self, population):
        pass

    def initialize_generation(self, generation):
        """Returns an instance of `Generation`."""
        population = self.initialize_population()
        log_fitnesses = self.get_log_fitnesses(population)
        return Generation(population, log_fitnesses)

    @abc.abstractmethod
    def initialize_population(self):
        pass

    @abc.abstractmethod
    def shall_stop(self, generation):
        pass
