import abc
import numpy as np
import tensorflow as tf


class Generation(object):
  """Helper class. Represents a generation in genetic algorithm."""

  def __init__(self, population, fitnesses):
    """
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`.
    """
    self.population = population
    self.fitenesses = fitenesses


class BaseGeneticAlgorithm(abc.ABC):
  """Abstract base class for genetic algorithms.

  NOTE:
    population: Tensor with shape `[n_individuals] + individual_shape`.
    fitnesses: Tensor with shape `[n_individuals]`.
  """

  def __call__(self):
    """Runs the genetic algorithm, and returns the final population and their
    fitness-values.

    Returns:
      An instance of `Generation`.
    """
    population = self.initialize_population()
    n_generations = tf.constant(0)
    fitenesses = self.get_fitnesses(population)
    cond = lambda x, y: tf.logical_not(self.shall_stop_evolve(x, y))
    final_population, final_fitenesses = \
        while_loop(cond, create_next_generation, [population, fitenesses])
    return Generation(final_population, final_fitenesses)

  def create_next_generation(self, population, fitenesses):
    """The iterate-body of the `while_loop`
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`.

    Returns:
      The same as the arguments.
    """
    new_populaiton = self.get_new_population(population, fitnesses)
    new_fitnesses = self.get_fitnesses(population)
    return new_populaiton, new_fitnesses

  def get_new_population(self, population, fitnesses):
    """
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`.

    Returns:
      Tensor with the shape `[n_individuals] + individual_shape`.
    """
    new_population = self.select(population, fitnesses)
    new_population = self.crossover(new_population, fitnesses)
    new_population = self.mutate(new_population, fitnesses)
    return new_population

  @abc.abstractmethod
  def initialize_population(self):
    """Returns a tensor with the shape `[n_individuals] + individual_shape`."""
    pass

  @abc.abstractmethod
  def shall_stop_evolve(self, fitnesses):
    """
    Args:
      fitnesses: Tensor with shape `[n_individuals]`.

    Returns:
      Boolean.
    """
    pass

  @abc.abstractmethod
  def get_log_fitenesses(self, population):
    """
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.

    Returns:
      Tensor with shape `[n_individuals]`.
    """
    pass

  @abc.abstractmethod
  def select(self, population, fitnesses):
    """
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`.

    Returns:
      Tensor with the shape `[n_individuals] + individual_shape`.
    """
    pass

  @abc.abstractmethod
  def crossover(self, population, fitnesses):
    """
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`.

    Returns:
      Tensor with the shape `[n_individuals] + individual_shape`.
    """
    pass

  @abc.abstractmethod
  def mutate(self, population, fitnesses):
    """
    Args:
      population: Tensor with the shape `[n_individuals] + individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`.

    Returns:
      Tensor with the shape `[n_individuals] + individual_shape`.
    """
    pass

