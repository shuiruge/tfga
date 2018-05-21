"""TensorFlow implementation of the abstract base class of genetic
algorithms."""

import abc
import numpy as np
import tensorflow as tf


class Generation(object):
  """Auxillary class."""

  def __init__(self, population, fitnesses=None):
    """
    Args:
      population: Tensor with shape `[n_individuals] + `
        `individual_shape`.
      fitnesses: Tensor with shape `[n_individuals]`, or
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
    self.generation = self.initialize_generation()

  def evolve(self):
    """Update the `self.generation`."""
    with tf.name_scope('Evolve'):
      # Population
      selection_op = self.select(self.generation)

      with tf.control_dependencies([selection_op]):
        crossover_op = self.crossover(self.generation)

      with tf.control_dependencies([crossover_op]):
        mutation_op = self.mutate(self.generation)

      # Fitness
      with tf.control_dependencies([mutation_op]):
        update_fitenesses_op = self.update_fitenesses()

      evolve_op = tf.group(selection_op, crossover_op,
                           mutation_op, update_fitenesses_op)
      return evolve_op
