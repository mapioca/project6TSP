import numpy as np
from typing import List
from TSPClasses import *

# Space Complexity: O(n^2)
#   - Contains an n x n cost matrix - O(n^2)
#   - Contains a partial tour (list) with up to n elements - O(n)
#   - Countains a bound (number) - O(1)
class State:
  def __init__(self, costs: np.ndarray, bound, partialTour: List[City]):
    self.costs = costs
    self.bound = bound
    self.partialTour = partialTour

  def __eq__(self, other):
    return self.bound == other.bound

  def __lt__(self, other):
    return self.bound < other.bound
