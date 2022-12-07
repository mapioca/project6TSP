#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from State import *
from typing import List
from queue import PriorityQueue
from py2opt.routefinder import RouteFinder


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    # Time Complexity: O(n^3)
    # 	- let n be the # of nodes/cities
    # 	- 3 nested loops, each O(n)
    #		- list copy (for route) is O(n)
    # 	- All other operations (math, comparisons, array access) are O(1)
    #
    # Space Complexity: O(n)
    #		- Stores a route (list) containing up to n items
    #   - Stores a set of visited cities (up to n items)
    #		- Other memory usage is constant
    #		- Not accounting for storage by parent object (such as _edge_exists array)
    def greedy(self, time_allowance=60.0):
        cities: List[City] = self._scenario.getCities()
        route = list()  # O(n) space as route is completed
        visited = set()  # O(n) space as nodes are visited
        results = {}
        startIndex = 0
        foundTour = False
        count = 0
        bssf = None

        startTime = time.time()

        for startCity in cities:  # O(n) time
            if time.time() - startTime >= time_allowance:
                break

            src: City = startCity
            visited.clear()
            visited.add(src)
            route.clear()
            route.append(src)

            for x in range(len(cities)):  # O(n) time
                min = math.inf
                minCity = None

                for dest in cities:  # O(n) time
                    if dest in visited:
                        continue

                    if self._scenario._edge_exists[src._index, dest._index]:
                        dist = src.costTo(dest)
                        if dist < min:
                            min = dist
                            minCity = dest

                if minCity is None:
                    break
                else:
                    visited.add(minCity)
                    route.append(minCity)
                    src = minCity

            if len(route) == len(cities) and self._scenario._edge_exists[route[-1]._index, route[0]._index]:
                foundTour = True
                count += 1
                solution = TSPSolution(route.copy())  # O(n) time

                if bssf is None or solution.cost < bssf.cost:
                    bssf = solution
            else:
                route = list()
                startIndex += 1

        endTime = time.time()

        results['cost'] = math.inf
        results['time'] = endTime - startTime
        results['count'] = count
        results['soln'] = None
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        if foundTour:
            results['cost'] = bssf.cost
            results['soln'] = bssf

        return results

    # Time Complexity: O(n^2)
    #		- 2 nested for loops (each O(n)) to populate cost matrix
    #		- All other operations (memory allocation, array access, etc.) are O(1)
    #
    # Space Complexity: O(n^2)
    #		- Creates an n x n cost matrix
    #		- Other memory usage is constant
    def createCostMatrix(self):
        cities = self._scenario.getCities()
        costs = np.empty((len(cities), len(cities)))  # O(1) time for memory allocation (not setting values)

        for src in cities:  # O(n) time
            for dest in cities:  # O(n) time
                if self._scenario._edge_exists[src._index, dest._index]:
                    costs[src._index, dest._index] = src.costTo(dest)
                else:
                    costs[src._index, dest._index] = math.inf

        return costs

    # Time Complexity: O(n^2)
    #		- O(n^2) to iterate over matrix to find row and column minimum values
    #		- O(n^2) to iterate over matrix to update values
    # 	- O(n) to sum row/column minimum value arrays
    # 	- All other operations are O(1)
    #
    # Space Complexity: O(n)
    #		- Does not account for space of input argument (an existing cost matrix)
    #		- Uses 2 arrays, each of length n, to store row and column minimum values
    def reduceCostMatrix(self, costs: np.ndarray):
        rowMins = costs.min(axis=1)  # Assume O(n^2) time implementation to iterate over entire matrix

        # O(n^2) time
        for x in range(costs.shape[0]):  # O(n) time
            if rowMins[x] == math.inf:
                rowMins[x] = 0
            costs[x, :] -= rowMins[x]  # O(n) time

        colMins = costs.min(axis=0)  # Assume O(n^2) time implementation to iterate over entire matrix

        # O(n^2) time
        for y in range(costs.shape[1]):  # O(n) time
            if colMins[y] == math.inf:
                colMins[y] = 0
            costs[:, y] -= colMins[y]  # O(n) time

        return np.sum(rowMins) + np.sum(colMins)  # O(n) time

    # Time Complexity: O(n^3)
    # 	- O(n) loop to iterate over cities
    #			- O(n^2) and O(n) copy operations
    #			- O(n) matrix updates
    #			- reduceCostMatrix() is O(n^2)
    #			- All other operations are O(1)
    #
    # Space Complexity: O(n^3)
    #		- Creates and returns a list of up to n-1 states - O(n)
    #			- Each state is O(n^2) space complexity (because of cost matrix)
    def expand(self, state: State):
        states: List[State] = list()
        srcIndex = state.partialTour[-1]._index

        city: City
        for city in self._scenario.getCities():  # O(n)
            destIndex = city._index
            if srcIndex == destIndex or state.costs[srcIndex, destIndex] == math.inf:
                continue
            else:
                # O(n^2) time to copy cost matrix
                # O(n) time to copy tour list
                newState: State = State(state.costs.copy(), state.bound, state.partialTour.copy())
                newState.partialTour.append(city)
                newState.bound += newState.costs[srcIndex, destIndex]
                newState.costs[srcIndex, :] = math.inf  # O(n) time
                newState.costs[:, destIndex] = math.inf  # O(n) time
                newState.costs[destIndex, srcIndex] = math.inf
                newState.bound += self.reduceCostMatrix(newState.costs)  # O(n^2) time
                states.append(newState)

        return states

    # Time Complexity: O(1)
    #		- All operations are constant time
    #
    # Space Complexity: O(n)
    #		- Creates and returns a new TSPSolution object
    #			- TSPSolution contains a partialTour (list) of up to n elements
    def toSolution(self, state: State):
        if len(state.partialTour) == len(self._scenario.getCities()):
            return TSPSolution(state.partialTour)
        else:
            return None

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    # Time Complexity: Worst case O(n!), average case O((b^n)*(n^3))
    #		- Let n be the number of cities
    #		- Let b be the average number of states added to the queue in each iteration
    #		- createCostMatrix() and reduceCostMatrix() are O(n^2)
    #		- queue.PriorityQueue (heap-based) put() and get() are O(log n)
    #		- Loop until priority queue is empty
    #			- Worst case is every possible state is explored - O(n!)
    #			- Average case is only b states are added to the queue each iteration - O(b^n)
    #			- expand() is O(n^3)
    #			- Queue operations are O(log n)
    #			- Other operations are O(1)
    #		- All other operations are O(1)
    #
    # Space Complexity: O((b^n)*(n^2))
    # 	- Has to store State objects in the priority queue
    #			- Average of b states are added to the queue each iteration - O(b^n)
    # 			- Each State object is O(n^2) space due to the cost matrix
    #  	- All other memory use is comparitively negligible
    def branchAndBound(self, time_allowance=60.0):
        q = PriorityQueue()
        numSolutions = 0
        maxQueueSize = 1
        numStates = 1
        numPrunedStates = 0

        costs: np.ndarray = self.createCostMatrix()  # O(n^2) time
        bound = self.reduceCostMatrix(costs)  # O(n^2) time
        initialPartialTour = [self._scenario.getCities()[0]]
        startState: State = State(costs, bound, initialPartialTour)

        greedyResults = self.greedy()  # O(n^3) time
        bssf = greedyResults['soln']
        bssfCost = greedyResults['cost']

        q.put((startState.bound / len(startState.partialTour), startState))  # O(log n) time

        startTime = time.time()

        while not q.empty() and time.time() - startTime < time_allowance:  # worst case O(n!), average case O(b^n) time
            state: State = q.get()[1]  # O(log n) time

            if state.bound < bssfCost:
                stateList: List[State] = self.expand(state)  # O(n^3) time
                numStates += len(stateList)

                curState: State
                for curState in stateList:  # O(n) time
                    solution = self.toSolution(curState)  # O(1) time
                    if solution is not None:
                        if solution.cost < bssfCost:
                            numSolutions += 1
                            bssf = solution
                            bssfCost = bssf.cost

                    elif curState.bound < bssfCost:
                        q.put((curState.bound / len(curState.partialTour), curState))  # O(log n)
                    else:
                        numPrunedStates += 1
            else:
                numPrunedStates += 1

            if q.qsize() > maxQueueSize:  # O(1) time
                maxQueueSize = q.qsize()  # O(1) time

        numPrunedStates += q.qsize()  # O(1) time

        endTime = time.time()

        results = {}
        results['cost'] = bssfCost
        results['time'] = endTime - startTime
        results['count'] = numSolutions
        results['soln'] = bssf
        results['max'] = maxQueueSize
        results['total'] = numStates
        results['pruned'] = numPrunedStates

        return results

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        start_time = time.time()
        ncities2 = ncities + 1
        reducedCost = [[0 for i in range(ncities2)] for j in range(ncities)]
        cities_names = []
        for i in range(ncities):
            current = cities[i]
            cities_names.append(current._index)
            for j in range(ncities):
                if j == 0:
                    reducedCost[current._index][ncities] = current.costTo(cities[j])
                destination = cities[j]
                reducedCost[current._index][destination._index] = current.costTo(cities[j])

        cities_names.append(0)
        route_finder = RouteFinder(reducedCost, cities_names, iterations=10)
        best_distance, best_route = route_finder.solve()
        path = []
        for i in best_route:
            path.append(cities[i])
        bssf = TSPSolution(path)
        end_time = time.time()
        results['cost'] = bssf.cost  # if foundTour else math.inf
        results['time'] = (end_time - start_time)
        results['count'] = None
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

