import numpy as np
from heapq import heappush, heappop
from animation import draw
import argparse

class Node():
    """
    cost_from_start - the cost of reaching this node from the starting node
    state - the state (row,col)
    parent - the parent node of this node, default as None
    """
    def __init__(self, state, cost_from_start, parent = None):
        self.state = state
        self.parent = parent
        self.cost_from_start = cost_from_start


class EightPuzzle():
    
    def __init__(self, start_state, goal_state, method, algorithm, array_index):
        self.start_state = start_state
        self.goal_state = goal_state
        self.visited = [] # state
        self.method = method
        self.algorithm = algorithm
        self.m, self.n = start_state.shape 
        self.array_index = array_index

    def goal_test(self, current_state):
        return np.array_equal(current_state, self.goal_state)

    def get_cost(self, current_state, next_state):
        return 1

    def get_successors(self, state):
        successors = []
        row_change = [0, 0, -1, 1]
        col_change = [-1, 1, 0, 0]

        empty_position = np.where(state == 0)
        empty_y, empty_x = empty_position[0][0], empty_position[1][0]

        for i in range(4):
            if ((empty_x + col_change[i] < self.n and empty_y + row_change[i] >= 0) and
                    (empty_x + col_change[i] >= 0 and empty_y + row_change[i] < self.m)):
                copy = state.copy()
                temp = copy[int(empty_y) + row_change[i]][int(empty_x) + col_change[i]]
                copy[int(empty_y) + row_change[i]][int(empty_x) + col_change[i]] = 0
                copy[int(empty_y)][int(empty_x)] = temp
                successors.append(copy)
        return successors

    # heuristics function
    def heuristics(self, state):
        cost = 0

        if self.method == 'Hamming':
            for r in range(len(state)):
                for c in range(len(state[r])):
                    if state[r][c] != self.goal_state[r][c]:
                        cost += 1

        elif self.method == 'Manhattan':
            correct_coordinates = {}
            for r in range(len(self.goal_state)):
                for c in range(len(self.goal_state[r])):
                    correct_coordinates[self.goal_state[r][c]] = [r, c]

            for r in range(len(state)):
                for c in range(len(state[r])):
                    correct_x_y = correct_coordinates[state[r][c]]
                    cost += abs(r - correct_x_y[0]) + abs(c - correct_x_y[1])

        return cost

    # priority of node 
    def priority(self, node):
        if self.algorithm == 'Greedy':
            return self.heuristics(node.state)
        elif self.algorithm == 'AStar':
            return self.heuristics(node.state) + node.cost_from_start
    
    # draw 
    def draw(self, node):
        path=[]
        while node.parent:
            path.append(node.state)
            node = node.parent
        path.append(self.start_state)

        draw(path[::-1], self.array_index, self.algorithm, self.method)

    # solve it
    def solve(self):
        state = self.start_state.copy()

        if self.goal_test(state):
            return state

        node = Node(state, 0, None)
        self.visited.append(node)
        index = 0
        priority_queue = [(self.priority(node), index, node)]


        while priority_queue:
            best_node = heappop(priority_queue)[2]

            successors = self.get_successors(best_node.state)

            for next_state in successors:
                seen = False
                for visited in self.visited:
                    if np.array_equal(next_state, visited):
                        seen = True
                        break

                if seen:
                    continue

                self.visited.append(next_state)

                next_node = Node(next_state, best_node.cost_from_start + 1, best_node)

                if self.goal_test(next_state):
                    self.draw(next_node)
                    return

                index = index + 1
                heappush(priority_queue, (self.priority(next_node), index, next_node))


if __name__ == "__main__":
    
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])
    start_arrays = [np.array([[1,2,0],[3,4,6],[7,5,8]]),
                    np.array([[8,1,3],[4,0,2],[7,6,5]])]
    methods = ["Hamming", "Manhattan"]
    algorithms = ['Greedy', 'AStar']
    
    parser = argparse.ArgumentParser(description='eight puzzle')

    parser.add_argument('-array', dest='array_index', required = True, type = int, help='index of array')
    parser.add_argument('-method', dest='method_index', required = True, type = int, help='index of method')
    parser.add_argument('-algorithm', dest='algorithm_index', required = True, type = int, help='index of algorithm')

    args = parser.parse_args()

    # Example:
    # Run this in the terminal using array 0, method Hamming, algorithm AStar:
    #     python eight_puzzle.py -array 0 -method 0 -algorithm 3
    game = EightPuzzle(start_arrays[args.array_index], goal, methods[args.method_index], algorithms[args.algorithm_index], args.array_index)
    game.solve()