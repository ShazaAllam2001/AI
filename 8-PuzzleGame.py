import time, math, heapq
import numpy as np
from copy import deepcopy 

SUCCESS = True
FAILURE = False
GOALSTATE = "012345678"

# Algorithms available
BFS = '0'
DFS = '1'
A_STAR_ECULIDEAN = '2_0'
A_STAR_MANHATTAN = '2_1'


# Node class ->
class Node:
    parent = None
    children = []
    grid = None
    direction = None
    depth = 0
    path_cost = 0
    estimated_cost = 0
    
    def __init__(self, grid, direction, depth):
        self.grid = grid
        self.direction = direction
        self.depth = depth
        # path_cost equals depth -as the cost of every edge is equal to 1-
        self.path_cost = depth 

    def __lt__(self, other):
        return self.depth > other.depth

    def __le__(self, other):
        return self.depth > other.depth

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    # Printing node
    def print_node_str(self):
        print(self.direction)
        for i in range(0, 9, 3):
            print(self.grid[i], self.grid[i+1], self.grid[i+2])

     # Printing node
    def print_node_arr(self):
        print(self.direction)
        for i in range(0,3):
            for j in range(0,3):
                print(self.grid[i][j],end = "  ") 
            print()

    # in each function of the four functions:
    # check first if it is a legal move -> if true apply the move and if false return None
    def move_to_up(self, g, i):
        if i-3 >= 0:
            newgrid = list(g)
            newgrid[i] = newgrid[i-3]
            newgrid[i-3] = '0'
            g = ''.join(newgrid)
            return Node(g,'Up',self.depth+1)
        return None

    def move_to_down(self, g, i):
        if i+3 < 9:
            newgrid = list(g)
            newgrid[i] = newgrid[i+3]
            newgrid[i+3] = '0'
            g = ''.join(newgrid)
            return Node(g,'Down',self.depth+1)
        return None
        
    def move_to_left(self, g, i):
        if i%3 > 0:
            newgrid = list(g)
            newgrid[i] = newgrid[i-1]
            newgrid[i-1] = '0'
            g = ''.join(newgrid)
            return Node(g,'Left',self.depth+1)
        return None

    def move_to_right(self, g, i):
        if i%3 < 2:
            newgrid = list(g)
            newgrid[i] = newgrid[i+1]
            newgrid[i+1] = '0'
            g = ''.join(newgrid)
            return Node(g,'Right', self.depth+1)
        return None

    # expanding each node with the possible movements in each of the four directions
    def expand_node_str(self):
        zeroPostion = self.grid.index('0')
                        
        Up = self.move_to_up(self.grid, zeroPostion)
        if Up!=None:
            # if the up node not equal none so add it to the node's children
            self.add_child(Up)

        Down = self.move_to_down(self.grid, zeroPostion)
        if Down!=None:
            # if the down node not equal none so add it to the node's children
            self.add_child(Down)
        
        Left = self.move_to_left(self.grid, zeroPostion)
        if Left!=None:
            # if the left node not equal none so add it to the node's children
            self.add_child(Left)

        Right = self.move_to_right(self.grid, zeroPostion)
        if Right!=None:
            # if the right node not equal none so add it to the node's children
            self.add_child(Right)

    # expand the node with the possible movements in each of the four directions
    def expand_node_arr(self):
        empty_space = np.argwhere(self.grid == 0)
        # the possible up and down moves
        if (empty_space[0][0] != 0):
            # up case
            newNode1 = Node(deepcopy(self.grid), "Up", self.depth+1)
            self.add_child(newNode1)
            newNode1.grid[empty_space[0][0]][empty_space[0][1]] = newNode1.grid[empty_space[0][0]-1][empty_space[0][1]]
            newNode1.grid[empty_space[0][0]-1][empty_space[0][1]] = 0
        if (empty_space[0][0] != 2):
            # down case
            newNode2 = Node(deepcopy(self.grid), "Down", self.depth+1)
            self.add_child(newNode2)
            newNode2.grid[empty_space[0][0]][empty_space[0][1]] = newNode2.grid[empty_space[0][0]+1][empty_space[0][1]]
            newNode2.grid[empty_space[0][0]+1][empty_space[0][1]] = 0

        # the possible right and left moves
        if (empty_space[0][1] != 0):
            # left case
            newNode3 = Node(deepcopy(self.grid), "Left", self.depth+1)
            self.add_child(newNode3)
            newNode3.grid[empty_space[0][0]][empty_space[0][1]] = newNode3.grid[empty_space[0][0]][empty_space[0][1]-1]
            newNode3.grid[empty_space[0][0]][empty_space[0][1]-1] = 0
        if (empty_space[0][1] != 2):
            # right case
            newNode4 = Node(deepcopy(self.grid), "Right", self.depth+1)
            self.add_child(newNode4)
            newNode4.grid[empty_space[0][0]][empty_space[0][1]] = newNode4.grid[empty_space[0][0]][empty_space[0][1]+1]
            newNode4.grid[empty_space[0][0]][empty_space[0][1]+1] = 0
        return
            
# -------------------------------------------------------------------------------------------------------------

# gameGraph class ->
class gameGraph:
    solution_path = []
    nodes_expanded = 0
    running_time = None

    def __init__(self):
        pass

    def is_solvable(self, grid):
        # number of inversions -> 
        # A pair of tiles their values are in reverse order of their appearance in goal state.
        inv_count = 0 
        for i in range(8):
            for j in range(i+1, 9):
                if grid[j]!='0' and grid[i]>grid[j]:
                    inv_count += 1

        return (inv_count % 2 == 0)
            
    # Tracing backwards to find the path between the root and the goal
    def pathTrace(self, node):
        path = []
        path.append(node)
        while node.parent != None:
            node = node.parent
            path.append(node)
        path.reverse()
        return path

    # Printing path between intial state and the goal
    def printSolution_str(self):
        for step in self.solution_path:
            print("Step", step.depth, ': ', end='') 
            step.print_node_str()
            print("_________________________________")
        print("cost_of_path =", self.solution_path[-1].path_cost)
        print("nodes_expanded =", self.nodes_expanded)
        print("search_depth =", self.solution_path[-1].depth)
        print("running_time =", self.running_time, "ms")

    # Main function for executing the puzze; solver
    def solve_puzzle(self, intialState, algorithm):
        if algorithm == '0':
            print("BFS Algotithm")
            solution_found = self.BFS(intialState)
            if solution_found:
                print("There is a solution.")
                self.printSolution_str()
            else:
                print("There is no solution.")
            
        elif algorithm == '1':
            print("DFS Algotithm")
            solution_found = self.DFS(intialState)
            if solution_found:
                print("There is a solution.")
                self.printSolution_str()
            else:
                print("There is no solution.")

        elif algorithm == '2_0':
            print("A* Algotithm")
            solution_found = self.A_star(intialState, self.Heuristic_Euclidean)
            if solution_found:
                print("There is a solution.")
                self.printSolution_arr("Euclidean")
            else:
                print("There is no solution.")

        elif algorithm == '2_1':
            print("A* Algotithm")
            solution_found = self.A_star(intialState, self.Heuristic_Manhattan)
            if solution_found:
                print("There is a solution.")
                self.printSolution_arr("Manhattan")
            else:
                print("There is no solution.")

    # ......................................................................................

    # BFS algorithm :-
    def BFS(self, initial_state): 
        start = time.time()
        intial_node = Node(initial_state, 'Initial state', 0)
        frontier = [intial_node]
        explored = set()
        
        if self.is_solvable(initial_state):
            while len(frontier) != 0:
                state = frontier.pop(0)
                explored.add(state.grid)
                
                if GOALSTATE == state.grid:
                    self.running_time = (time.time() - start)* 10**3
                    self.nodes_expanded = len(explored) - 1
                    self.solution_path = self.pathTrace(state)
                    return SUCCESS
                    
                state.expand_node_str()
                
                for child in state.children:
                    if child not in frontier and child.grid not in explored:
                        frontier.append(child)

        return FAILURE

    # DFS algorithm :-
    def DFS(self, initial_state): 
        start = time.time()
        intial_node = Node(initial_state, 'Initial state', 0)
        frontier = [intial_node]
        explored = set()
        
        if self.is_solvable(initial_state):
            while len(frontier) != 0:
                state = frontier.pop(-1)
                explored.add(state.grid)
                
                if GOALSTATE == state.grid:
                    self.running_time = (time.time() - start)* 10**3
                    self.nodes_expanded = len(explored) - 1
                    self.solution_path = self.pathTrace(state)
                    return SUCCESS
                    
                state.expand_node_str()
                
                for child in state.children:
                    if child not in frontier and child.grid not in explored:
                        frontier.append(child)

        return FAILURE

    # ......................................................................................

    # convert the string input to 2d array
    def string_to_2dArray(self, state):
        arr = []
        for d in state:
            if (d.isdigit()):
                arr.append(int(d))
        arr = np.reshape(arr, (3, 3))
        return arr

    # Printing path between intial state and the goal
    def printSolution_arr(self, Heuristic_type):
        for step in self.solution_path:
            print("Step", step.depth, ': ', end='') 
            step.print_node_arr()
            print("_________________________________")
        print("Heuristic_type :-", Heuristic_type)
        print("cost_of_path =", self.solution_path[-1].path_cost)
        print("nodes_expanded =", self.nodes_expanded)
        print("search_depth =", self.solution_path[-1].depth)
        print("running_time =", self.running_time, "ms")

    # test if this state is the goal state 
    def goal_test(self , state):
        if (np.array_equal( state, np.array([[0,1,2],[3,4,5],[6,7,8]]) )):
            return True
        return False

    # Higher order function to calculate the heuristic
    # take the hueristic function as attribute and use it
    def calculate_Heuristic(self, state, heuristic_func):
        h = 0 
        goal_coordinates = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
        for i in range(0,3):
            for j in range(0,3):
                x = state[i][j]
                if x == 0:
                    continue
                h += heuristic_func(i, j , goal_coordinates[x])
        return h

    # calculate the Manhattan Heuristic
    def Heuristic_Manhattan(self ,i, j , goal_coordinates):
        return abs(i - goal_coordinates[0]) + abs(j - goal_coordinates[1])

    # calculate the Euclidean Heuristic
    def Heuristic_Euclidean(self, i, j , goal_coordinates):
        return math.sqrt( (i - goal_coordinates[0])**2 + (j - goal_coordinates[1])**2 )

    # A* algorithm :-
    def A_star(self , initial_state, heuristic_func):
        start = time.time()
        state = self.string_to_2dArray(initial_state)
        initial_node = Node(state, 'Intial state', 0)
        frontier = []
        cost = self.calculate_Heuristic(state, heuristic_func)
        initial_node.estimated_cost = cost
        heapq.heappush(frontier , (cost, initial_node))
        explored = []

        if self.is_solvable(initial_state):
            while frontier:
                c, state = heapq.heappop(frontier)
                explored.append(state.grid)

                if (self.goal_test(state.grid)):
                    self.running_time = (time.time() - start)* 10**3
                    self.nodes_expanded = len(explored) - 1
                    self.solution_path = self.pathTrace(state)
                    return SUCCESS
                    
                state.expand_node_arr()

                for child in state.children:
                    # if neighbor in frontier
                    flag = 0
                    j = 0
                    for co,no in frontier:
                        if np.array_equal(child.grid , no.grid):
                            if (child.estimated_cost < co):
                                # update the priority
                                frontier[j] = (child.estimated_cost, child)
                                heapq.heapify(frontier)
                            flag = 1
                            break
                        j += 1
                    # if neighbor not in explored & not in frontier
                    if flag == 0:
                        flag2 = 0
                        for e in explored:
                            if np.array_equal(child.grid , e):
                                flag2 = 1
                                break
                        if flag2 == 0 :
                            cost = child.path_cost + self.calculate_Heuristic(child.grid , heuristic_func)
                            child.estimated_cost = cost
                            heapq.heappush(frontier , (cost,child))
        return FAILURE


# -------------------------------------------------------------------------------------------------------------   

Game = gameGraph()

#Game.solve_puzzle("102345678", BFS)

#Game.solve_puzzle("102345678", DFS)

#Game.solve_puzzle("102345678", A_STAR_ECULIDEAN)

Game.solve_puzzle("125348067", A_STAR_MANHATTAN)

