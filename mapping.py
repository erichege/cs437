import picar_4wd as fc
import time
import numpy as np
np.set_printoptions(threshold = np.inf)
import matplotlib.pyplot as plt
import math
import nodeClass as nc
from warnings import warn
import heapq

def get_xy(angle, distance):
	
	x = int(distance * np.sin(np.radians(angle)))
	#use 49 as car because base 0
	x += 49
	# catch edge case with rounding
	if x >= 100:
		x = 99
	
	y = int(distance * np.cos(np.radians(angle)))
	# catch edge case with rounding
	if y >= 100:
		y = 99
	return x,y
	
def draw_connection(map, x, y, last_x, last_y):
	
	if (x == last_x):
		slope = 0
	else:
		slope = (y-last_y)/(x-last_x)
	temp_y = 0
	for i in range(x-last_x):
		# use floor to assume closer, conservative approach
		temp_y += slope/(x-last_x)
		new_y = int(last_y + temp_y)
		if new_y >= 100:
			new_y = 99
		new_x = last_x + i
		if new_x >= 100:
			new_x = 99
		
		map[new_y, new_x] = 1
		
	return map
		

def fill_map(map):
	# Used to signal 2 positives in a row
	connected = False
	last_x = 0
	last_y = 0
	for angle in range(-60,61,5):
		dist = fc.get_distance_at(angle)
		if dist == -2:
			connected = False
			last_x = 0
			last_y = 0
			continue
		
		x,y = get_xy(angle,dist)
		
		map[y,x] = 1
		
		# If the next reading is far away likely not same object so don't combine
		if abs(last_y - y) > 15:
			connected = False
			last_x = 0
			last_y = 0
		# if two hits add in the inbetween info	
		if connected:
			map = draw_connection(map, x, y, last_x, last_y)
			last_x = x
			last_y = y
		else:
			last_x = x
			last_y = y
			connected = True
	return map
			
def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path
    
# draw a 5-4-4-3-1 circle around a point
def add_padding(map):
	for i in range(map.shape[1]):
		for j in range(map.shape[0]):
			if map[i,j] == 1:
				xc = j
				yc = i

				for z in range(0,2):
					for k in range(0,6):
						x_temp_1 = xc + k
						x_temp_2 = xc - k
						y_temp_1 = yc + z
						y_temp_2 = yc - z
						
						if (x_temp_1 < 99) and (y_temp_1 < 99):
							if map[y_temp_1,x_temp_1] != 1:
								map[y_temp_1, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_1 < 99:
							if map[y_temp_1,x_temp_2] != 1:
								map[y_temp_1, x_temp_2] = 2
						if x_temp_1 < 99 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_1] != 1:
								map[y_temp_2, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_2] != 1:
								map[y_temp_2, x_temp_2] = 2
				for z in range(2,4):
					for k in range(0,5):
						x_temp_1 = xc + k
						x_temp_2 = xc - k
						y_temp_1 = yc + z
						y_temp_2 = yc - z
						if (x_temp_1 < 99) and (y_temp_1 < 99):
							if map[y_temp_1,x_temp_1] != 1:
								map[y_temp_1, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_1 < 99:
							if map[y_temp_1,x_temp_2] != 1:
								map[y_temp_1, x_temp_2] = 2
						if x_temp_1 < 99 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_1] != 1:
								map[y_temp_2, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_2] != 1:
								map[y_temp_2, x_temp_2] = 2
				for z in range(4,5):
					for k in range(0,4):
						x_temp_1 = xc + k
						x_temp_2 = xc - k
						y_temp_1 = yc + z
						y_temp_2 = yc - z
						if (x_temp_1 < 99) and (y_temp_1 < 99):
							if map[y_temp_1,x_temp_1] != 1:
								map[y_temp_1, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_1 < 99:
							if map[y_temp_1,x_temp_2] != 1:
								map[y_temp_1, x_temp_2] = 2
						if x_temp_1 < 99 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_1] != 1:
								map[y_temp_2, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_2] != 1:
								map[y_temp_2, x_temp_2] = 2
				for z in range(5,6):
					for k in range(0,2):
						x_temp_1 = xc + k
						x_temp_2 = xc - k
						y_temp_1 = yc + z
						y_temp_2 = yc - z
						if (x_temp_1 < 99) and (y_temp_1 < 99):
							if map[y_temp_1,x_temp_1] != 1:
								map[y_temp_1, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_1 < 99:
							if map[y_temp_1,x_temp_2] != 1:
								map[y_temp_1, x_temp_2] = 2
						if x_temp_1 < 99 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_1] != 1:
								map[y_temp_2, x_temp_1] = 2
						if x_temp_2 > -1 and y_temp_2 > -1:
							if map[y_temp_2,x_temp_2] != 1:
								map[y_temp_2, x_temp_2] = 2
					
				
	return map
                             

def astar(maze, start, end):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """
 
    # Create start and end node
    start_node = nc.Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = nc.Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (maze.shape[0] * maze.shape[1] // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
   

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          return return_path(current_node)       
        
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (maze.shape[1] - 1) or node_position[0] < 0 or node_position[1] > (maze.shape[0]-1)  or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            #y,x format due to numpy arrays being the way they are
            if maze[node_position[1],node_position[0]] != 0:
                continue

            # Create new node
            new_node = nc.Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if child in closed_list:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None			
def translate_path(path):
    to_return_direction = []
    to_return_length = []
    #assume forward start
    curr_dir = (0,1)
    # second direction
    one_dir = (path[1][0]-path[0][0], path[1][1]-path[0][1])
    # Third direction
    two_dir = (path[2][0]-path[1][0], path[2][1]-path[1][1])
    #Forward
    if (curr_dir == one_dir == two_dir):
        to_return_direction.append('Forward')
        to_return_length.append(1)
    elif (curr_dir == one_dir != two_dir):
        to_return_direction.append('Forward')
        to_return_length.append(1)
    elif (curr_dir != one_dir == two_dir):
        if (one_dir == (1,0)):    
            to_return_direction.append('Left')
            to_return_length.append(1)
        else:
            to_return_direction.append('Right')
            to_return_length.append(1)
    #diag
    else:
        if (one_dir == (1,0)):    
            to_return_direction.append('LeftDiag')
            to_return_length.append(1)
        else:
            to_return_direction.append('RightDiag')
            to_return_length.append(1)
    for i in range(1,len(path) -2):
        curr_dir = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        # second direction
        one_dir = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        # Third direction
        two_dir = (path[i+2][0]-path[i+1][0], path[i+2][1]-path[i+1][1])
        print(curr_dir,one_dir,two_dir)
        #Forward
        if (curr_dir == one_dir == two_dir):
	    # already forward
            if (to_return_direction[-1] == 'Forward'):
                to_return_length[-1] += 1
            else: 
                to_return_direction.append('Forward')
                to_return_length.append(1)
        elif (curr_dir == one_dir != two_dir):
	    # already forward
            if (to_return_direction[-1] == 'Forward'):
                to_return_length[-1] += 1
            else: 
                to_return_direction.append('Forward')
                to_return_length.append(1)
            
        elif (curr_dir != one_dir == two_dir):
            if (one_dir == (1,0)):    
                to_return_direction.append('Left')
                to_return_length.append(1)
            else:
                to_return_direction.append('Right')
                to_return_length.append(1)
    #diag
        else:
	    # Not currently in diag
            if (to_return_direction[-1] != 'LeftDiag' and to_return_direction[-1] != 'RightDiag'):
                if (one_dir == (1,0)):    
                    to_return_direction.append('LeftDiag')
                    to_return_length.append(1)
                else:
                    to_return_direction.append('RightDiag')
                    to_return_length.append(1)
            else:
                    to_return_length[-1] += 1
    return to_return_direction,to_return_length
def drive_path(directions, steps):

    for i in range(len(directions)):

        if directions[i] == 'Forward':
            print('forward')
            fc.forward(1)
            time.sleep(.025 *steps[i])
            fc.stop()
            #Backwards
	#Left
        elif directions[i] == 'Left':
            print('left')
            fc.turn_left(1.3)
            time.sleep(.75)
            fc.stop()
            fc.forward(1)
            time.sleep(.2)
            fc.stop()
            
	    #Right
        elif directions[i] == 'Right':
            print('right')
            fc.turn_right(1.3)
            time.sleep(.75)
            fc.stop()
            fc.forward(1)
            time.sleep(.2)
            fc.stop()
	    
        elif directions[i] == 'RightDiag':
	    #helps weird almost wiggles
            if steps[i] <=2:
                print('rightdiag')
                fc.turn_right(.5)
                time.sleep(.25)
                fc.stop()
                fc.forward(1)
                time.sleep(.10* steps[i])
                fc.stop()
            else:
                fc.turn_right(.5)
                time.sleep(.25)
                fc.stop()
                fc.forward(1)
                time.sleep(.05* steps[i])
                fc.stop()
        elif directions[i] == 'LeftDiag':
            print('leftdiag')
	    #helps weird almost wiggles
            if steps[i] <=2:
                fc.turn_left(.5)
                time.sleep(.25)
                fc.stop()
                fc.forward(1)
                time.sleep(.10* steps[i])
                fc.stop()
            else:
                fc.turn_left(.5)
                time.sleep(.25)
                fc.stop()
                fc.forward(1)
                time.sleep(.05* steps[i])
                fc.stop()

if __name__ == '__main__':
	

	map = np.zeros((100,100), dtype=int)
	
	map = fill_map(map)
	map = add_padding(map)
	
	path = astar(map, (49,0),(49,60))
	
	for tup in path:
	    map[tup[1],tup[0]] = -1


	x,y=translate_path(path)
	drive_path(x,y)
	print(x)
	print(y)
	plt.imshow(map)
	plt.show()
	
