import picar_4wd as fc
import numpy as np
np.set_printoptions(threshold = np.inf)
import matplotlib.pyplot as plt
import math

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
	for i in range(x-last_x):
		# use floor to assume closer, conservative approach
		temp_y = math.floor(slope/(x-last_x))
		
		map[last_y + temp_y, last_x+i,] = 1
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
			
			
		


if __name__ == '__main__':
	

	map = np.zeros((100,100), dtype=int)
	fill_map(map)
	print(map)


	plt.imshow(map, origin = 'lower')
	plt.show()

