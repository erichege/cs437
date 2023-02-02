class Node:
    """
    A node class for A* Pathfinding
 	Credit for this: Nicholas Swift
  	as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
		
		# Distance between current node and start node
        self.g = 0
        # Estimated distance from current to end node
        self.h = 0
        # Total cost of the node
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f
