class Point:
    def __init__(self, coordinates):
        """
        Initialize a Point object with given coordinates.

        Parameters:
        coordinates (list or tuple): The coordinates of the point.
        """
        self.coordinates = coordinates

    def distance_to(self, other_point):
        """
        Calculate the squared Euclidean distance between this point and another point.

        Parameters:
        other_point (Point): Another Point object.

        Returns:
        float: Squared Euclidean distance between the points.
        """
        return sum((a - b) ** 2 for a, b in zip(self.coordinates, other_point.coordinates))

    def __str__(self):
        return str(self.coordinates)


class Element:
    def __init__(self, elem_point, elem_class):
        """
        Initialize an Element object with a point and its associated class.

        Parameters:
        elem_point (Point): The Point object.
        elem_class: The class associated with the Element.
        """
        self.point = elem_point
        self.elem_class = elem_class

    def __str__(self):
        return str(self.point) + ", class: " + str(self.elem_class)


class Node:
    def __init__(self, element_list, depth, d):
        """
        Initialize a Node object for a k-d tree.

        Parameters:
        element_list (list): List of Element objects.
        depth (int): Current depth in the tree.
        d (int): Dimensionality of the points.
        """
        axis = depth % d

        element_list.sort(key=lambda x: x.point.coordinates[axis])
        median = len(element_list) // 2

        self.element = element_list[median]

        if len(element_list[:median]) != 0:
            self.left = Node(element_list[:median], depth + 1, d)
        else:
            self.left = None

        if len(element_list[median + 1:]) != 0:
            self.right = Node(element_list[median + 1:], depth + 1, d)
        else:
            self.right = None

    def __str__(self):
        return "Node: " + str(self.element) + "\n   left: " + str(self.left.element) + "\n   right: " + str(
            self.right.element)
