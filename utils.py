def find_k_nearest_neighbors(node, query_point, k, d, depth=0, k_nearest=None, distance_type="squared euclidean"):
    """
    Find the k-nearest neighbors to a query point in a k-d tree.

    Parameters:
    node (Node): The root node of the k-d tree.
    query_point (Point): The query point for which nearest neighbors are to be found.
    k (int): The number of nearest neighbors to find.
    d (int): Dimensionality of the points.
    depth (int): Current depth in the tree (default=0).
    k_nearest (list): List to store the k-nearest neighbors (default=None).

    Returns:
    list: A list containing tuples of distances and corresponding classes of the k-nearest neighbors.
    """
    if k_nearest is None:
        k_nearest = []

    if node is None:
        return k_nearest

    axis = depth % d

    if query_point.coordinates[axis] < node.element.point.coordinates[axis]:
        nearer_subtree = node.left
        further_subtree = node.right
    else:
        nearer_subtree = node.right
        further_subtree = node.left

    k_nearest = find_k_nearest_neighbors(node=nearer_subtree, query_point=query_point, k=k, d=d, depth=depth + 1, k_nearest=k_nearest, distance_type=distance_type)

    distance_to_query = query_point.distance_to(node.element.point, distance_type)

    if len(k_nearest) < k or distance_to_query < k_nearest[-1][0]:
        k_nearest.append((distance_to_query, node.element.elem_class))
        k_nearest.sort(key=lambda x: x[0])
        k_nearest = k_nearest[:k]

    if len(k_nearest) < k or abs(query_point.coordinates[axis] - node.element.point.coordinates[axis]) < k_nearest[-1][0]:
        k_nearest = find_k_nearest_neighbors(node=further_subtree, query_point=query_point, k=k, d=d, depth=depth + 1, k_nearest=k_nearest, distance_type=distance_type)

    return k_nearest
