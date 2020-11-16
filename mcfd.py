import numpy as np


def generate_mask(shape, coordinates, radius, dtype=np.int32):
    # inspired from https://stackoverflow.com/a/8650741

    dim_count = len(shape)
    mask = None
    for i in range(dim_count):
        n = shape[i]
        coordinate = coordinates[i]
        indexes = np.ogrid[-coordinate:n - coordinate]

        new_shape = np.ones(dim_count, dtype=dtype)
        new_shape[i] = n
        # print(new_shape)
        tensor_component_of_indexes = np.reshape(indexes, tuple(new_shape))

        tensor_component_squared = tensor_component_of_indexes**2
        if mask is None:
            mask = tensor_component_squared
        else:
            mask = mask + tensor_component_squared

    mask = mask <= radius**dim_count

    return mask


def generate_tensor(shape, cluster_count=3, cluster_radius=3, low=0, high=255, eps=10, dtype=np.int32):
    """
    It will return a tensor of shape 'shape' with 'cluster_count' neighborhood with center locations chosen randomly.
        A random value <point_value> will be chosen for each cluster and all values in the neighborhood of radius 'cluster_radius'
        will have values in (point_value-eps, point_value+eps)

    :param shape: Numpy tensor shape
    :param cluster_count: How many clusters to generate
    :param cluster_radius: The radius of the cluster
    :param low: Lowest value of data values space
    :param high: Highest value of data values space
    :param eps: The epsilon used to generate neighborhood.
    :param dtype: Default is np.int32
    """

    tensor = np.zeros(shape, dtype)

    for i in range(cluster_count + 1):
        # cluster = {}

        coordinates = []
        for dim in shape:
            coordinate = np.random.randint(0, high=dim)
            coordinates.append(coordinate)

        coordinates = tuple(coordinates)

        mask = generate_mask(shape, coordinates, cluster_radius)

        point_value = np.random.randint(low=low, high=high, size=1)

        random_values = np.random.randint(low=point_value-eps, high=point_value+eps+1, size=shape)
        random_values_masked = np.ma.array(random_values, mask=np.logical_not(mask))

        idx = (mask > 0)
        tensor[idx] = random_values_masked[idx]

        print(tensor)

    return tensor


