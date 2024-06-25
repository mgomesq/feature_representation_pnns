import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import Dataset

def outside_n_sphere(point, n_sphere_radius, n_sphere_center):
    ''' Checks if provided point is inside/outside an n-sphere
    '''

    assert len(n_sphere_center) == len(point), "Point and Sphere have to be in same dimension!"

    radius_point =  np.sqrt(sum((point - n_sphere_center)**2))
    is_outside = radius_point > n_sphere_radius
    return is_outside


def generate_balanced_dataset(n_points, bounds, n_dimension, n_sphere_radius, n_sphere_center):
    ''' Create a dataframe to be used for training classification of points
    inside an n-sphere

    Arguments
    ---------
    n_points - number of datapoints
    bounds - definition of bounds of datapoints (-bound,+bound)
    n_dimension - number of dimensions of the n-sphere and points
    n_sphere_radius - radius of the n_sphere
    n_sphere_center - central point of the n_sphere
    '''

    assert bounds > 0, 'Bounds should be a positive non zero number'
    assert len(n_sphere_center) == n_dimension, 'Number of dimentions must match'
    assert isinstance(n_sphere_center, np.ndarray)

    outside_points = []
    inside_points = []

    while len(outside_points) + len(inside_points) < n_points:
        random_point = ((np.random.rand(n_dimension)*2) - 1) * bounds

        is_outside = outside_n_sphere(random_point, n_sphere_radius=n_sphere_radius, n_sphere_center=n_sphere_center)

        if is_outside and len(outside_points) < n_points//2:
            outside_points.append(random_point)

        elif not is_outside and len(inside_points) < n_points//2:
            inside_points.append(random_point)

    data = pd.DataFrame(outside_points+inside_points)
    data.columns = [f'x_{i}' for i in range(n_dimension)]

    is_outside = data.apply(
                    partial(
                        outside_n_sphere,
                        n_sphere_radius=n_sphere_radius,
                        n_sphere_center=n_sphere_center
                        ),
                    axis=1
                    )

    data['is_outside'] = is_outside.astype(float)

    return data


def generate_dataset(n_points, bounds, n_dimension, n_sphere_radius, n_sphere_center):
    ''' Create a dataframe to be used for training classification of points
    inside an n-sphere

    Arguments
    ---------
    n_points - number of datapoints
    bounds - definition of bounds of datapoints (-bound,+bound)
    n_dimension - number of dimensions of the n-sphere and points
    n_sphere_radius - radius of the n_sphere
    n_sphere_center - central point of the n_sphere
    '''

    assert bounds > 0, 'Bounds should be a positive non zero number'
    assert len(n_sphere_center) == n_dimension, 'Number of dimentions must match'
    assert isinstance(n_sphere_center, np.ndarray)

    points_table = ((np.random.rand(n_points,n_dimension)*2) - 1) * bounds

    data = pd.DataFrame(points_table)
    data.columns = [f'x_{i}' for i in range(n_dimension)]

    is_outside = data.apply(
                    partial(
                        outside_n_sphere,
                        n_sphere_radius=n_sphere_radius,
                        n_sphere_center=n_sphere_center
                        ),
                    axis=1
                    )

    data['is_outside'] = is_outside.astype(float)

    return data

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]