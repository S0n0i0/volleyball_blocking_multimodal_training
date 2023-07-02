from scipy.signal import savgol_filter
import numpy as np

def compute_acceleration(tensor: np.ndarray) -> np.ndarray: # (543, 133, 3)
    points, frames, dims = np.shape(tensor)
    new_data = []
    for point in range(points):
        new_dims = []
        for dim in range(dims):
            new_dims.append(savgol_filter(tensor[point, :, dim], 3, 2))
        new_data.append(new_dims)
    tensor = np.array(new_data)
    acceleration = np.diff(tensor, 2, axis=1)
    '''points, dims, frames = np.shape(tensor)
    new_data = []
    for point in range(points):
        new_dims = []
        for dim in range(dims):
            new_dims.append(savgol_filter(tensor[point, dim, :], 3, 2))
        new_data.append(new_dims)
    tensor = np.array(new_data)
    acceleration = np.diff(tensor, 2, axis=1)'''

    return acceleration
