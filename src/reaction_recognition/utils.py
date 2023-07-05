from scipy.signal import savgol_filter
import numpy as np
import csv

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

def export_to_csv(file,mode,row):
    with open(file, mode=mode, newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

def get_features(source,ids: list[int], attribute: str = None,function = None,other_arguments: dict[str,any] = {}) -> list:
    features = []
    if function is None:
        for group in ids:
            if attribute is None:
                features += [[source[id] for id in group]]
            else:
                features += [[getattr(source[id],attribute) for id in group]]
    else:
        for group in ids:
            if attribute is None:
                features +=[function([source[id] for id in group],**other_arguments)]
            else:
                features +=[function([getattr(source[id],attribute) for id in group],**other_arguments)]
    return features

'''def get_landmarks_y_averages(source,ids: list[int]) -> list:
    averages = []
    for group in ids:
        averages += [np.average([source[id].y for id in ids[group]],weights=[source[id].visibility for id in ids[group]])]
    return averages'''