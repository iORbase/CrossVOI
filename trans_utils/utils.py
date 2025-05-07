import torch
import numpy as np

def get_matrix(gpu, matrix, list, Set):
    mats = []
    for name in Set:
        index = list.index(name)
        mats.append(matrix[index])

    mats = np.array(mats)
    mats = torch.FloatTensor(mats)
    return mats.to(gpu)
