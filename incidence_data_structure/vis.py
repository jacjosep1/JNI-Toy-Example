import matplotlib.pyplot as plt
import numpy as np

def matrix_heatmap(tensor : np.ndarray, title=""):
    tensor = tensor.T
    n, m = tensor.shape

    y_idx, x_idx = np.nonzero(tensor) 
    values = tensor[y_idx, x_idx]

    heatmap, _, _ = np.histogram2d(
        x_idx, y_idx,
        bins=[m, n], 
        weights=values
    )
    plt.imshow(heatmap, origin='upper', cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

