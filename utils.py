import numpy as np
from scipy.ndimage import convolve


# Optical flow operations
def calculate_gradient(I_1, I_2, edge_filter):
    gradient_dict = {}
    edge_x, edge_y = edge_filter
    I_x = convolve(I_1, edge_x)
    I_y = convolve(I_1, edge_y)
    I_t = I_2 - I_1

    gradient_dict["I_x"] = I_x
    gradient_dict["I_y"] = I_y
    gradient_dict["I_t"] = I_t
    return gradient_dict


def calc_norm_and_linear_equation(I_x, I_y, I_t, threshold=1e-4):
    A = np.array([
        [np.sum(I_x ** 2), np.sum(I_x * I_y)],
        [np.sum(I_x * I_y), np.sum(I_y ** 2)]
    ])
    b = np.array([
        -np.sum(I_x * I_t),
        -np.sum(I_y * I_t)
    ])
    if np.linalg.det(A) > threshold:
        return np.linalg.solve(A, b)
    else:
        return 0, 0


def calculate_optical_flow(I_x, I_y, I_t, kernel_size):
    u = np.zeros(I_x.shape)
    v = np.zeros(I_x.shape)

    half_kernel = kernel_size // 2
    for i in range(half_kernel, I_x.shape[0] - half_kernel):
        for j in range(half_kernel, I_x.shape[1] - half_kernel):
            I_x_window = I_x[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]
            I_y_window = I_y[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]
            I_t_window = I_t[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]

            uv = calc_norm_and_linear_equation(I_x_window, I_y_window, I_t_window)
            u[i, j] = uv[0]
            v[i, j] = uv[1]

    return u, v
