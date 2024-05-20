import cv2
import matplotlib.pyplot as plt
from filters import get_filter
from generate_consecutive_frames import get_frames
from utils import *


def plot_optical_flow(I, I_2, u, v, step=1, scale=1):
    plt.figure(figsize=(12, 6))

    # İlk görüntü
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Image 1')

    # İkinci görüntü ve optik akış vektörleri
    plt.subplot(1, 2, 2)
    plt.imshow(I_2, cmap='gray')

    # Quiver fonksiyonunu kullanarak optik akış vektörlerini çiz
    X, Y = np.meshgrid(np.arange(0, u.shape[1]), np.arange(0, u.shape[0]))
    plt.quiver(X[::step, ::step], Y[::step, ::step], v[::step, ::step], u[::step, ::step], color='red', angles='xy',
               scale_units='xy', scale=scale)
    plt.title('Optical Flow')

    plt.show()


def normalize_flow(u, v, min_val=0, max_val=10):
    u_norm = np.zeros_like(u)
    v_norm = np.zeros_like(v)

    u_nonzero = u != 0
    v_nonzero = v != 0

    if np.any(u_nonzero):
        u_norm[u_nonzero] = (u[u_nonzero] - u[u_nonzero].min()) / (u[u_nonzero].max() - u[u_nonzero].min()) * (
                    max_val - min_val) + min_val

    if np.any(v_nonzero):
        v_norm[v_nonzero] = (v[v_nonzero] - v[v_nonzero].min()) / (v[v_nonzero].max() - v[v_nonzero].min()) * (
                    max_val - min_val) + min_val
    return u_norm, v_norm


if __name__ == "__main__":
    I_1, I_2 = get_frames(360, 480, False)

    flt = get_filter("scharr")
    gradient_matrix = calculate_gradient(I_1, I_2, flt)

    I_x = gradient_matrix["I_x"]
    I_y = gradient_matrix["I_y"]
    I_t = gradient_matrix["I_t"]

    u, v = calculate_optical_flow(I_x, I_y, I_t, 5)
    u, v = normalize_flow(u, v)

    plot_optical_flow(I_1, I_2, u, v, step=4, scale=2)
