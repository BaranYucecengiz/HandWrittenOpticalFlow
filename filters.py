import numpy as np

filters_dict = {
    "sobel": (np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]),
              np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
              ),
    "prewitt": (np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]]),
                np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])
                ),
    "scharr": (np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]]),
               np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])
               )
}


def get_filter(name):
    return filters_dict[name]

if __name__ == "__main__":
    print(get_filter("sobel")[0])