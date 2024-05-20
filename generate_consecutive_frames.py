import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

def get_frames(height, width, random_color=True):
    frame1 = np.zeros((height, width), dtype=np.uint8)
    frame2 = np.zeros((height, width), dtype=np.uint8)

    shape_type = random.choice(['rectangle', 'circle'])
    size = random.randint(20, min(height, width) // 4)

    if shape_type == 'rectangle':
        x1 = random.randint(0, max(1, width - size))
        y1 = random.randint(0, max(1, height - size))
        x2, y2 = x1 + size, y1 + size

        if random_color:
            for i in range(x1, x2):
                for j in range(y1, y2):
                    gray_value = random.randint(0, 255)
                    frame1[j, i] = gray_value
                    if (j + 10 < height) and (i + 10 < width):  # sınırları kontrol et
                        frame2[j + 10, i + 10] = gray_value
        else:
            gray_value = random.randint(0, 255)
            cv2.rectangle(frame1, (x1, y1), (x2, y2), gray_value, -1)
            cv2.rectangle(frame2, (x1 + 10, y1 + 10), (x2 + 10, y2 + 10), gray_value, -1)

    elif shape_type == 'circle':
        radius = size // 2
        x1 = random.randint(radius, max(1, width - radius))
        y1 = random.randint(radius, max(1, height - radius))

        if random_color:
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    if i ** 2 + j ** 2 <= radius ** 2:
                        if 0 <= x1 + i < width and 0 <= y1 + j < height:
                            gray_value = random.randint(0, 255)
                            frame1[y1 + j, x1 + i] = gray_value
                            if 0 <= x1 + i + 10 < width and 0 <= y1 + j + 10 < height:  # sınırları kontrol et
                                frame2[y1 + j + 10, x1 + i + 10] = gray_value
        else:
            gray_value = random.randint(0, 255)
            cv2.circle(frame1, (x1, y1), radius, gray_value, -1)
            cv2.circle(frame2, (x1 + 10, y1 + 10), radius, gray_value, -1)

    return frame1, frame2


if __name__ == '__main__':
    frame1, frame2 = get_frames(120, 180, False)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Frame 1')
    plt.imshow(frame1, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Frame 2')
    plt.imshow(frame2, cmap='gray')

    plt.show()
