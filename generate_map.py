import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def generate_structured_map(size=400):
    map = np.zeros((size, size), dtype=np.uint8)
    
    def add_rectangle(x, y, width, height, angle=0):
        nonlocal map
        rect = np.zeros((size, size), dtype=np.uint8)
        box_points = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ])
        if angle != 0:
            center = (x + width / 2, y + height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            box_points = cv2.transform(np.array([box_points]), rotation_matrix[:2, :])[0]
            box_points = np.round(box_points).astype(int)
        
        box_points = np.clip(box_points, 0, size - 1)  # Ensure points are within bounds
        cv2.fillConvexPoly(rect, box_points, 1)
        map = np.maximum(map, rect)
    
    def add_circle(x, y, radius):
        nonlocal map
        Y, X = np.ogrid[:size, :size]
        dist = (X - x)**2 + (Y - y)**2
        map[dist <= radius**2] = 1
    
    def add_ellipse(x, y, r1, r2, angle=0):
        nonlocal map
        Y, X = np.ogrid[:size, :size]
        ellipse_eq = ((X - x)**2 / r1**2) + ((Y - y)**2 / r2**2)
        ellipse = (ellipse_eq <= 1).astype(np.uint8)
        
        if angle != 0:
            center = (x, y)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_ellipse = cv2.warpAffine(ellipse, rotation_matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            map = np.maximum(map, rotated_ellipse)
        else:
            map = np.maximum(map, ellipse)

    def add_large_areas():
        for _ in range(random.randint(1, 4)):
            area_size = random.randint(size // 10, size // 5)
            x_start = random.randint(0, size - area_size)
            y_start = random.randint(0, size - area_size)
            angle = random.randint(0, 360)
            add_rectangle(x_start, y_start, area_size, area_size, angle)

    def add_corridors():
        for _ in range(random.randint(6, 12)):
            corridor_width = random.randint(5, 10)
            start_x = random.randint(0, size - corridor_width)
            start_y = random.randint(0, size - corridor_width)
            length = size
            angle = random.randint(0, 360)
            if random.choice([True, False]):
                add_rectangle(start_x, start_y, corridor_width, length, angle)
            else:
                add_rectangle(start_x, start_y, length, corridor_width, angle)

    def add_rooms():
        for _ in range(random.randint(15, 30)):
            room_size = random.randint(10, 20)
            room_x = random.randint(0, size - room_size)
            room_y = random.randint(0, size - room_size)
            angle = random.randint(0, 360)
            add_rectangle(room_x, room_y, room_size, room_size, angle)
    
    def add_shapes():
        for _ in range(random.randint(5, 10)):
            shape_type = random.choice(['circle', 'ellipse'])
            x = random.randint(10, size - 10)
            y = random.randint(10, size - 10)
            if shape_type == 'circle':
                radius = random.randint(5, 15)
                add_circle(x, y, radius)
            elif shape_type == 'ellipse':
                r1 = random.randint(10, 20)
                r2 = random.randint(5, 15)
                angle = random.randint(0, 360)
                add_ellipse(x, y, r1, r2, angle)

    add_large_areas()
    add_corridors()
    add_rooms()
    add_shapes()

    return map

def rotate_map(map, angle):
    """
    Rotaciona o mapa completo dado em um ângulo aleatório.
    
    Parâmetros:
    - map: O mapa a ser rotacionado (array 2D).
    - angle: Ângulo de rotação.
    
    Retorna:
    - O mapa rotacionado.
    """
    (h, w) = map.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_map = cv2.warpAffine(map, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_map

def get_random_patch_coordinates(map_size, patch_size):
        x_start = random.randint(0, map_size - patch_size[0])
        y_start = random.randint(0, map_size - patch_size[1])
        return x_start, y_start

def extract_patch(map, patch_size):
    angle = random.randint(0, 360)
    rotated_map = rotate_map(map, angle)
    x, y = get_random_patch_coordinates(map.shape[0], patch_size)
    patch = rotated_map[x:x+patch_size[0], y:y+patch_size[1]]
    return patch

def save_patch(patch, index):
    cv2.imwrite(f'maps/map{index}.png', patch * 255)

def extract_and_save_patches(map):
    patch1 = extract_patch(map, patch_size = (random.randint(200,400), random.randint(200,400)))
    patch2 = extract_patch(map, patch_size = (random.randint(200,400), random.randint(200,400)))
    save_patch(patch1, 1)
    save_patch(patch2, 2)
    plot_maps(map, patch1, patch2)

def plot_maps(fullmap, patch1, patch2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Full map")
    plt.imshow(fullmap, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("Patch 1")
    plt.imshow(patch1, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Patch 2")
    plt.imshow(patch2, cmap='gray')
    
    plt.show()


if __name__ == "__main__":
    map = generate_structured_map(size=550)
    extract_and_save_patches(map)
