import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate

def detect_and_describe_features(image):
    """
    Detecta e descreve características usando ORB.
    
    Parâmetros:
    - image: Imagem em escala de cinza como array numpy 2D.
    
    Retorna:
    - keypoints: Pontos-chave detectados.
    - descriptors: Descritores dos pontos-chave.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """
    Encontra correspondências entre descritores usando um matcher.
    
    Parâmetros:
    - descriptors1: Descritores do primeiro mapa.
    - descriptors2: Descritores do segundo mapa.
    
    Retorna:
    - matches: Lista de correspondências encontradas.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return matches

def find_transformation(keypoints1, keypoints2, matches):
    """
    Encontra a transformação (translação e rotação) entre dois mapas com base nas correspondências.
    
    Parâmetros:
    - keypoints1: Pontos-chave do primeiro mapa.
    - keypoints2: Pontos-chave do segundo mapa.
    - matches: Correspondências encontradas.
    
    Retorna:
    - dx, dy: Translação necessária para alinhar o map1 com o map2.
    - angle: Ângulo de rotação necessário.
    """

    if len(matches) >= 25:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Encontra a matriz de transformação afim (incluindo rotação) usando RANSAC
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is not None and M.size == 6:
            # Verifica se a matriz tem o formato esperado
            dx, dy = M[0, 2], M[1, 2]

            # Calcula o ângulo de rotação a partir da matriz de transformação
            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

            return dx, dy, angle
        else:
            print("Matriz de transformação não encontrada ou com formato inválido.")
    else:
        print("Número insuficiente de correspondências para calcular a transformação.")
    
    # Retorna valores default se a transformação não puder ser calculada
    return None, None, None

def rotate_image(image, angle):
    """
    Rotaciona a imagem com base no ângulo fornecido.
    
    Parâmetros:
    - image: Imagem como array 2D (numpy).
    - angle: Ângulo de rotação em graus.
    
    Retorna:
    - Imagem rotacionada.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_image

def overlay_maps(image1, image2, dx, dy, angle, error_threshold=10):
    """
    Sobrepõe dois mapas com a translação e rotação encontrada e calcula a precisão da sobreposição.
    
    Parâmetros:
    - image1: Primeiro mapa como array 2D (numpy).
    - image2: Segundo mapa como array 2D (numpy).
    - dx: Translação no eixo x.
    - dy: Translação no eixo y.
    - angle: Ângulo de rotação do segundo mapa em relação ao primeiro.
    - error_threshold: Limite máximo aceitável para o erro médio absoluto entre as áreas sobrepostas.
    
    Retorna:
    - combined_map: Uma imagem combinada dos dois mapas.
    - error: Erro médio absoluto entre as áreas sobrepostas.
    """
    # Rotaciona o segundo mapa
    (h2, w2) = image2.shape[:2]
    center = (w2 // 2, h2 // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Aplica rotação ao segundo mapa
    rotated_image2 = cv2.warpAffine(image2, rotation_matrix, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
   # Calcula o tamanho necessário para a largura da imagem combinada
    combined_width = max(
        image1.shape[1] + max(int(dx), 0),  # Se dx for positivo, adiciona ao tamanho da primeira imagem
        rotated_image2.shape[1] + max(-int(dx), 0)  # Se dx for negativo, ajusta para o segundo mapa
    )

    # Calcula o tamanho necessário para a altura da imagem combinada
    combined_height = max(
        image1.shape[0] + max(int(dy), 0),  # Se dy for positivo, adiciona ao tamanho da primeira imagem
        rotated_image2.shape[0] + max(-int(dy), 0)  # Se dy for negativo, ajusta para o segundo mapa
    )
    
    # Cria a imagem combinada inicializada com zeros
    combined_map = np.zeros((combined_height, combined_width), dtype=np.uint8)
    
    # Define posições de início para sobreposição dos mapas
    map1_x_start = max(int(dx), 0)
    map1_y_start = max(int(dy), 0)
    map2_x_start = max(-int(dx), 0)
    map2_y_start = max(-int(dy), 0)
    
    # Define os limites para colocar rotated_image2 na combined_map
    map2_end_y = min(map2_y_start + rotated_image2.shape[0], combined_map.shape[0])
    map2_end_x = min(map2_x_start + rotated_image2.shape[1], combined_map.shape[1])
    combined_map[map2_y_start:map2_end_y, map2_x_start:map2_end_x] = rotated_image2[:map2_end_y - map2_y_start, :map2_end_x - map2_x_start]
    
    # Define os limites para colocar image1 na combined_map
    map1_end_y = min(map1_y_start + image1.shape[0], combined_map.shape[0])
    map1_end_x = min(map1_x_start + image1.shape[1], combined_map.shape[1])
    
    # Sobrepõe image1 na combined_map usando o máximo dos pixels
    combined_map[map1_y_start:map1_end_y, map1_x_start:map1_end_x] = np.maximum(
        combined_map[map1_y_start:map1_end_y, map1_x_start:map1_end_x], 
        image1[:map1_end_y - map1_y_start, :map1_end_x - map1_x_start])
    
    # Calcular a precisão da sobreposição
    overlap_start_x = max(map1_x_start, map2_x_start)
    overlap_start_y = max(map1_y_start, map2_y_start)
    overlap_end_x = min(map1_x_start + image1.shape[1], map2_x_start + rotated_image2.shape[1])
    overlap_end_y = min(map1_y_start + image1.shape[0], map2_y_start + rotated_image2.shape[0])
    
    if overlap_end_x > overlap_start_x and overlap_end_y > overlap_start_y:
        # Define a região de sobreposição
        overlap1 = image1[(overlap_start_y - map1_y_start):(overlap_end_y - map1_y_start), 
                          (overlap_start_x - map1_x_start):(overlap_end_x - map1_x_start)]
        overlap2 = rotated_image2[(overlap_start_y - map2_y_start):(overlap_end_y - map2_y_start), 
                                  (overlap_start_x - map2_x_start):(overlap_end_x - map2_x_start)]
        
        # Calcular o erro médio absoluto entre as regiões de sobreposição
        error = np.mean(np.abs(overlap1 - overlap2))
        
        # Verifica se o erro ultrapassa o limite
        if error > error_threshold:
            print(f"Aviso: A sobreposição dos mapas não é precisa! Erro calculado: {error:.2f} (limite: {error_threshold})")
    else:
        error = float('inf')  # Define o erro como infinito se não houver sobreposição válida
        print("Aviso: Não há sobreposição entre os mapas.")
    
    return combined_map, error

def read_robot_positions(file_path):
    with open(file_path, 'r') as file:
        pos = json.load(file)
    pos1 = {'position': (pos['1']['position'][0], pos['1']['position'][1]), 'angle': pos['1']['angle']}
    pos2 = {'position': (pos['2']['position'][0], pos['2']['position'][1]), 'angle': pos['2']['angle']} 
    return pos1, pos2

def get_new_robot_positions(pos1, pos2, dx, dy, angle, map2_shape):

    new_pos1_x = pos1['position'][0]
    new_pos1_y = pos1['position'][1]

    vertices = np.array([
        [0, 0],
        [map2_shape[1], 0],
        [map2_shape[1], map2_shape[0]],
        [0, map2_shape[0]],
        [0, 0]  # Fechar o quadrado
    ])
    
    # Criar o polígono da área original
    original_polygon = Polygon(vertices)

    # Transformar a área
    transformed_polygon = rotate(original_polygon, angle, origin='centroid')
    transformed_polygon = translate(transformed_polygon, xoff=dx, yoff=dy)

    # Transformar o ponto
    transformed_point = rotate(Point(pos2['position'][0], pos2['position'][1]), angle, origin=original_polygon.centroid)
    transformed_point = translate(transformed_point, xoff=dx, yoff=dy)

    return new_pos1_x, new_pos1_y, transformed_point.x, transformed_point.y

# Exemplo de uso
if __name__ == "__main__":
    # Carregar as imagens como arrays 2D
    image1 = cv2.imread('maps/map1.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('maps/map2.png', cv2.IMREAD_GRAYSCALE)

    pos1, pos2 = read_robot_positions('maps/pos.json')

    # Detectar e descrever características
    keypoints1, descriptors1 = detect_and_describe_features(image1)
    keypoints2, descriptors2 = detect_and_describe_features(image2)

    # Encontrar correspondências
    matches = match_features(descriptors1, descriptors2)

    # Encontrar a transformação
    dx, dy, angle = find_transformation(keypoints1, keypoints2, matches)
    if dx != None:
        print(f"O ângulo de rotação necessário é: {angle} graus")

        # Rotacionar o segundo mapa
        rotated_map = rotate_image(image2, angle)
        
        keypoints3, descriptors3 = detect_and_describe_features(rotated_map)
        # Encontrar correspondências
        matches = match_features(descriptors1, descriptors3)

        dx, dy, _ = find_transformation(keypoints1, keypoints3, matches)
        if dx != None:
            print(f"A translação necessária é: dx={dx}, dy={dy}")

            # Combinar os mapas com a translação e rotação
            combined_map, error = overlay_maps(image1, rotated_map, dx, dy, 0, error_threshold=20)

            if error < 20:
                print('error: ', error)

                new_pos1_x, new_pos1_y, new_pos2_x, new_pos2_y = get_new_robot_positions(pos1, pos2, -dx, dy, angle, map2_shape=image2.shape)

                shape1 = image1.shape
                shape2 = image2.shape
                shapecomb = combined_map.shape
                
                # Visualizar os mapas
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.title("Map 1")
                plt.scatter(pos1['position'][0], pos1['position'][1], c='r')
                plt.imshow(image1, cmap='gray', extent=[0, shape1[1], 0,  shape1[0]])
                
                plt.subplot(1, 3, 2)
                plt.title("Map 2")
                plt.scatter(pos2['position'][0], pos2['position'][1], c='g')
                plt.imshow(image2, cmap='gray', extent=[0, shape2[1], 0,  shape2[0]])

                min_extent_x = min(-dx,0)
                min_extent_y = shapecomb[1] + min(-dx,0) if shape1[0] > shape2[0] + dy else 0
                max_extent_x = min(dy,0)
                max_extent_y = shapecomb[0] + min(dy,0)

                plt.subplot(1, 3, 3)
                plt.title("Combined Map")
                plt.scatter(new_pos1_x, new_pos1_y, c='r')
                plt.scatter(new_pos2_x, new_pos2_y, c='g')
                print(new_pos2_y)
                plt.imshow(combined_map, cmap='gray', extent=[min_extent_x, min_extent_y, max_extent_x, max_extent_y])
                
                plt.show()