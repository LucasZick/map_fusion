import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class Position:
    x: float
    y: float

def load_maps(folder='maps'):
    image1 = cv2.imread(f'{folder}/map1.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(f'{folder}/map2.png', cv2.IMREAD_GRAYSCALE)
    with open(f'{folder}/pos.json', 'r') as file:
        pos = json.load(file)
    pos1 = {'position': (pos['1']['position'][0], pos['1']['position'][1]), 'angle': pos['1']['angle']}
    pos2 = {'position': (pos['2']['position'][0], pos['2']['position'][1]), 'angle': pos['2']['angle']} 
    return image1, image2, pos1, pos2

def detect_and_describe_features(image):
    """
    Detects and describes features using ORB.
    
    Parameters:
    - image: Grayscale image as a 2D numpy array.
    
    Returns:
    - keypoints: Detected keypoints.
    - descriptors: Descriptors of the keypoints.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """
    Finds matches between descriptors using a matcher.
    
    Parameters:
    - descriptors1: Descriptors from the first map.
    - descriptors2: Descriptors from the second map.
    
    Returns:
    - matches: List of found matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return matches

def find_transformation(keypoints1, keypoints2, matches):
    """
    Finds the affine transformation (translation and rotation) between two maps based on matches.
    
    Parameters:
    - keypoints1: Keypoints from the first map.
    - keypoints2: Keypoints from the second map.
    - matches: Found matches.
    
    Returns:
    - M: Affine transformation matrix (2x3).
    - dx, dy: Translation needed to align map2 with map1.
    - angle: Rotation angle needed.
    """

    if len(matches) >= 25:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the affine transformation matrix using RANSAC
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is not None and M.size == 6:
            # Extract translation
            dx, dy = M[0, 2], M[1, 2]

            # Calculate rotation angle from the transformation matrix
            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

            return M, dx, dy, angle
        else:
            print("Affine transformation matrix not found or invalid format.")
    else:
        print("Insufficient number of matches to calculate transformation.")

    # Return None values if transformation cannot be calculated
    return None, None, None, None

def overlay_maps(image1, image2, M, error_threshold=10):
    """
    Overlays two maps using the affine transformation matrix and calculates the overlay precision.
    
    Parameters:
    - image1: First map as a 2D numpy array.
    - image2: Second map as a 2D numpy array.
    - M: Affine transformation matrix (2x3) aligning map2 to map1.
    - error_threshold: Maximum acceptable mean absolute error for overlay precision.
    
    Returns:
    - combined_map: Combined image of the two maps.
    - error: Mean absolute error between overlapping regions.
    - map1_x_start, map1_y_start: Start positions of map1 in the combined_map.
    - map2_x_start, map2_y_start: Start positions of map2 in the combined_map.
    - M_translated: Translated affine transformation matrix.
    """
    # Apply the affine transformation to map2 to align it with map1
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # Define the corners of map2
    corners_map2 = np.array([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform the corners using the affine matrix M
    transformed_corners = cv2.transform(corners_map2, M)

    # Define the corners of map1
    corners_map1 = np.array([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Combine all corners to determine the size of the combined map
    all_corners = np.vstack((
        transformed_corners,
        corners_map1
    ))

    # Calculate the bounding rectangle of all corners
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    # Update the transformation matrix to include the translation
    M_translated = M.copy()
    M_translated[:, 2] += translation

    # Calculate the size of the combined map
    combined_width = xmax - xmin
    combined_height = ymax - ymin

    # Create the combined map initialized with zeros
    combined_map = np.zeros((combined_height, combined_width), dtype=np.uint8)

    # Warp map2 using the translated affine matrix
    transformed_map2 = cv2.warpAffine(image2, M_translated, (combined_width, combined_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    combined_map = np.maximum(combined_map, transformed_map2)

    # Place map1 onto the combined map
    combined_map[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]] = np.maximum(
        combined_map[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]],
        image1
    )

    # Calculate the overlay precision
    # Warp map1 to the combined map space
    warped_map1 = np.zeros_like(combined_map)
    warped_map1[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]] = image1

    # Logical AND to find overlapping regions
    overlap = np.logical_and(transformed_map2 > 0, warped_map1 > 0)

    if np.any(overlap):
        overlap1 = warped_map1[overlap]
        overlap2 = transformed_map2[overlap]
        error = np.mean(np.abs(overlap1 - overlap2))
        if error > error_threshold:
            print(f"Warning: Map overlay is not precise! Calculated error: {error:.2f} (threshold: {error_threshold})")
    else:
        error = float('inf')  # Set error to infinity if no valid overlap
        print("Warning: No valid overlap between the maps.")

    # Calculate the start positions of map1 and map2 in the combined map
    map1_x_start, map1_y_start = translation
    map2_x_start, map2_y_start = 0, 0  # map2 has been transformed with M_translated

    return combined_map, error, map1_x_start, map1_y_start, map2_x_start, map2_y_start, M_translated

def apply_affine_transform(x, y, M):
    """
    Applies an affine transformation to a point (x, y) using matrix M.
    
    Parameters:
    - x, y: Coordinates of the point.
    - M: Affine transformation matrix (2x3).
    
    Returns:
    - x_transformed, y_transformed: Transformed coordinates.
    """
    point = np.array([x, y, 1], dtype=np.float32).reshape(3, 1)
    transformed = M @ point
    return transformed[0, 0], transformed[1, 0]

def merge_positions(pos1, pos2, M_translated, map1_x_start, map1_y_start):
    """
    Combines the robot positions from both maps into the combined map.
    
    Parameters:
    - pos1: Dictionary with 'position' (tuple) and 'angle' for the robot in map1.
    - pos2: Dictionary with 'position' (tuple) and 'angle' for the robot in map2.
    - M_translated: Translated affine transformation matrix (2x3) applied to map2.
    - map1_x_start: Translation in x for map1 within the combined map.
    - map1_y_start: Translation in y for map1 within the combined map.
    
    Returns:
    - newpos1: Position of Robot 1 in the combined map.
    - newpos2: Position of Robot 2 in the combined map.
    """
    # Position of Robot 1 in map1
    pos1_x, pos1_y = pos1['position']
    
    # Adjust Robot 1's position based on the translation
    newpos1_x = pos1_x + map1_x_start
    newpos1_y = pos1_y + map1_y_start
    newpos1 = Position(x=newpos1_x, y=newpos1_y)
    
    # Position of Robot 2 in map2
    pos2_x, pos2_y = pos2['position']
    
    # Apply the affine transformation to Robot 2's position
    newpos2_x, newpos2_y = apply_affine_transform(pos2_x, pos2_y, M_translated)
    newpos2 = Position(x=newpos2_x, y=newpos2_y)
    
    return newpos1, newpos2

def plot_results(map1, map2, combined_map, pos1, pos2, newpos1, newpos2):
    """
    Plots the individual maps and the combined map with robot positions.
    
    Parameters:
    - map1: First map as a 2D numpy array.
    - map2: Second map as a 2D numpy array.
    - combined_map: Combined map as a 2D numpy array.
    - pos1: Original position of Robot 1 in map1.
    - pos2: Original position of Robot 2 in map2.
    - newpos1: Position of Robot 1 in the combined map.
    - newpos2: Position of Robot 2 in the combined map.
    """
    # Visualize the maps
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Map 1")
    plt.scatter(pos1['position'][0], pos1['position'][1], c='r', label='Robot 1')
    plt.imshow(map1, cmap='gray')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.title("Map 2")
    plt.scatter(pos2['position'][0], pos2['position'][1], c='g', label='Robot 2')
    plt.imshow(map2, cmap='gray')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.title("Combined Map")
    plt.scatter(newpos1.x, newpos1.y, c='r', label='Robot 1')
    plt.scatter(newpos2.x, newpos2.y, c='g', label='Robot 2')
    plt.imshow(combined_map, cmap='gray')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(image1, cmap='gray')
    plt.scatter(pos1['position'][0], pos1['position'][1], c='r', label='Robot 1')
    plt.savefig(f"results_robot_positions/{1}/patch1.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(image2, cmap='gray')
    plt.scatter(pos2['position'][0], pos2['position'][1], c='g', label='Robot 2')
    plt.savefig(f"results_robot_positions/{1}/patch2.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(combined_map, cmap='gray')
    plt.scatter(newpos1.x, newpos1.y, c='r', label='Robot 1')
    plt.scatter(newpos2.x, newpos2.y, c='g', label='Robot 2')
    plt.savefig(f"results_robot_positions/{1}/combined_map.png", bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Load the images as 2D arrays
    image1, image2, pos1, pos2 = load_maps()
    
    # Detect and describe features
    keypoints1, descriptors1 = detect_and_describe_features(image1)
    keypoints2, descriptors2 = detect_and_describe_features(image2)
    
    # Find matches
    matches = match_features(descriptors1, descriptors2)
    
    # Find the transformation
    M, dx, dy, angle = find_transformation(keypoints1, keypoints2, matches)
    
    if M is not None:
        print(f"The affine transformation matrix M:\n{M}")
        print(f"The required rotation angle is: {angle:.2f} degrees")
        
        # Combine the maps with the transformation
        combined_map, error, map1_x_start, map1_y_start, map2_x_start, map2_y_start, M_translated = overlay_maps(
            image1, image2, M, error_threshold=20)
        
        print('Mean absolute error:', error)
        
        if error < 20:
            # Merge positions
            newpos1, newpos2 = merge_positions(
                pos1, pos2, M_translated, map1_x_start, map1_y_start)
            
            # Display the new positions in the console
            print(f"New position of Robot 1 in the combined map: ({newpos1.x:.2f}, {newpos1.y:.2f})")
            print(f"New position of Robot 2 in the combined map: ({newpos2.x:.2f}, {newpos2.y:.2f})")
            
            # Plot the results
            plot_results(image1, image2, combined_map, pos1, pos2, newpos1, newpos2)
        else:
            print("Overlay error exceeds the threshold. Position merging will not be performed.")
    else:
        print("Transformation could not be calculated.")
