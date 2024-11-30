import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
np.set_printoptions(threshold=sys.maxsize)

def detect_edges(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize for faster processing (optional)
    img = cv2.resize(img, (300, 300))
    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    return edges

def dfs(edges):
    m, n = edges.shape
    # find starting point
    r = 0
    c = 0
    found = False
    for i in range(m):
        for j in range(n):
            if edges[i][j] != 0:
                r = i
                c = j
                found = True
                break
        if found:
            break
    # find path
    path = []
    seen = [[False for i in range(n)] for j in range(m)]
    seen[r][c] = True
    path.append([r, c])
    dir = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, -1], [-1, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2], [1, -2], [0, -2], [-1, -2], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, 2]]
    while True:
        found = False
        for d in dir:
            nr = r + d[0]
            nc = c + d[1]
            if nr >= 0 and nr < m and nc >= 0 and nc < n and not seen[nr][nc] and edges[nr][nc] != 0:
                r = nr
                c = nc
                found = True
                break
        if found:
            path.append([r, c])
            seen[r][c] = True
            continue
        break
    return path

def lstsq(A, b):
    # find lesat squares solution
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    return [params[1], params[0], np.sqrt(params[2] + params[1]**2 + params[0]**2)]


def draw_circles(edges, dfs_path, num_circles=50):
    # Create a blank canvas
    height, width = edges.shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    # draw circles
    for i in range(0, num_circles):
        points = []
        num_points = 50
        for j in range(num_points):
            points.append(dfs_path[(i * len(dfs_path)//num_circles + j) % len(dfs_path)])

        points = np.array(points)
        A = []
        b = []
        for j in range(num_points):
            A.append([2*points[j][0], 2*points[j][1], 1])
            b.append(points[j][0]**2 + points[j][1]**2)
        A = np.array(A)
        b = np.array(b)
        h, k, r = lstsq(A, b)

        radius = int(r)
        center = (int(h), int(k))
        color = 255
        thickness = 1
        
        cv2.circle(canvas, center, radius, color, thickness)
    
    return canvas

def main(image_path):
    # Detect edges
    edges = detect_edges(image_path)
    # find path to trace shape
    dfs_path = dfs(edges)
    # Draw circles on the canvas
    circle_image = draw_circles(edges, dfs_path)
    # edges[edges == 255] = 1
    # np.savetxt('edges.txt', edges, fmt='%d')
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Circles Approximation")
    plt.imshow(circle_image, cmap="gray")
    plt.axis("off")
    
    plt.show()

# Path to the uploaded image
uploaded_image_path = "bat.jpeg"
main(uploaded_image_path)
