import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

def detect_edges(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize for faster processing (optional)
    img = cv2.resize(img, (1500, 1500))
    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    return edges

def dfs_helper(edges, seen, r, c):
    m, n = edges.shape
    # find path
    path = []
    seen[r][c] = True
    path.append([r, c])
    # look distance 1, 2 blocks away
    dir = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, -1], [-1, 1]]
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
        # if len(path) < 10:
        #     return []
        break
    return path

def dfs(edges):
    m, n = edges.shape
    # find starting point
    r = 0
    c = 0
    paths = []
    seen = [[False for i in range(n)] for j in range(m)]
    found = False
    for i in range(m):
        for j in range(n):
            if edges[i][j] != 0 and not seen[i][j]:
                paths.append(dfs_helper(edges, seen, i, j))
    return paths

def lstsq(A, b):
    # find least squares solution
    try:
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        return [params[1], params[0], np.sqrt(params[2] + params[1] ** 2 + params[0] ** 2)]
    except:
        return [0, 0, 0]

def lstsq_error(h, k, r, points):
    y = points[:, 0]
    x = points[:, 1]
    # mean square error
    MSE = np.mean((np.sqrt((x-h) ** 2 + (y-k)**2) - r) ** 2)
    # root mean square error
    return np.sqrt(MSE)

def draw_circles(shape, dfs_paths):
    # Create a blank canvas
    height, width = shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    # draw circles
    weird_points = [[]]
    for dfs_path in dfs_paths:
        idx = 0
        while idx < len(dfs_path):
            # binary search on segment
            l, r = 3, len(dfs_path)-1
            while l <= r:
                m = (l+r)//2
                # find lstsq circle
                points = []
                for i in range(0, m):
                    points.append(dfs_path[(idx + i) % len(dfs_path)])
                points = np.array(points)
                A = []
                b = []
                for j in range(m):
                    A.append([2 * points[j][0], 2 * points[j][1], 1])
                    b.append(points[j][0] ** 2 + points[j][1] ** 2)
                A = np.array(A)
                b = np.array(b)
                h, k, radius = lstsq(A, b)
                RMSE = lstsq_error(h, k, radius, points)
                # check error
                if RMSE > 0.4:
                    r = m - 1
                else:
                    l = m + 1
            # final segment length = l
            if l < 10: # check for points that are hard to fit (usually those that are almost straight line)
                if len(weird_points[-1]) > 75:
                    weird_points.append([])
                for i in range(0, l):
                    weird_points[-1].append(dfs_path[(idx + i) % len(dfs_path)])
            else: # draw circles
                if len(weird_points[-1]) != 0:
                    weird_points.append([])
                points = []
                for i in range(0, l):
                    points.append(dfs_path[(idx + i) % len(dfs_path)])
                points = np.array(points)
                A = []
                b = []
                for j in range(l):
                    A.append([2 * points[j][0], 2 * points[j][1], 1])
                    b.append(points[j][0] ** 2 + points[j][1] ** 2)
                A = np.array(A)
                b = np.array(b)
                h, k, r = lstsq(A, b)
                cv2.circle(canvas, (int(h), int(k)), int(r), 255, 1)
            idx += l
    for points in weird_points: # redraw circles that fit weird points
        points = np.array(points)
        A = []
        b = []
        for j in range(len(points)):
            A.append([2 * points[j][0], 2 * points[j][1], 1])
            b.append(points[j][0] ** 2 + points[j][1] ** 2)
        A = np.array(A)
        b = np.array(b)
        h, k, r = lstsq(A, b)
        cv2.circle(canvas, (int(h), int(k)), int(r), 255, 1)

    return canvas

def main(image_path):
    # Detect edges
    edges = detect_edges(image_path)
    # find path to trace shape
    dfs_path = dfs(edges)
    # Draw circles on the canvas
    circle_image = draw_circles(edges.shape, dfs_path)
    edges[edges == 255] = 1
    np.savetxt('edges.txt', edges, fmt='%d')
    
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

# go through all images
for image in os.listdir("images"):
    image_path = os.path.join("images", image)
    main(image_path)
# go through select image
# image_path = r"images\bat.jpeg"
# main(image_path)
