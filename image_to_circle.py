import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import deque
np.set_printoptions(threshold=sys.maxsize)

def detect_edges(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize for faster processing (optional)
    img = cv2.resize(img, (1500, 1500))
    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    return edges

def bfs(edges):
    # bfs in 2 directions from random starting points to get paths in order
    m, n = edges.shape
    r, c = 0, 0
    paths = []
    seen = [[False for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            if edges[i][j] != 0 and not seen[i][j]:
                path = deque([[i, j]])
                lq = deque([])
                rq = deque([])
                dir = [[-1, 0], [-1, -1], [0, -1], [1, -1]]
                for d in dir:
                    nr = i + d[0]
                    nc = j + d[1]
                    if not seen[nr][nc] and edges[nr][nc] != 0:
                        seen[nr][nc] = True
                        lq.append([nr, nc])
                        path.appendleft([nr, nc])
                dir = [[-1, 1], [0, 1], [1, 1], [1, 0]]
                for d in dir:
                    nr = i + d[0]
                    nc = j + d[1]
                    if not seen[nr][nc] and edges[nr][nc] != 0:
                        seen[nr][nc] = True
                        rq.append([nr, nc])
                        path.append([nr, nc])
                dir = [[-1, 0], [-1, -1], [0, -1], [1, -1], [-1, 1], [0, 1], [1, 1], [1, 0]]
                while len(lq) > 0 or len(rq) > 0:
                    size = len(lq)
                    for _ in range(size):
                        r, c = lq.popleft()
                        for d in dir:
                            nr = r + d[0]
                            nc = c + d[1]
                            if not seen[nr][nc] and edges[nr][nc] != 0:
                                seen[nr][nc] = True
                                lq.append([nr, nc])
                                path.appendleft([nr, nc])
                    size = len(rq)
                    for _ in range(size):
                        r, c = rq.popleft()
                        for d in dir:
                            nr = r + d[0]
                            nc = c + d[1]
                            if not seen[nr][nc] and edges[nr][nc] != 0:
                                seen[nr][nc] = True
                                rq.append([nr, nc])
                                path.append([nr, nc])
                path = list(path)
                paths.append(path)
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

def draw_circles(shape, paths):
    # Create a blank canvas
    height, width = shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    # draw circles
    weird_points = [[]]
    for path in paths:
        idx = 0
        while idx < len(path):
            # binary search on segment
            l, r = 20, len(path)-1
            while l <= r:
                m = (l+r)//2
                # find lstsq circle
                points = []
                for i in range(0, m):
                    points.append(path[(idx + i) % len(path)])
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
                if RMSE > 0.41:
                    r = m - 1
                else:
                    l = m + 1
            # final segment length = l
            if l < 30: # check for points that are hard to fit (usually those that are almost straight line)
                if len(weird_points[-1]) > 60:
                    weird_points.append([])
                for i in range(0, l):
                    weird_points[-1].append(path[(idx + i) % len(path)])
            else: # draw circles
                if len(weird_points[-1]) != 0:
                    weird_points.append([])
                points = []
                for i in range(0, l):
                    points.append(path[(idx + i) % len(path)])
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
    paths = bfs(edges)
    # Draw circles on the canvas
    circle_image = draw_circles(edges.shape, paths)
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
