import numpy as np
import cv2
import math
import pytesseract
import pandas as pd
import re
import xlsxwriter
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def pt_to_tuple(p):
    return (int(round(p[0])), int(round(p[1])))

def pt(x, y, dtype=np.float):
    """Create a point in 2D space at <x>, <y>"""
    return np.array((x, y), dtype=dtype)

def norm_angel(theta):
    pi2 = 2 * np.pi
    if theta >= pi2:
        m = math.floor(theta/pi2)
        if theta/pi2 - m > 0.99999:
            m += 1
        theta_norm = theta - m * pi2
    elif theta < 0:
        m = math.ceil(theta/pi2)
        if theta/pi2 - m < -0.99999:
            m -= 1
        theta_norm = abs(theta - m * pi2)
    else:
        theta_norm = theta
    return theta_norm

def generate_lines(lines):
    lines_hough = []
    for line in lines:
        rho, theta = line[0]
        theta_norm = norm_angel(theta)
        if abs((np.pi/2) - theta_norm) > (np.pi/4):
            line_dir = 'v'
        else:
            line_dir = 'h'
        lines_hough.append((rho, theta, theta_norm, line_dir))
    return lines_hough

def detectlines(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150, 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, round(0.4 * img.shape[1]))
    if lines is None:
        lines = []
    lines_hough = generate_lines(lines)
    return lines_hough

def project_lines(lines, img_w, img_h):
    if img_w <= 0:
        raise ValueError('image witdh must be > 0')
    if img_h <= 0:
        raise ValueError('image height must be > 0')

    lines_ab = []
    for i, (rho, theta) in enumerate(lines):
        # calculate intersections with canvas dimension minima/maxima
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_miny = rho / cos_theta if cos_theta != 0 else float("inf")  # x for a minimal y (y=0)
        y_minx = rho / sin_theta if sin_theta != 0 else float("inf")  # y for a minimal x (x=0)
        x_maxy = (rho - img_w * sin_theta) / cos_theta if cos_theta != 0 else float("inf")  # x for maximal y (y=img_h)
        y_maxx = (rho - img_h * cos_theta) / sin_theta if sin_theta != 0 else float("inf")  # y for maximal x (y=img_w)

        def border_dist(v, border):
            return v if v <= 0 else v - border
        # set the possible points
        # some of them will be out of canvas
        possible_pts = [
            ([x_miny, 0], (border_dist(x_miny, img_w), 0)),
            ([0, y_minx], (border_dist(y_minx, img_h), 1)),
            ([x_maxy, img_h], (border_dist(x_maxy, img_w), 0)),
            ([img_w, y_maxx], (border_dist(y_maxx, img_h), 1)),
        ]

        # get the valid and the dismissed (out of canvas) points
        valid_pts = []
        dismissed_pts = []
        for p, dist in possible_pts:
            if 0 <= p[0] <= img_w and 0 <= p[1] <= img_h:
                valid_pts.append(p)
            else:
                dismissed_pts.append((p, dist))

        # from the dismissed points, get the needed ones that are closed to the canvas
        n_needed_pts = 2 - len(valid_pts)
        if n_needed_pts > 0:
            dismissed_pts_sorted = sorted(dismissed_pts, key=lambda x: abs(x[1][0]), reverse=True)

            for _ in range(n_needed_pts):
                p, (dist, coord_idx) = dismissed_pts_sorted.pop()
                p[coord_idx] -= dist  # correct
                valid_pts.append(p)

        p1 = pt(*valid_pts[0])
        p2 = pt(*valid_pts[1])

        lines_ab.append((p1, p2))
    return lines_ab

def ab_lines(lines_hough, w, h):
    projected = project_lines([l[:2] for l in lines_hough], w, h)
    return [(p1, p2, line_dir) for (p1, p2), (_, _, _, line_dir) in zip(projected, lines_hough)]

def draw_line(img, lines, w, h):
    img_line = img.copy()
    lines_ab = ab_lines(lines, w, h)
    for i, (p1, p2, line_dir) in enumerate(lines_ab):
        line_color = (0, 0, 255) if line_dir == 'h' else (0, 255, 0)
        cv2.line(img_line, pt_to_tuple(p1), pt_to_tuple(p2), line_color, 2)
    return img_line

def find_clusters_1d(vals, dist_thresh):

    if type(vals) is not np.ndarray:
        raise TypeError("vals must be a NumPy array")

    if dist_thresh < 0:
        raise ValueError("dist_thresh must be positive")

    clusters = []

    if len(vals) > 0:
        pos_indices_sorted = np.argsort(vals)      # indices of sorted values
        gaps = np.diff(vals[pos_indices_sorted])   # calculate distance between sorted values

        cur_clust = [pos_indices_sorted[0]]  # initialize with first index

        if len(vals) > 1:
            for idx, gap in zip(pos_indices_sorted[1:], gaps):
                if gap >= dist_thresh:           # create new cluster
                    clusters.append(np.array(cur_clust))
                    cur_clust = []
                cur_clust.append(idx)

        clusters.append(np.array(cur_clust))

    assert len(vals) == sum(map(len, clusters))

    return clusters

def zip_clusters_and_values(clusters, values):
    """
    Combine cluster indices in <clusters> (as returned from find_clusters_1d_break_dist) with the respective values
    in <values>.
    Return list of tuples, each tuple representing a cluster and containing two NumPy arrays:
    1. cluster indices into <values>, 2. values of this cluster
    """
    if type(values) is not np.ndarray:
        raise TypeError("values must be a NumPy array")

    clusters_w_vals = []
    for c_ind in clusters:
        c_vals = values[c_ind]
        clusters_w_vals.append((c_ind, c_vals))

    return clusters_w_vals

def find_clusters(lines, direction, method, w, h, **kwargs):
    if not lines:
        raise ValueError("no lines")
    if direction not in ('v', 'h'):
        raise ValueError('invalid direction')
    lines_in_dir = [l for l in lines if l[3]==direction]

    if len(lines_in_dir)==0:
        return []

    lines_ab = ab_lines(lines_in_dir, w, h)

    coord_idx = 0 if direction == 'v' else 1
    positions = np.array([(l[0][coord_idx] + l[1][coord_idx]) / 2 for l in lines_ab])
    clusters = method(positions, **kwargs)
    if type(clusters) != list:
        raise ValueError("'method' returned invalid clusters (must be list)")

    if len(clusters) > 0 and type(clusters[0]) != np.ndarray:
        raise ValueError("'method' returned invalid cluster elements (must be list of numpy.ndarray objects)")

    clusters_w_vals = zip_clusters_and_values(clusters, positions)
    return clusters_w_vals

def draw_line_clusters(img, direction, clusters_w_vals):
    img_draw = img.copy()

    if direction not in ('h', 'v'):
        raise ValueError("invalid value for 'direction': '%s'" % direction)

    n_colors = len(clusters_w_vals)
    color_incr = max(1, round(255 / n_colors))

    for i, (_, vals) in enumerate(clusters_w_vals):
        i += 2
        line_color = (
            (color_incr * i) % 256,
            (color_incr * i * i) % 256,
            (color_incr * i * i * i) % 256,
        )
        draw_line_in_dir(img_draw, direction, vals, line_color)
    return img_draw

def draw_line_in_dir(img, direction, line_positions, line_color, line_width = 2):

    if direction not in ('h', 'v'):
        raise ValueError("invalid value for 'direction': '%s'" % direction)

    h, w = img.shape[:2]

    for pos in line_positions:
        pos = int(round(pos))

        if direction == 'h':
            p1 = (0, pos)
            p2 = (w, pos)
        else:
            p1 = (pos, 0)
            p2 = (pos, h)

        cv2.line(img, p1, p2, line_color, line_width)

def calc_cluster_centers_1d(clusters_w_vals, method=np.median):
    """
    Calculate the cluster centers (for 1D clusters) using <method>.
    <clusters_w_vals> must be a sequence of tuples t where t[1] contains the values (as returned from
    zip_clusters_and_values).
    """
    return [method(vals) for _, vals in clusters_w_vals]

def process_img(img):
    # resize = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def erosion(img):
    kernel = np.ones((2,2), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    return eroded

def get_cols(img, page_col_pos):
    imgg = img.copy()
    cols = []
    for i in range(len(page_col_pos)):
        if i == 0:
            img_col =imgg[0:imgg.shape[0], 0:int(page_col_pos[i])]
        else:
            img_col =imgg[0:imgg.shape[0], int(page_col_pos[i-1]):int(page_col_pos[i])]
        cols.append(img_col)
    return cols

def get_rows(img, page_row_pos):
    imgg = img.copy()
    rows = []
    for i in range(len(page_row_pos)):
        if i == 0:
            img_row = imgg[0:int(page_row_pos[i]), 0:img.shape[1]]
        else:
            img_row = imgg[int(page_row_pos[i-1]):int(page_row_pos[i]), 0:imgg.shape[1]]
        rows.append(img_row)
    return rows

# img_path = r'D:\Desktop\PBE\VB GUI IT3_SO HOA_SO TTTT QBH-2.png'
# img_path = r'D:\Desktop\PBE\1_6.png'
img_path = r'D:\Desktop\PBE\D2D_2.png'
image = cv2.imread(img_path)

def extractTable(image, name):
    h, w = image.shape[:2]

    lines = detectlines(image)
    # print(len(lines))
    img_line = draw_line(image, lines, w, h)

    vertical_clusters = find_clusters(lines, 'v', find_clusters_1d, w, h, dist_thresh=30)
    horizontal_clusters = find_clusters(lines, 'h', find_clusters_1d, w, h, dist_thresh=30)
    # print(len(vertical_clusters))
    img_clus = draw_line_clusters(image, 'h', horizontal_clusters)


    page_col_pos = np.array(calc_cluster_centers_1d(vertical_clusters))
    page_row_pos = np.array(calc_cluster_centers_1d(horizontal_clusters))
    # print('found %d column borders:' % len(page_col_pos))
    # print(page_row_pos)

    page_rows = get_rows(image, page_row_pos)
    # print(page_cols[1])

    page_box = []
    for row in page_rows:
        box = get_cols(row, page_col_pos)
        page_box.append(box)


    page_txt = []
    # i = 0
    # crop= []
    for boxes in page_box:
        col_txt = []
        for box in boxes:
            # crop.append(bo/x)
            boxed = process_img(box)
            # crop.append(boxed)
            txt = pytesseract.image_to_string(boxed, lang='vie_fast', config=r'--psm 3')
            txt = txt.replace("\n", " ", -1)
            col_txt.append(txt)
            print(txt)
        page_txt.append(col_txt)
        # page_txt[str(i)] = col_txt
        # i = i + 1

    data = pd.DataFrame(page_txt)
    data.to_excel(name)
    # i = 0
    # for c in crop:
        # cv2.imwrite(os.path.join(r'D:\Desktop\PBE\crop', str(i)+'.jpg'), c)
        # i = i+ 1
        # cv2.imshow('1', c)
        # cv2.waitKey(0)
        # cv2.destroyWindow('1')

    # cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    # cv2.imshow('1', img_line)
    # cv2.imshow('2', img_clus)
    # cv2.waitKey(0)

name = 'vd1.xlsx'
extractTable(image, name)
# print(len(page_box))

# for boxs in page_box:
#     for box in boxs:
#         #cv2.namedWindow('1', cv2.WINDOW_NORMAL)
#         cv2.imshow('1', box)
#         cv2.waitKey(0)
#         cv2.destroyWindow('1')

# cv2.namedWindow('1', cv2.WINDOW_NORMAL)
# cv2.imshow('1', imgg)
# cv2.waitKey(0)