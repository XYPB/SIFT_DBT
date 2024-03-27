import numpy as np

def calc_box(box):
    return abs(box[2] - box[0]) * abs(box[3] - box[1])

def calc_intersection(box1, box2, canvas_size):
    # (h1, w1, h2, w2)
    canvas1 = np.zeros(canvas_size)
    canvas2 = np.zeros(canvas_size)
    canvas1[box1[0]:box1[2], box1[1]:box1[3]] = 1
    canvas2[box2[0]:box2[2], box2[1]:box2[3]] = 1
    cap = np.sum(np.logical_and(canvas1, canvas2))
    return cap



def calc_iou(box1, box2, canvas_size):
    # (h1, w1, h2, w2)
    cap = calc_intersection(box1, box2, canvas_size)
    area1 = calc_box(box1)
    area2 = calc_box(box2)
    cup = area1 + area2 - cap
    iou = max(cap / cup, 1e-4)
    return iou