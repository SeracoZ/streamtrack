import numpy as np
from filterpy.kalman import KalmanFilter

def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.array([cx, cy, w, h], dtype=float)


def cxcywh_to_xyxy(state4):
    cx, cy, w, h = state4
    return [
        float(cx - w/2),
        float(cy - h/2),
        float(cx + w/2),
        float(cy + h/2)
    ]


class SimpleKF:
    """SORT-style 8D KF using FilterPy with correct column-vector shapes."""

    def __init__(self, init_bbox):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        z = xyxy_to_cxcywh(init_bbox)

        # FilterPy stores x as column vector (8×1)
        self.kf.x = np.zeros((8,1), dtype=float)
        self.kf.x[:4, 0] = z   # <-- FIXED assignment

        dt = 1.0
        self.kf.F = np.array([
            [1,0,0,0, dt,0, 0, 0],
            [0,1,0,0, 0, dt,0, 0],
            [0,0,1,0, 0, 0, dt,0],
            [0,0,0,1, 0, 0, 0, dt],
            [0,0,0,0, 1,0, 0, 0],
            [0,0,0,0, 0,1, 0, 0],
            [0,0,0,0, 0,0, 1, 0],
            [0,0,0,0, 0,0, 0, 1],
        ], dtype=float)

        self.kf.H = np.zeros((4,8))
        self.kf.H[0,0] = 1
        self.kf.H[1,1] = 1
        self.kf.H[2,2] = 1
        self.kf.H[3,3] = 1

        self.kf.P *= 10
        self.kf.R *= 1
        self.kf.Q *= 0.1


    def predict(self):
        self.kf.predict()
        # .x is still (8,1); take [:4,0]
        return cxcywh_to_xyxy(self.kf.x[:4,0])


    def update(self, det_box):
        z = xyxy_to_cxcywh(det_box)
        self.kf.update(z)
        return cxcywh_to_xyxy(self.kf.x[:4,0])


class KFTrackerManager:
    def __init__(self):
        self.trackers = {}

    def init_if_needed(self, tid, bbox):
        if tid not in self.trackers:
            self.trackers[tid] = SimpleKF(bbox)

    def update_with_det(self, tid, det_bbox):
        self.init_if_needed(tid, det_bbox)
        return self.trackers[tid].update(det_bbox)

    def predict(self, tid):
        if tid not in self.trackers:
            return None
        return self.trackers[tid].predict()

    def get_kf_boxes(self, id_list):
        out = {}
        for tid in id_list:
            if tid in self.trackers:
                out[tid] = cxcywh_to_xyxy(self.trackers[tid].kf.x[:4,0])
            else:
                out[tid] = None
        return out
