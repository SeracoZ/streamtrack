import numpy as np

class LightKalman:
    """
    Light OC-SORT-style Kalman filter on [x, y, s, r, vx, vy, vs, vr].
    x, y: box center
    s: scale (area)
    r: aspect ratio
    v*: velocities
    """
    def __init__(self, bbox):
        # bbox: [x1, y1, x2, y2]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = max(bbox[2] - bbox[0], 1e-3)
        h  = max(bbox[3] - bbox[1], 1e-3)
        s  = w * h
        r  = w / h

        # state
        self.x = np.array([cx, cy, s, r, 0., 0., 0., 0.], dtype=np.float32)

        # state covariance
        self.P = np.eye(8, dtype=np.float32) * 10.0

        # constant velocity model
        dt = 1.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = dt  # position += velocity * dt

        # measurement model z = [x, y, s, r]
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(8, dtype=np.float32) * 1e-2   # process noise
        self.R = np.eye(4, dtype=np.float32) * 1e-1   # meas noise

    def predict(self):
        # x = F x
        self.x = self.F @ self.x
        # P = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.to_bbox()

    def update(self, bbox):
        # bbox -> z = [x, y, s, r]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = max(bbox[2] - bbox[0], 1e-3)
        h  = max(bbox[3] - bbox[1], 1e-3)
        s  = w * h
        r  = w / h
        z  = np.array([cx, cy, s, r], dtype=np.float32)

        # innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # update state
        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.to_bbox()

    def to_bbox(self):
        cx, cy, s, r = self.x[0], self.x[1], self.x[2], self.x[3]
        w = np.sqrt(max(s * r, 1e-3))
        h = max(s / (w + 1e-6), 1e-3)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]
