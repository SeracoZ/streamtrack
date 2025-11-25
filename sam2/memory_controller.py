import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque


class ConsistencyVerifier:
    """
    Detects:
      1) Occlusion → stop memory update
      2) ID switch → reassign to correct ID
    Using MULTI-FRAME history, not just last frame.

    History includes:
        - past obj_ptr (appearance)
        - past bbox center & area (geometry)
        - past IoU trends
        - visibility score trend
    """

    def __init__(
        self,
        hist_len=5,
        tau_visibility=0.25,
        tau_app_low=0.45,
        tau_iou_low=0.1,
        tau_switch_margin=0.20,
    ):
        self.hist_len = hist_len
        self.tau_visibility = tau_visibility
        self.tau_app_low = tau_app_low
        self.tau_iou_low = tau_iou_low
        self.tau_switch_margin = tau_switch_margin

        # track_id → history buffers
        self.hist_ptr = defaultdict(lambda: deque(maxlen=hist_len))
        self.hist_bbox = defaultdict(lambda: deque(maxlen=hist_len))
        self.hist_iou = defaultdict(lambda: deque(maxlen=hist_len))
        self.hist_vis = defaultdict(lambda: deque(maxlen=hist_len))

    # ----------------------------------------------------------------------

    @staticmethod
    def cosine(a, b):
        if a is None or b is None:
            return 0.0
        return F.cosine_similarity(a, b, dim=-1).item()

    @staticmethod
    def mask_to_bbox(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [
            float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max())
        ]

    @staticmethod
    def bbox_center_area(bbox):
        if bbox is None:
            return (0, 0, 0)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        return cx, cy, area

    @staticmethod
    def bbox_iou(b1, b2):
        if b1 is None or b2 is None:
            return 0.0
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    # ----------------------------------------------------------------------
    # Main update — store history for future checks
    # ----------------------------------------------------------------------

    def update_prev(self, track_id, obj_ptr, mask_np, score_logit):
        bbox = self.mask_to_bbox(mask_np)
        vis = float(torch.sigmoid(score_logit).item())

        # append to history
        self.hist_ptr[track_id].append(obj_ptr.detach().cpu())
        self.hist_bbox[track_id].append(bbox)
        self.hist_vis[track_id].append(vis)

        # compute IoU with last frame for trend analysis
        if len(self.hist_bbox[track_id]) >= 2:
            b2 = self.hist_bbox[track_id][-1]
            b1 = self.hist_bbox[track_id][-2]
            iou = self.bbox_iou(b1, b2)
            self.hist_iou[track_id].append(iou)
        else:
            self.hist_iou[track_id].append(1.0)

    # ----------------------------------------------------------------------
    # Occlusion check with trend
    # ----------------------------------------------------------------------

    def check_occlusion(self, track_id):
        vis_hist = self.hist_vis[track_id]
        if len(vis_hist) == 0:
            return False

        # Current visibility
        curr_vis = vis_hist[-1]

        # sudden drop AND below threshold → occlusion
        if curr_vis < self.tau_visibility:
            return True

        # If last 2-3 vis values are decaying → occlusion
        if len(vis_hist) >= 3:
            if vis_hist[-1] < vis_hist[-2] < vis_hist[-3]:
                if vis_hist[-1] < self.tau_visibility + 0.1:
                    return True

        return False

    # ----------------------------------------------------------------------
    # Multi-frame ID check
    # ----------------------------------------------------------------------

    def multi_frame_app_similarity(self, track_id, obj_ptr):
        """
        Compare obj_ptr with average of last N ptr embeddings.
        """
        ptr_hist = self.hist_ptr[track_id]
        if len(ptr_hist) == 0:
            return 1.0
        mean_prev = torch.stack(list(ptr_hist)).mean(dim=0)
        return self.cosine(obj_ptr.cpu(), mean_prev)

    def multi_frame_motion_iou(self, track_id, curr_bbox):
        """
        Compare current bbox with previous N bboxes (mean IoU).
        """
        bbox_hist = self.hist_bbox[track_id]
        if len(bbox_hist) == 0:
            return 1.0
        ious = [self.bbox_iou(curr_bbox, b) for b in bbox_hist]
        return float(sum(ious) / len(ious))

    def check_id_switch(self, track_id, obj_ptr, mask_np):
        if len(self.hist_ptr[track_id]) == 0:
            return track_id

        curr_bbox = self.mask_to_bbox(mask_np)

        # (A) Multi-frame appearance similarity
        sim_app = self.multi_frame_app_similarity(track_id, obj_ptr)

        # (B) Multi-frame IoU / geometry continuity
        sim_iou = self.multi_frame_motion_iou(track_id, curr_bbox)

        mismatch = (sim_app < self.tau_app_low and sim_iou < self.tau_iou_low)

        if not mismatch:
            return track_id

        # Cross-track reassign
        best_id = track_id
        best_score = sim_app

        for other_id in self.hist_ptr:
            if other_id == track_id:
                continue

            sim_app2 = self.multi_frame_app_similarity(other_id, obj_ptr)

            if sim_app2 - best_score > self.tau_switch_margin:
                best_score = sim_app2
                best_id = other_id

        if best_id != track_id:
            print(f"[Verifier] ID CORRECTED: {track_id} → {best_id}")

        return best_id

    # ----------------------------------------------------------------------
    # API exposed to SAM2 tracker
    # ----------------------------------------------------------------------

    def analyze(self, track_id, obj_ptr, score_logit, mask_np):
        """
        Uses multi-frame occlusion + multi-frame ID consistency.
        """
        # 1) Update history first (but temporarily)
        self.update_prev(track_id, obj_ptr, mask_np, score_logit)

        # 2) Occlusion detection
        if self.check_occlusion(track_id):
            return {
                "is_occluded": True,
                "correct_id": track_id
            }

        # 3) ID-switch detection
        correct_id = self.check_id_switch(track_id, obj_ptr, mask_np)

        return {
            "is_occluded": False,
            "correct_id": correct_id
        }
