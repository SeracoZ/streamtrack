import cv2
import numpy as np
import os


def draw_score_on_frame(frame, box, obj_id, occ_info, qual_info):
    """
    Draw occlusion + quality scores next to the bbox.
    """
    x1, y1, x2, y2 = map(int, box)

    occ_score = occ_info["occlusion_score"] if occ_info else None
    qual_score = qual_info["final_quality"] if qual_info else None

    # Put text above the box
    text1 = f"ID {obj_id}"
    text2 = f"occ={occ_score:.2f}" if occ_score is not None else "occ=?"
    text3 = f"qual={qual_score:.2f}" if qual_score is not None else "qual=?"

    cv2.putText(frame, text1, (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, text2, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
    cv2.putText(frame, text3, (x1, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

def visualize_tracking(frame, out_obj_ids, out_mask_logits, id_colors, save_path=None):
    out_mask_logits = out_mask_logits.detach().cpu()

    def get_color(obj_id):
        if obj_id not in id_colors:
            rng = np.random.default_rng(seed=obj_id * 99991)
            id_colors[obj_id] = tuple(int(c) for c in rng.integers(0, 255, size=3))
        return id_colors[obj_id]

    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).numpy().squeeze(0)
        color = get_color(obj_id)

        mask_vis = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)

        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            continue
        x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)

# --- IOU helper ---
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def is_full_body(kc, min_visible=3):
    """
    kc: keypoint confidences, shape (17,)
    """
    LOWER_BODY = [11, 12, 13, 14, 15, 16]
    visible = sum(kc[i] > 0.2 for i in LOWER_BODY)
    return visible >= min_visible



def filter_overlapping_bboxes(bboxes, contain_thresh=0.9):
    """
    Remove smaller boxes that are almost fully contained within larger ones.
    Keep only the larger box.
    """
    if len(bboxes) <= 1:
        return bboxes

    keep = [True] * len(bboxes)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]

    for i in range(len(bboxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue
            box_i, box_j = bboxes[i], bboxes[j]

            xi1, yi1, xi2, yi2 = box_i
            xj1, yj1, xj2, yj2 = box_j
            inter_x1, inter_y1 = max(xi1, xj1), max(yi1, yj1)
            inter_x2, inter_y2 = min(xi2, xj2), min(yi2, yj2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            small_area = min(areas[i], areas[j])
            contain_ratio = inter_area / small_area if small_area > 0 else 0

            if contain_ratio >= contain_thresh:
                # keep the larger one
                if areas[i] >= areas[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [b for b, k in zip(bboxes, keep) if k]


def visualize_pose(frame, results, save_path=None):
    """
    Visualize YOLO pose keypoints and skeletons on a frame.
    Args:
        frame: np.ndarray (BGR)
        results: YOLO output for one frame
        save_path: optional path to save the image
    """
    import cv2
    import numpy as np

    frame_vis = frame.copy()
    keypoints = results.keypoints.xy.cpu().numpy()  # (N,17,2)
    kpt_conf = results.keypoints.conf.cpu().numpy()  # (N,17)

    # COCO keypoint connections
    skeleton = [
        (5, 7), (7, 9),     # left arm
        (6, 8), (8, 10),    # right arm
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16), # right leg
        (5, 6), (11, 12),   # shoulders, hips
        (5, 11), (6, 12)    # torso
    ]

    for i in range(len(keypoints)):
        pts = keypoints[i]
        confs = kpt_conf[i]
        color = (0, 255, 0)
        # draw skeleton
        for (a, b) in skeleton:
            if confs[a] > 0.3 and confs[b] > 0.3:
                cv2.line(frame_vis,
                         (int(pts[a][0]), int(pts[a][1])),
                         (int(pts[b][0]), int(pts[b][1])),
                         color, 2)
        # draw keypoints
        for (x, y), c in zip(pts, confs):
            if c > 0.3:
                cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 0, 255), -1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame_vis)
    return frame_vis


def adjust_box_to_pose(box, keypoints, kpt_conf=None, conf_thresh=0.3, margin=10):
    """
    Expand or reshape YOLO bbox to fully include the visible pose keypoints.
    Args:
        box: [x1, y1, x2, y2] from YOLO
        keypoints: (17, 2) array of keypoint xy
        kpt_conf: (17,) array of keypoint confs
        conf_thresh: threshold to include keypoints
        margin: padding (pixels)
    Returns:
        new_box: adjusted [x1, y1, x2, y2]
    """
    if kpt_conf is None:
        kpt_conf = np.ones(len(keypoints))

    valid = kpt_conf > conf_thresh
    if not np.any(valid):
        return box  # fallback

    xs, ys = keypoints[valid, 0], keypoints[valid, 1]
    x1p, y1p, x2p, y2p = xs.min(), ys.min(), xs.max(), ys.max()

    # Combine YOLO box & keypoint range
    x1 = min(box[0], x1p) - margin
    y1 = min(box[1], y1p) - margin
    x2 = max(box[2], x2p) + margin
    y2 = max(box[3], y2p) + margin
    return [max(x1, 0), max(y1, 0), x2, y2]