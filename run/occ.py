import numpy as np


# ============================================================
# IoU utilities
# ============================================================
def bbox_iou(a, b):
    """IoU between [x1,y1,x2,y2], safe for None."""
    if a is None or b is None:
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0

    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    if inter <= 0:
        return 0.0

    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def mask_iou(mask1, mask2):
    """mIoU between two SAM2 binary masks."""
    if mask1 is None or mask2 is None:
        return 0.0

    inter = np.logical_and(mask1, mask2).sum()
    if inter == 0:
        return 0.0

    union = mask1.sum() + mask2.sum() - inter
    return inter / union if union > 0 else 0.0


def compute_logit_confidence(logit_mask, mask_bin):
    """Mean logit value over positive mask region."""
    inside = logit_mask[mask_bin > 0]
    if inside.size == 0:
        return -999  # extremely low confidence
    return float(inside.mean())


# ============================================================
# Main occlusion logic (modified with BEFORE-MASK REASONING)
# ============================================================
def occlusion_check(
    predictor,
    inference_state,
    frame_idx,
    track_ids,
    track_logits,
    KF_boxes,
    YOLO_boxes,
    shrink_thr=0.95,
    iou_thr=0.30,
    miss_det_thr=2,
    miou_thr=0.80
):
    """
    Occlusion detection includes:

    1. YOLO missing
    2. KF overlap
    3. Mask shrinking
    4. Mask mIoU > 0.8 collapse
    5. NEW: front/back occluder detection using BEFORE-MASK:
        - previous mask area
        - previous mask logit confidence
        - previous bbox (KF predicted center)
    """

    occ_tracks = inference_state.setdefault("occ_tracks", {})
    occ_ids_set = inference_state.setdefault("occluded_ids", set())

    # --------------------------------------------------------
    # Extract masks, areas, logits for each ID
    # --------------------------------------------------------
    logits_cpu = track_logits.detach().cpu()
    masks = {}
    areas = {}
    logits_dict = {}  # pixel logits for each ID

    for idx, tid in enumerate(track_ids):
        logit = logits_cpu[idx].numpy().squeeze()
        mask_bin = (logit > 0).astype(np.uint8)

        masks[tid] = mask_bin
        areas[tid] = int(mask_bin.sum())
        logits_dict[tid] = logit

    # ========================================================
    # FIRST: HANDLE mIoU COLLAPSE (JOINED MASKS)
    # ========================================================
    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i + 1:]:
            iou = mask_iou(masks[tid1], masks[tid2])
            if iou > miou_thr:

                state1 = occ_tracks.setdefault(
                    tid1,
                    {"prev_area": None, "prev_conf": None, "prev_bbox": None,
                     "last_area": None, "missing_det": 0, "occluded": False}
                )
                state2 = occ_tracks.setdefault(
                    tid2,
                    {"prev_area": None, "prev_conf": None, "prev_bbox": None,
                     "last_area": None, "missing_det": 0, "occluded": False}
                )

                # Current values
                conf1 = compute_logit_confidence(logits_dict[tid1], masks[tid1])
                conf2 = compute_logit_confidence(logits_dict[tid2], masks[tid2])

                # Previous values
                p_area1, p_area2 = state1["prev_area"], state2["prev_area"]
                p_conf1, p_conf2 = state1["prev_conf"], state2["prev_conf"]

                # ---- Decide front/back score ----
                score1 = 0
                score2 = 0

                # historical area (larger → front)
                if p_area1 is not None and p_area2 is not None:
                    score1 += (p_area1 > p_area2)
                    score2 += (p_area2 > p_area1)

                # historical confidence (higher → front)
                if p_conf1 is not None and p_conf2 is not None:
                    score1 += (p_conf1 > p_conf2)
                    score2 += (p_conf2 > p_conf1)

                # fallback: current logit confidence
                score1 += (conf1 > conf2)
                score2 += (conf2 > conf1)

                # Determine back (occluded) ID
                back = tid2 if score1 > score2 else tid1

                # Mark occluded
                st = occ_tracks[back]
                if not st["occluded"]:
                    st["occluded"] = True
                    occ_ids_set.add(back)
                    print(f"[Frame {frame_idx}] 🟠 mIoU collapse → ID {back} OCCLUDED (using before-mask history)")

    # --------------------------------------------------------
    # YOLO visibility check
    # --------------------------------------------------------
    id_has_det = {tid: False for tid in track_ids}

    if YOLO_boxes is not None:
        for tid in track_ids:
            kf_box = KF_boxes.get(tid, None)
            if kf_box is None:
                continue

            for det in YOLO_boxes:
                if det is None:
                    continue
                if bbox_iou(kf_box, det) > 0.3:
                    id_has_det[tid] = True
                    break

    # ========================================================
    # SECOND: KF-BASED OCCLUSION WITH BEFORE-MASK LOGIC
    # ========================================================
    for tid in track_ids:

        state = occ_tracks.setdefault(
            tid,
            {"prev_area": None, "prev_conf": None, "prev_bbox": None,
             "last_area": None, "missing_det": 0, "occluded": False}
        )

        KF_box_i = KF_boxes.get(tid, None)

        # --- YOLO missing counter
        if id_has_det[tid]:
            state["missing_det"] = 0
        else:
            state["missing_det"] += 1

        # --- Mask shrinking detection
        cur_area = areas[tid]
        last_area = state["last_area"]
        shrinking = (last_area is not None) and (cur_area < shrink_thr * last_area)

        # --- KF overlap
        KF_overlap = False
        for other_id in track_ids:
            if other_id == tid:
                continue
            if bbox_iou(KF_box_i, KF_boxes.get(other_id, None)) > iou_thr:
                KF_overlap = True
                break

        # --- KF occlusion confirmation (NEED FRONT/BACK DECISION)
        if state["missing_det"] >= miss_det_thr and KF_overlap:

            # if shrinking, current track is the back one
            if shrinking:
                back = tid
            else:
                # Use historical/ current confidence to decide back
                conf_self = compute_logit_confidence(logits_dict[tid], masks[tid])
                p_conf_self = state["prev_conf"]

                back = None
                for other_id in track_ids:
                    if other_id == tid:
                        continue
                    conf_other = compute_logit_confidence(logits_dict[other_id], masks[other_id])
                    p_conf_other = occ_tracks.get(other_id, {}).get("prev_conf")

                    # Compare previous-frame confidence first
                    if p_conf_other is not None and p_conf_self is not None:
                        if p_conf_other > p_conf_self:
                            back = tid
                            break
                    # fallback current frame
                    if conf_other > conf_self:
                        back = tid
                        break

            # mark only the occluded (back) ID
            if back is not None:
                st = occ_tracks.setdefault(back, {"last_area": None, "missing_det": 0, "occluded": False})
                if not st["occluded"]:
                    st["occluded"] = True
                    occ_ids_set.add(back)
                    print(f"[Frame {frame_idx}] 🟡 KF OCCLUSION → ID {back} (determined by history-based front/back)")

        # --------------------------------------------------------
        # Store previous-frame information
        # --------------------------------------------------------
        state["prev_area"] = cur_area
        state["prev_conf"] = compute_logit_confidence(logits_dict[tid], masks[tid])
        state["prev_bbox"] = KF_box_i

        state["last_area"] = cur_area
