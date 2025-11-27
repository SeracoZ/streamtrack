import numpy as np

def iou(a, b):
    """Standard IoU for two bounding boxes."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def add_new_objects(
    inference_state,
    cur_boxes,
    existing_boxes,
    existing_masks,
    rel_idx,
    seen_ids,
    birth_frames=3,
    area_ratio_thresh=0.7,
    iou_match_thresh=0.3
):
    """
    Strict Add module.
    cur_boxes: list of YOLO/pose-detected full-body boxes
    existing_boxes: list of SAM2 mask-derived boxes
    existing_masks: list of binary SAM2 masks (H,W)
    """

    # --- 1. Create M_non (non-occupied mask by SAM2) ---
    if len(existing_masks) > 0:
        M_union = np.zeros_like(existing_masks[0], dtype=np.uint8)
        for m in existing_masks:
            M_union = np.logical_or(M_union, m)
        M_non = np.logical_not(M_union).astype(np.uint8)
    else:
        M_non = None  # no tracked objects yet

    # --- Candidate memory inside inference_state ---
    if "cand_buffer" not in inference_state:
        inference_state["cand_buffer"] = {}  # cid → {"box", "hits", "last"}
    cand = inference_state["cand_buffer"]

    new_adds = []  # (new_id, box)

    # === LOOP ALL DETECTIONS ===
    for det_box in cur_boxes:

        # 2. Hungrian-style matching: if det overlaps any existing box -> NOT new
        overlaps = [iou(det_box, ebox) for ebox in existing_boxes]
        if overlaps and max(overlaps) >= iou_match_thresh:
            continue  # matched to existing person → skip

        # 3. M_non test: confirm the new box is mostly outside SAM2 regions
        if M_non is not None:
            x1, y1, x2, y2 = map(int, det_box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(x2, M_non.shape[1]); y2 = min(y2, M_non.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = M_non[y1:y2, x1:x2]
            ratio = crop.sum() / max(1, crop.size)

            # If the new detection overlaps too much with existing SAM2 masks → NOT new
            if ratio < area_ratio_thresh:
                continue

        # 4. Temporal birth buffer: maintain stable candidate for ≥ birth_frames
        matched_cid = None
        for cid, info in cand.items():
            if iou(det_box, info["box"]) > 0.5:
                matched_cid = cid
                break

        if matched_cid is None:
            cid = len(cand) + 1
            cand[cid] = {
                "box": det_box,
                "hits": 1,
                "last": rel_idx,
            }
        else:
            cand[matched_cid]["hits"] += 1
            cand[matched_cid]["box"] = det_box
            cand[matched_cid]["last"] = rel_idx

            # Confirm new object
            if cand[matched_cid]["hits"] >= birth_frames:
                new_id = max(seen_ids) + 1 if seen_ids else 1
                new_adds.append((new_id, det_box))
                seen_ids.add(new_id)

                # delete confirmed cid
                del cand[matched_cid]

    # 5. Remove stale candidates (not seen for 1 frame)
    stale = []
    for cid, info in cand.items():
        if rel_idx - info["last"] > 1:
            stale.append(cid)
    for cid in stale:
        del cand[cid]

    return new_adds
