import numpy as np

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_quality_for_objects(existing_boxes, det_boxes, det_scores):
    """
    Compute per-object QUALITY using only:
        - SAM2 existing boxes  (existing_boxes)
        - YOLO det boxes       (det_boxes)
        - YOLO detection scores (det_scores)

    Same behavior/structure as compute_occlusion_for_objects().
    """

    quality_list = []

    for sam_box in existing_boxes:
        best_iou = 0.0
        best_det_score = 0.0
        best_det_box = None

        # 1️⃣ Find the best matching YOLO box
        for det_box, score in zip(det_boxes, det_scores):
            cur_iou = iou(sam_box, det_box)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_det_box = det_box
                best_det_score = score

        # 2️⃣ Size stability (SAM vs detection area)
        sam_area = (sam_box[2] - sam_box[0]) * (sam_box[3] - sam_box[1])

        if best_det_box is not None:
            det_area = (best_det_box[2] - best_det_box[0]) * (best_det_box[3] - best_det_box[1])
            if det_area > 0:
                area_similarity = min(sam_area, det_area) / max(sam_area, det_area)
            else:
                area_similarity = 0.0
        else:
            area_similarity = 0.0

        # 3️⃣ Final quality score (weighted sum)
        final_quality = (
            best_iou * 0.5 +
            area_similarity * 0.3 +
            best_det_score * 0.2
        )

        quality_list.append({
            "sam_box": sam_box,
            "best_det_box": best_det_box,
            "best_iou": float(best_iou),
            "area_similarity": float(area_similarity),
            "det_score": float(best_det_score),
            "final_quality": float(final_quality),
            "is_good": final_quality > 0.55    # threshold — same idea as occlusion
        })

    return quality_list



def attach_quality_to_inference_state(inference_state, quality_list, out_obj_ids, rel_idx):
    """
    Mirror attach_occlusion_to_inference_state.

    Stores:
        non_cond_frame_outputs[rel_idx]["quality"]
        cond_frame_outputs[rel_idx]["quality"]
    """

    if "quality_info" not in inference_state:
        inference_state["quality_info"] = {}

    for obj_id, q in zip(out_obj_ids, quality_list):
        inference_state["quality_info"][obj_id] = q

        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is None:
            continue

        obj_dict = inference_state["output_dict_per_obj"][obj_idx]

        # write to non_cond
        nco = obj_dict.get("non_cond_frame_outputs", {})
        if rel_idx in nco:
            nco[rel_idx]["quality"] = q

        # write to cond
        cfo = obj_dict.get("cond_frame_outputs", {})
        if rel_idx in cfo:
            cfo[rel_idx]["quality"] = q
