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


def compute_occlusion_for_objects(existing_boxes, det_boxes, det_scores):
    """
    Use YOLO detection to compute occlusion information for each tracked object.

    Args:
        existing_boxes: list of SAM2 predicted boxes aligned with out_obj_ids
                        e.g. [[x1,y1,x2,y2], ...]
        det_boxes:      list of Yolo detection boxes (filtered full-body)
        det_scores:     detection confidences for det_boxes

    Returns:
        occlusion_list: list of dicts, same length as existing_boxes
                        Each element corresponds to ONE object.
    """
    occlusion_list = []

    for sam_box in existing_boxes:
        best_iou = 0.0
        best_det_score = 0.0
        best_det_box = None

        for det_box, det_score in zip(det_boxes, det_scores):
            iou_val = iou(sam_box, det_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_det_score = det_score
                best_det_box = det_box

        sam_area = (sam_box[2] - sam_box[0]) * (sam_box[3] - sam_box[1])

        if best_det_box is not None:
            det_area = (best_det_box[2] - best_det_box[0]) * (best_det_box[3] - best_det_box[1])
            area_ratio = sam_area / det_area if det_area > 0 else 0.0
        else:
            area_ratio = 0.0

        # Weighted occlusion score (lower means less occluded)
        occlusion_score = (
            (1 - best_iou) * 0.5 +
            (1 - area_ratio) * 0.3 +
            (1 - best_det_score) * 0.2
        )

        occlusion_list.append({
            "sam_box": sam_box,
            "det_box": best_det_box,
            "det_score": float(best_det_score),
            "iou_with_det": float(best_iou),
            "area_ratio": float(area_ratio),
            "occlusion_score": float(occlusion_score),
            "is_occluded": occlusion_score > 0.6
        })

    return occlusion_list



def attach_occlusion_to_inference_state(inference_state, occlusion_list, out_obj_ids, rel_idx):
    """
    Attach occlusion information to SAM2 memory output_dict_per_obj.

    Args:
        inference_state: predictor inference state (dict)
        occlusion_list: list of occlusion dicts from compute_occlusion_for_objects()
        out_obj_ids: list of SAM2 active object IDs
        rel_idx: current frame index

    Effect:
        Adds occlusion information to:
            inference_state["output_dict_per_obj"][obj_id]["non_cond_frame_outputs"][rel_idx]
        and (if exists):
            inference_state["output_dict_per_obj"][obj_id]["cond_frame_outputs"][rel_idx]
    """

    if "occlusion_info" not in inference_state:
        inference_state["occlusion_info"] = {}

    for obj_id, occ in zip(out_obj_ids, occlusion_list):
        inference_state["occlusion_info"][obj_id] = occ

        obj_dict = inference_state["output_dict_per_obj"].get(obj_id, None)
        if obj_dict is None:
            continue

        # Non-conditioning memory entry
        nco = obj_dict.get("non_cond_frame_outputs", {})
        if rel_idx in nco:
            nco[rel_idx]["det_occlusion"] = occ

        # Cond frame memory entry
        cfo = obj_dict.get("cond_frame_outputs", {})
        if rel_idx in cfo:
            cfo[rel_idx]["det_occlusion"] = occ
