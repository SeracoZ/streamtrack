import numpy as np

class MemoryController:
    def __init__(self, promote_gate=8, iou_thresh=0.4, area_jump=0.4):
        self.promote_gate = promote_gate   # N frames stable → promote t-N
        self.iou_thresh = iou_thresh       # min IOU for “good frame”
        self.area_jump = area_jump         # max area jump ratio
        self.prev_boxes = {}               # previous frame boxes
        self.good_count = {}               # sliding counter

    # ------------------------
    # Utility Functions
    # ------------------------
    def _mask_to_box(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _iou(self, a, b):
        if a is None or b is None:
            return 0
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def _is_good_frame(self, prev_box, new_box):
        if prev_box is None or new_box is None:
            return False
        iou = self._iou(prev_box, new_box)
        if iou < self.iou_thresh:
            return False
        areaA = (prev_box[2]-prev_box[0])*(prev_box[3]-prev_box[1])
        areaB = (new_box[2]-new_box[0])*(new_box[3]-new_box[1])
        if abs(areaA - areaB) / max(areaA, 1) > self.area_jump:
            return False
        return True

    # ------------------------
    # Promotion of memory
    # ------------------------
    def _promote_non_cond(self, inference_state, obj_id, promote_frame):
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is None:
            return False
        obj_dict = inference_state["output_dict_per_obj"][obj_idx]

        nc = obj_dict["non_cond_frame_outputs"]
        cond = obj_dict["cond_frame_outputs"]

        if promote_frame in nc:
            cond[promote_frame] = nc.pop(promote_frame)
            print(f"[MemoryController] PROMOTED frame {promote_frame} → cond_frame (obj {obj_id})")
            return True
        return False

    # ------------------------
    # MAIN UPDATE CALL
    # Called each frame after propagate_in_video yields results
    # ------------------------
    def update(self, inference_state, frame_idx, obj_ids, masks):
        """
        masks: [N, 1, H, W] numpy or torch (video_res_masks)
        """

        masks_np = masks.detach().cpu().numpy().squeeze(1)

        for obj_id, mask in zip(obj_ids, masks_np):
            # Init tracking structures if first time
            if obj_id not in self.prev_boxes:
                self.prev_boxes[obj_id] = None
            if obj_id not in self.good_count:
                self.good_count[obj_id] = 0

            # Extract bbox
            new_box = self._mask_to_box(mask)
            prev_box = self.prev_boxes[obj_id]

            # Evaluate quality
            if self._is_good_frame(prev_box, new_box):
                self.good_count[obj_id] += 1
            else:
                self.good_count[obj_id] = 0

            # Attempt promotion when stable
            if self.good_count[obj_id] >= self.promote_gate:
                promote_frame = frame_idx - self.promote_gate
                self._promote_non_cond(inference_state, obj_id, promote_frame)
                self.good_count[obj_id] = 0

            # Update stored state
            self.prev_boxes[obj_id] = new_box
