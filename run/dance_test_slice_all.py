import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
import cv2


from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO



def visualize_tracking(frame, out_obj_ids, out_mask_logits, id_colors, save_path=None):
    """
    Draw SAM2 masks & boxes for all active objects and save to file (no display).
    """
    out_mask_logits = out_mask_logits.detach().cpu()

    def get_color(obj_id):
        if obj_id not in id_colors:
            id_colors[obj_id] = tuple(int(c) for c in np.random.randint(0, 255, 3))
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

def find_initial_bboxes(video_path, frame_names, max_tries=5):
    """
    Try first several frames until a valid detection is found.
    Return (frame_idx, bboxes) or (None, [])
    """
    for i in range(min(max_tries, len(frame_names))):
        frame_path = os.path.join(video_path, frame_names[i])
        results = det_model(frame_path, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        kpts = results.keypoints.conf.cpu().numpy()

        # filter persons
        bboxes = [
            boxes[j].tolist()
            for j in range(len(boxes))
            if classes[j] == 0 and scores[j] >= 0.4 and is_full_body(kpts[j])
        ]
        bboxes = filter_overlapping_bboxes(bboxes, contain_thresh=0.9)

        if bboxes:
            print(f"✅ Initial detection found at frame {i}")
            return i, bboxes

    print("❌ No valid person detected in the first several frames.")
    return None, []


def run_sequence(predictor, video_path, frame_names, offset, writer, save_vis=True, quiet=False):
    """Run SAM2+YOLO tracking with tqdm progress and frame saving only."""
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    # --- Initial detection ---
    first_frame = os.path.join(video_path, frame_names[0])
    results = det_model(first_frame, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()  # (N,4)
    scores = results.boxes.conf.cpu().numpy()  # (N,)
    classes = results.boxes.cls.cpu().numpy()  # (N,)
    kpts = results.keypoints.conf.cpu().numpy()  # (N,17)

    # Cleanest filtering loop
    filtered = [
        boxes[i].tolist()
        for i in range(len(boxes))
        if classes[i] == 0
           and scores[i] >= 0.4
           and is_full_body(kpts[i])  # ✅ full-body check
    ]

    if not filtered:
        person_idx = [i for i in range(len(boxes)) if classes[i] == 0]
        if person_idx:
            # pick person with max confidence
            best_i = max(person_idx, key=lambda i: scores[i])
            print("⚠️ No full-body person detected → using highest-score person as fallback")
            filtered = [(boxes[best_i].tolist(), scores[best_i])]
        else:
            print("❌ No person at all in first frame → SKIP sequence")
            return
    bboxes = [b for b, s in filtered]

    bboxes = filter_overlapping_bboxes(bboxes, contain_thresh=0.9)   # ✅ added filter

    if not bboxes:
        print("⚠️ No person detected, skipping...")
        return

    id_colors, seen_ids = {}, set()
    for obj_id, box in enumerate(bboxes, start=1):
        predictor.add_new_points_or_box(inference_state, frame_idx=0,
                                        obj_id=obj_id, box=np.array(box, dtype=np.float32))
        seen_ids.add(obj_id)

    vis_dir = os.path.join("vis_results", os.path.basename(video_path)) if save_vis else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # === MAIN PROPAGATION LOOP ===
    with torch.no_grad():
        for rel_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # === DEFERRED REMOVAL (executed safely at the start of each frame) ===
            if "pending_remove" in inference_state and inference_state["pending_remove"]:
                to_remove = list(inference_state["pending_remove"])
                for rid in to_remove:
                    try:
                        predictor.soft_remove_object(inference_state, rid)
                        tqdm.write(f"🚫 Soft-removed object {rid} at frame {rel_idx}")
                    except Exception as e:
                        tqdm.write(f"[WARN] Failed to remove {rid}: {e}")
                    # cleanup local records
                    inference_state["last_seen"].pop(rid, None)
                    if "seen_ids" in locals() and rid in seen_ids:
                        seen_ids.remove(rid)
                inference_state["pending_remove"].clear()

            abs_idx = offset + rel_idx

            frame_path = os.path.join(video_path, frame_names[rel_idx])
            frame = cv2.imread(frame_path)

            # YOLO detect every frame
            ########################################################################33
            results = det_model(frame_path, verbose=False)[0]

            boxes = results.boxes.xyxy.cpu().numpy()  # (N,4)
            scores = results.boxes.conf.cpu().numpy()  # (N,)
            classes = results.boxes.cls.cpu().numpy()  # (N,)
            kpts = results.keypoints.conf.cpu().numpy()  # (N,17)

            # ✅ Cleanest version
            cur_boxes = [
                boxes[i].tolist()
                for i in range(len(boxes))
                if classes[i] == 0
                   and scores[i] >= 0.4
                   and is_full_body(kpts[i])  # ✅ ONLY full-body dancers
            ]

            #################################################################################

            # compute existing boxes from masks
            out_mask_logits_cpu = out_mask_logits.detach().cpu()
            existing_boxes = []
            for i in range(len(out_obj_ids)):
                mask = (out_mask_logits_cpu[i] > 0.0).numpy().squeeze(0)
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                existing_boxes.append([x1, y1, x2, y2])

            # === CLEANUP PHASE ===
            DISAPPEAR_THRESH = 10
            HIGH_IOU_THRESH = 0.9

            # Initialize helper dicts
            if "last_seen" not in inference_state:
                inference_state["last_seen"] = {obj_id: rel_idx for obj_id in out_obj_ids}
            if "pending_remove" not in inference_state:
                inference_state["pending_remove"] = set()

            # Update last seen
            for obj_id in out_obj_ids:
                inference_state["last_seen"][obj_id] = rel_idx

            # 1️⃣ Detect disappeared objects
            for obj_id, last_f in list(inference_state["last_seen"].items()):
                if rel_idx - last_f > DISAPPEAR_THRESH:
                    inference_state["pending_remove"].add(obj_id)


            # 3️⃣ Defer actual removal until next frame
            if inference_state["pending_remove"]:
                tqdm.write(f"[Frame {rel_idx}] Marked for removal: {list(inference_state['pending_remove'])}")

            # === HARD REMOVAL PHASE ===
            RETIRE_THRESH = 20  # number of frames since last seen before permanent deletion

            if "last_seen" in inference_state:
                for obj_id, last_f in list(inference_state["last_seen"].items()):
                    if rel_idx - last_f > RETIRE_THRESH:
                        # Hard-remove only if already soft-removed earlier
                        if "removed_obj_ids" in inference_state and obj_id in inference_state["removed_obj_ids"]:
                            try:
                                predictor.hard_remove_object(inference_state, obj_id)
                                tqdm.write(f"🧹 Hard-removed object {obj_id} (stale > {RETIRE_THRESH})")
                            except Exception as e:
                                tqdm.write(f"[WARN] Hard-remove failed for {obj_id}: {e}")

            # --- Queue new detections (with overlap filtering) ---
            pending_new = []
            for det_box in cur_boxes:
                overlaps = [iou(det_box, ebox) for ebox in existing_boxes]
                if not overlaps or max(overlaps) < 0.3:
                    new_id = max(seen_ids) + 1 if seen_ids else 1
                    tqdm.write(f"🟢 Candidate new object {new_id} at frame {rel_idx}")
                    pending_new.append((new_id, det_box))

            # ✅ Filter overlapping candidates before assigning IDs
            new_objects = []
            if pending_new:
                boxes_only = [b for _, b in pending_new]
                filtered_boxes = filter_overlapping_bboxes(boxes_only, contain_thresh=0.9)

                final_new_objects = []
                for new_id, box in pending_new:
                    if any(np.allclose(box, fb, atol=1e-2) for fb in filtered_boxes):
                        final_new_objects.append((new_id, box))
                        seen_ids.add(new_id)   # ✅ add only after filtering
                        new_objects.append(new_id)
                inference_state["pending_new_objects"] = final_new_objects

                if new_objects:
                    # Draw filtered detections
                    for new_id, det_box in final_new_objects:
                        x1, y1, x2, y2 = map(int, det_box)
                        color = (0, 255, 0)

                        ################################3
                        label = f"New ID {new_id}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


                        #########################################
            else:
                inference_state["pending_new_objects"] = []

            # --- Write results ---
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits_cpu[i] > 0.0).numpy().squeeze(0)
                y, x = np.where(mask)
                if len(x) == 0 or len(y) == 0:
                    continue
                x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                w, h = x2 - x1, y2 - y1
                line = f"{abs_idx+1},{out_obj_id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
                writer.write(line)

            # --- Save visualization frame only (no display) ---
            if save_vis:
                save_path = os.path.join(vis_dir, f"{rel_idx:06d}.jpg")
                visualize_tracking(frame, out_obj_ids, out_mask_logits, id_colors,
                                   save_path=save_path)

            torch.cuda.empty_cache()

    del inference_state
    torch.cuda.empty_cache()


def main():
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    for seq in sorted(os.listdir(split_dir)):
        img_dir = os.path.join(split_dir, seq, "img1")
        if not os.path.isdir(img_dir):
            continue
        test_list = sorted(os.listdir(split_dir))[19]
        if seq not in test_list:
            continue

        print(f"\n=== Processing sequence: {seq} ===")
        frame_names = sorted(
            [p for p in os.listdir(img_dir) if p.lower().endswith((".jpg", ".jpeg"))],
            key=lambda p: int(os.path.splitext(p)[0])
        )

        save_dir = f"results/{split}"
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, f"{seq}.txt")

        with open(out_file, "w") as f:
            num_frames = len(frame_names)

            # Direct mode
            if num_frames <= MAX_DIRECT:
                print(f" → Direct processing ({num_frames} frames)")
                run_sequence(predictor, img_dir, frame_names, offset=0, writer=f, save_vis=False, quiet=False)

            # Sliding mode
            else:
                num_slices = (num_frames + SLIDE_SIZE - 1) // SLIDE_SIZE
                print(f" → Sliding mode: {num_frames} frames, {num_slices} slices")

                frame_idx = 0
                slice_idx = 1
                while frame_idx < num_frames:
                    slice_frames = frame_names[frame_idx:frame_idx+SLIDE_SIZE]
                    print(f"   → Slice {slice_idx}/{num_slices} "
                          f"(frames {frame_idx+1}-{min(frame_idx+SLIDE_SIZE, num_frames)})")

                    # prepare tmp dir
                    if os.path.exists(TMP_DIR):
                        shutil.rmtree(TMP_DIR)
                    os.makedirs(TMP_DIR, exist_ok=True)

                    for fn in slice_frames:
                        os.symlink(os.path.join(img_dir, fn),
                                   os.path.join(TMP_DIR, fn))

                    run_sequence(predictor, TMP_DIR, slice_frames, offset=frame_idx, writer=f)

                    shutil.rmtree(TMP_DIR)
                    frame_idx += SLIDE_SIZE
                    slice_idx += 1

        print(f"Saved results to {out_file}")

if __name__ == "__main__":
    device = torch.device("cuda")

    # checkpoints
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    det_model = YOLO('checkpoints/yolov8n-pose.pt')

    # DanceTrack root
    dancetrack_root = "/home/seraco/Project/data/MOT/dancetrack"
    split = "test"  # change to "train" or "test" as needed
    split_dir = os.path.join(dancetrack_root, split)

    MAX_DIRECT = 1400  # threshold for direct processing
    SLIDE_SIZE = 1300  # slice length
    TMP_DIR = "tmp_chunk"  # temporary folder for sliding mode

    main()