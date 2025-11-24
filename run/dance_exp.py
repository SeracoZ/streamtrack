import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
import cv2

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

from filter import clean_mask
from kf import LightKalman
from occlusion import compute_occlusion_for_objects, attach_occlusion_to_inference_state
from quality import compute_quality_for_objects, attach_quality_to_inference_state
from vis import draw_score_on_frame, visualize_tracking, iou, is_full_body, filter_overlapping_bboxes, adjust_box_to_pose


def run_sequence(predictor, video_path, frame_names, writer, save_vis=True, quiet=False):
    """Run SAM2+YOLO tracking with tqdm progress and frame saving only."""
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    # --- NEW: candidate buffer for strict adding ---
    if "candidate_new" not in inference_state:
        inference_state["candidate_new"] = {}  # cid → {box, hit, last_frame}

    # --- Initial detection ---
    first_frame = os.path.join(video_path, frame_names[0])
    results = det_model(first_frame, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    kpts_xy = results.keypoints.xy.cpu().numpy()
    kpts_conf = results.keypoints.conf.cpu().numpy()

    bboxes = []
    for i in range(len(boxes)):
        if int(classes[i]) != 0 or scores[i] < 0.4:
            continue
        if not is_full_body(kpts_conf[i]):
            continue

        new_box = adjust_box_to_pose(
            boxes[i],
            kpts_xy[i],
            kpts_conf[i],
            conf_thresh=0.3
        )
        bboxes.append(new_box)

    bboxes = filter_overlapping_bboxes(bboxes, contain_thresh=0.9)

    if not bboxes:
        print("⚠️ No person detected, skipping...")
        return

    id_colors, seen_ids = {}, set()
    inference_state["kf_states"] = {}  # obj_id -> LightKalman
    inference_state["kf_pred_box"] = {}  # obj_id -> [x1,y1,x2,y2]
    inference_state["last_box"] = {}  # optional: last observed SAM2 box
    for obj_id, box in enumerate(bboxes, start=1):
        predictor.add_new_points_or_box(inference_state, frame_idx=0,
                                        obj_id=obj_id, box=np.array(box, dtype=np.float32))
        seen_ids.add(obj_id)
        inference_state["kf_states"][obj_id] = LightKalman(box)
        inference_state["kf_pred_box"][obj_id] = box
        inference_state["last_box"][obj_id] = box

    vis_dir = os.path.join("vis_results", os.path.basename(video_path)) if save_vis else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    all_quality_scores = []
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

            frame_path = os.path.join(video_path, frame_names[rel_idx])
            frame = cv2.imread(frame_path)

            if rel_idx == 30:
                print('Time to debug')
            # YOLO detect every frame
            results = det_model(frame_path, verbose=False)[0]

            # Get detection results for THIS frame
            boxes_f = results.boxes.xyxy.cpu().numpy()
            scores_f = results.boxes.conf.cpu().numpy()
            classes_f = results.boxes.cls.cpu().numpy()
            kpts_xy_f = results.keypoints.xy.cpu().numpy()
            kpts_conf_f = results.keypoints.conf.cpu().numpy()

            cur_boxes = []

            for i in range(len(boxes_f)):
                if int(classes_f[i]) != 0 or scores_f[i] < 0.4:
                    continue
                if not is_full_body(kpts_conf_f[i]):
                    continue

                # Adjust bbox to full-body extent
                new_box = adjust_box_to_pose(
                    boxes_f[i],
                    kpts_xy_f[i],
                    kpts_conf_f[i],
                    conf_thresh=0.3
                )
                cur_boxes.append(new_box)

            cur_boxes = filter_overlapping_bboxes(cur_boxes, contain_thresh=0.9)

            # === Visualize filtered boxes ===
            vis_frame = frame.copy()
            for box in cur_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow box
                cv2.putText(vis_frame, "filtered_box", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            save_dir_boxes = f"bbox_vis/{os.path.basename(video_path)}"
            os.makedirs(save_dir_boxes, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir_boxes, f"{rel_idx:06d}.jpg"), vis_frame)

            # compute existing boxes from masks
            out_mask_logits_cpu = out_mask_logits.detach().cpu()
            existing_boxes = []
            for i in range(len(out_obj_ids)):
                mask = (out_mask_logits_cpu[i] > 0.0).numpy().squeeze(0)
                mask = clean_mask(mask)
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                existing_boxes.append([x1, y1, x2, y2])

            # === KF UPDATE / PREDICT (light OC-SORT) ===
            if "kf_states" not in inference_state:
                inference_state["kf_states"] = {}
            if "kf_pred_box" not in inference_state:
                inference_state["kf_pred_box"] = {}
            if "last_box" not in inference_state:
                inference_state["last_box"] = {}

            # map obj_id -> box from SAM2 for visible ones
            visible_box_map = {}
            for i, obj_id in enumerate(out_obj_ids):
                if i < len(existing_boxes):
                    visible_box_map[obj_id] = existing_boxes[i]

            # 1) update KF with current visible boxes
            for obj_id, box in visible_box_map.items():
                if obj_id not in inference_state["kf_states"]:
                    inference_state["kf_states"][obj_id] = LightKalman(box)
                kf = inference_state["kf_states"][obj_id]
                pred_box = kf.update(box)
                inference_state["kf_pred_box"][obj_id] = pred_box
                inference_state["last_box"][obj_id] = box

            # 2) predict KF for occluded tracks
            occluded_ids = inference_state.get("occluded_ids", set())
            for obj_id in occluded_ids:
                kf = inference_state["kf_states"].get(obj_id, None)
                if kf is None:
                    # if never had KF, fall back to last_box
                    box = inference_state["last_box"].get(obj_id, None)
                    if box is not None:
                        inference_state["kf_states"][obj_id] = LightKalman(box)
                        kf = inference_state["kf_states"][obj_id]
                    else:
                        continue
                pred_box = kf.predict()
                inference_state["kf_pred_box"][obj_id] = pred_box

            # === Compute occlusion using YOLO detections ===
            occlusion_list = compute_occlusion_for_objects(
                existing_boxes=existing_boxes,
                det_boxes=cur_boxes,
                det_scores=scores_f
            )

            # === Attach occlusion info directly into SAM2's memory structure ===
            attach_occlusion_to_inference_state(
                inference_state=inference_state,
                occlusion_list=occlusion_list,
                out_obj_ids=out_obj_ids,
                rel_idx=rel_idx
            )

            # === Compute frame quality using existing + YOLO ===
            quality_list = compute_quality_for_objects(
                existing_boxes=existing_boxes,
                det_boxes=cur_boxes,
                det_scores=scores_f
            )

            for q in quality_list:
                all_quality_scores.append(q["final_quality"])

            # === Attach quality to SAM memory ===
            attach_quality_to_inference_state(
                inference_state=inference_state,
                quality_list=quality_list,
                out_obj_ids=out_obj_ids,
                rel_idx=rel_idx
            )

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

            #############################################################
            # === STRICT 3-FRAME CONFIRMATION FOR ADDING NEW OBJECTS ===
            #############################################################
            new_objects = []
            occluded_ids = inference_state.get("occluded_ids", set())

            for det_box in cur_boxes:
                # --- 1) Check if it matches an existing *visible* SAM2 box ---
                overlaps = [iou(det_box, ebox) for ebox in existing_boxes]
                matched_visible = (overlaps and max(overlaps) >= 0.3)

                if matched_visible:
                    # it already corresponds to a visible SAM2 track; let SAM2 handle it
                    continue

                # --- 2) Try to revive an occluded track using KF prediction (light OC-SORT) ---
                best_id = None
                best_iou = 0.0

                for lost_id in occluded_ids:
                    kf_box = inference_state["kf_pred_box"].get(lost_id, None)
                    if kf_box is None:
                        continue
                    i = iou(det_box, kf_box)
                    if i > best_iou:
                        best_iou = i
                        best_id = lost_id

                # OC-SORT-style gating threshold (tune 0.3–0.4 on DanceTrack)
                if best_id is not None and best_iou > 0.35:
                    # 🔁 Revive old track instead of creating a new ID
                    tqdm.write(f"[Frame {rel_idx}] 🔄 Revived occluded ID {best_id} via KF (IoU={best_iou:.2f})")

                    # mark as visible again: remove from occluded set
                    if "occluded_ids" in inference_state and best_id in inference_state["occluded_ids"]:
                        inference_state["occluded_ids"].remove(best_id)
                    # update last_seen and last_box
                    inference_state["last_seen"][best_id] = rel_idx
                    inference_state["last_box"][best_id] = det_box
                    # also gently guide SAM2 by adding a box prompt once (optional but helpful)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=rel_idx,
                        obj_id=best_id,
                        box=np.array(det_box, dtype=np.float32),
                    )
                    # Don't create a new ID for this det_box
                    continue

                # --- 3) If no occluded track matches well, create a new one as before ---
                new_id = max(seen_ids) + 1 if seen_ids else 1
                tqdm.write(f"🟢 Queuing new object {new_id} at frame {rel_idx}")

                if "pending_new_objects" not in inference_state:
                    inference_state["pending_new_objects"] = []
                inference_state["pending_new_objects"].append((new_id, det_box))
                new_objects.append(new_id)
                seen_ids.add(new_id)

                # init KF for the new ID as well
                inference_state["kf_states"][new_id] = LightKalman(det_box)
                inference_state["kf_pred_box"][new_id] = det_box
                inference_state["last_box"][new_id] = det_box

            # === VISUALIZE SCORE (if enabled) ===
            if SHOW_SCORE:
                for idx, obj_id in enumerate(out_obj_ids):
                    if idx >= len(existing_boxes):
                        continue

                    box = existing_boxes[idx]
                    occ_info = occlusion_list[idx] if idx < len(occlusion_list) else None
                    qual_info = quality_list[idx] if idx < len(quality_list) else None

                    draw_score_on_frame(frame, box, obj_id, occ_info, qual_info)

            # --- Write results ---
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits_cpu[i] > 0.0).numpy().squeeze(0)
                y, x = np.where(mask)
                if len(x) == 0 or len(y) == 0:
                    continue
                x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                w, h = x2 - x1, y2 - y1
                line = f"{rel_idx+1},{out_obj_id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
                writer.write(line)

            # --- Save visualization frame only (no display) ---
            if save_vis:
                save_path = os.path.join(vis_dir, f"{rel_idx:06d}.jpg")
                visualize_tracking(frame, out_obj_ids, out_mask_logits, id_colors,
                                   save_path=save_path)

            torch.cuda.empty_cache()

    del inference_state
    torch.cuda.empty_cache()

    # === DRAW QUALITY SCORE DISTRIBUTION ===
    if SHOW_SCORE and len(all_quality_scores) > 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(all_quality_scores, bins=30, color='steelblue', edgecolor='black')
        plt.title("Quality Score Distribution")
        plt.xlabel("quality score")
        plt.ylabel("count")

        os.makedirs("score_distribution", exist_ok=True)
        plt.savefig(f"score_distribution/{os.path.basename(video_path)}_qual_hist.png")
        plt.close()


def main():
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    for seq in sorted(os.listdir(split_dir)):
        img_dir = os.path.join(split_dir, seq, "img1")
        if not os.path.isdir(img_dir):
            continue
        #test_list = sorted(os.listdir(split_dir))[11]
        test_list = sorted(os.listdir(split_dir))[1]
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
                run_sequence(predictor, img_dir, frame_names, writer=f, save_vis=True, quiet=False)

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
    SHOW_SCORE = False  # or False to disable all quality/occlusion visualization

    # checkpoints
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    det_model = YOLO('checkpoints/yolov8x-pose.pt')

    # DanceTrack root
    dancetrack_root = "/home/seraco/Project/data/MOT/dancetrack"
    split = "exp"  # change to "train" or "test" as needed
    split_dir = os.path.join(dancetrack_root, split)

    MAX_DIRECT = 1300  # threshold for direct processing
    SLIDE_SIZE = 1200  # slice length
    TMP_DIR = "tmp_chunk"  # temporary folder for sliding mode

    main()