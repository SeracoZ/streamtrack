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
from add import add_new_objects
from occ import bbox_iou, occlusion_check
from kf import KFTrackerManager
from vis import (visualize_tracking, is_full_body, filter_overlapping_bboxes, adjust_box_to_pose,
                 full_visualize_tracking)


kf_manager = KFTrackerManager()


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
        if int(classes[i]) != 0 or scores[i] < 0.2:
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

    #bboxes = filter_overlapping_bboxes(bboxes, contain_thresh=0.9)

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
    save_mask = True
    if save_mask:
        mask_save_dir = os.path.join("mask_logs", os.path.basename(video_path))
        os.makedirs(mask_save_dir, exist_ok=True)

    # === MAIN PROPAGATION LOOP ===
    with torch.no_grad():
        for rel_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            frame_path = os.path.join(video_path, frame_names[rel_idx])
            frame = cv2.imread(frame_path)

            # YOLO detect every frame
            results = det_model(frame_path, verbose=False)[0]

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

            # === CLEANUP PHASE ===
            DISAPPEAR_THRESH = 30

            if "last_seen" not in inference_state:
                inference_state["last_seen"] = {obj_id: rel_idx for obj_id in out_obj_ids}
            if "pending_remove" not in inference_state:
                inference_state["pending_remove"] = set()

            # Update last seen
            for obj_id in out_obj_ids:
                inference_state["last_seen"][obj_id] = rel_idx

            # Detect disappeared objects
            for obj_id, last_f in list(inference_state["last_seen"].items()):
                if rel_idx - last_f > DISAPPEAR_THRESH:
                    inference_state["pending_remove"].add(obj_id)

            # REMOVAL PHASE
            RETIRE_THRESH = 50  # number of frames since last seen before permanent deletion
            if "last_seen" in inference_state:
                for obj_id, last_f in list(inference_state["last_seen"].items()):
                    if rel_idx - last_f > RETIRE_THRESH:
                        if not hasattr(predictor, "_pending_hard_remove"):
                            predictor._pending_hard_remove = set()
                        predictor._pending_hard_remove.add(obj_id)

            # Add Module
            existing_masks_clean = []
            for i in range(len(out_obj_ids)):
                mask = (out_mask_logits_cpu[i] > 0.0).numpy().squeeze()
                mask = clean_mask(mask)
                existing_masks_clean.append(mask)

            new_objects = add_new_objects(
                inference_state=inference_state,
                cur_boxes=cur_boxes,
                existing_boxes=existing_boxes,
                existing_masks=existing_masks_clean,
                rel_idx=rel_idx,
                seen_ids=seen_ids,
                birth_frames=3,  # ensure stable appearance
                area_ratio_thresh=0.7,  # candidate must be mostly outside SAM masks
                iou_match_thresh=0.3  # detection is an old object if IoU >= 0.3
            )

            # Add new objects to SAM2
            if new_objects:
                inference_state.setdefault("pending_new_objects", [])
                for new_id, box in new_objects:
                    tqdm.write(f"🟢 NEW PERSON {new_id} at frame {rel_idx}")
                    inference_state["pending_new_objects"].append((new_id, box))

            # write result
            current_ids = out_obj_ids
            valid_ids = set(inference_state["obj_ids"])

            keep_idx = [i for i, oid in enumerate(current_ids) if oid in valid_ids]

            final_ids = [current_ids[i] for i in keep_idx]
            final_logits = out_mask_logits[keep_idx]

            assert len(final_ids) == final_logits.shape[0], \
                f"Mismatch after filtering: {len(final_ids)} vs {final_logits.shape[0]}"

            # === OCCLUSION REASONING (KF + MASK + DET) ========================

            # 1. Build SAM2 mask boxes (stable & robust)
            sam_boxes = {}
            mask_areas = {}
            out_masks = final_logits.detach().cpu()

            for idx, tid in enumerate(final_ids):
                mask = (out_masks[idx] > 0).numpy().squeeze()
                ys, xs = np.where(mask)

                if len(xs) == 0 or len(ys) == 0:
                    sam_boxes[tid] = None
                    mask_areas[tid] = 0
                    continue

                # Use tight bounding box, MUCH more reliable than percentiles
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                sam_boxes[tid] = [float(x1), float(y1), float(x2), float(y2)]
                mask_areas[tid] = int(mask.sum())

            # 2. Build YOLO detection dict (track_id → xyxy)
            det_dict = {}

            # cur_boxes is already xyxy in your code (double-checked)
            for idx, tid in enumerate(final_ids):
                if idx < len(cur_boxes):
                    x1, y1, x2, y2 = cur_boxes[idx]
                    det_dict[tid] = [x1, y1, x2, y2]

            # 3. KF update/predict (only once per ID per frame)
            KF_boxes = {}

            for tid in final_ids:
                if tid in det_dict:
                    KF_boxes[tid] = kf_manager.update_with_det(tid, det_dict[tid])
                else:
                    KF_boxes[tid] = kf_manager.predict(tid)

            # 4. Collect YOLO boxes for occlusion logic
            YOLO_boxes = list(det_dict.values())

            # 5. Call occlusion logic
            occlusion_check(
                predictor=predictor,
                inference_state=inference_state,
                frame_idx=rel_idx,
                track_ids=final_ids,
                track_logits=final_logits,
                KF_boxes=KF_boxes,
                YOLO_boxes=YOLO_boxes,
                shrink_thr=0.95,
                iou_thr=0.30,
                miss_det_thr=2,
            )

            for idx, tid in enumerate(final_ids):
                mask = (final_logits[idx] > 0).cpu().numpy().squeeze()
                y, x = np.where(mask)
                if len(x) == 0: continue
                x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                w, h = x2 - x1, y2 - y1
                writer.write(f"{rel_idx + 1},{tid},{x1},{y1},{w},{h},1,-1,-1,-1\n")

            '''
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
            '''

            # --- Save visualization frame only (no display) ---
            VIS_FULL = False
            if save_vis:
                save_path = os.path.join(vis_dir, f"{rel_idx:06d}.jpg")
                if VIS_FULL:
                    full_visualize_tracking(
                        frame,
                        final_ids,
                        final_logits,
                        id_colors,
                        det_boxes=cur_boxes,  # pose-adjusted
                        det_boxes_original=boxes_f,  # raw YOLO output
                        save_path=save_path
                    )
                else:
                    visualize_tracking(frame, final_ids, final_logits, id_colors,
                                       save_path=save_path)
                #visualize_tracking(frame, out_obj_ids, out_mask_logits, id_colors,
                                   #save_path=save_path)
            # === SAVE MASK IMAGES PER ID (ONE FOLDER PER ID) ===
            for idx, tid in enumerate(final_ids):
                # 1. ID folder
                id_dir = os.path.join(mask_save_dir, f"id_{tid}")
                os.makedirs(id_dir, exist_ok=True)

                # 2. Get mask as uint8
                mask = (final_logits[idx] > 0).cpu().numpy().squeeze().astype(np.uint8)

                # 3. Convert to 0/255 image for visualization
                mask_img = (mask * 255).astype(np.uint8)

                # 4. Save image
                save_path = os.path.join(id_dir, f"frame_{rel_idx:06d}.png")
                cv2.imwrite(save_path, mask_img)

            torch.cuda.empty_cache()

    del inference_state
    torch.cuda.empty_cache()



def main():
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    for seq in sorted(os.listdir(split_dir)):
        img_dir = os.path.join(split_dir, seq, "img1")
        if not os.path.isdir(img_dir):
            continue
        #test_list = sorted(os.listdir(split_dir))[11]
        test_list = sorted(os.listdir(split_dir))[7]
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

    # checkpoints
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    det_model = YOLO('checkpoints/yolov8x-pose-p6.pt')

    # DanceTrack root
    dancetrack_root = "/home/seraco/Project/data/MOT/dancetrack"
    split = "test_part"  # change to "train" or "test" as needed
    split_dir = os.path.join(dancetrack_root, split)

    MAX_DIRECT = 1300  # threshold for direct processing
    SLIDE_SIZE = 1200  # slice length
    TMP_DIR = "tmp_chunk"  # temporary folder for sliding mode

    main()