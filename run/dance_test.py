import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

device = torch.device("cuda")

# checkpoints
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
det_model = YOLO('checkpoints/yolov8x.pt')

# DanceTrack root (modify this!)
dancetrack_root = "/home/seraco/Project/data/MOT/dancetrack"
split = "test"   # change to "train" or "test" as needed
split_dir = os.path.join(dancetrack_root, split)

for seq in sorted(os.listdir(split_dir)):
    img_dir = os.path.join(split_dir, seq, "img1")
    if not os.path.isdir(img_dir):
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
        # init predictor on full sequence
        inference_state = predictor.init_state(video_path=img_dir)
        predictor.reset_state(inference_state)

        # detect persons on first frame
        first_frame = os.path.join(img_dir, frame_names[0])
        det_results = det_model(first_frame)
        bboxes = []
        for r in det_results:
            for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(cls) == 0:  # 0 = person
                    bboxes.append(box.tolist())

        if len(bboxes) == 0:
            print(f"⚠️  No person detected in {seq}, skipping...")
            continue

        # add seed detections
        for obj_id, box in enumerate(bboxes, start=1):
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=np.array(box, dtype=np.float32),
            )

        # propagate across the full sequence
        with torch.no_grad():
            for rel_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                abs_idx = rel_idx  # already full sequence index
                if abs_idx >= len(frame_names):
                    break

                out_mask_logits = out_mask_logits.detach().cpu()
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).numpy().squeeze(0)
                    if mask.ndim != 2:
                        continue
                    y, x = np.where(mask)
                    if len(x) == 0 or len(y) == 0:
                        continue
                    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                    w, h = x2 - x1, y2 - y1
                    line = f"{abs_idx+1},{out_obj_id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
                    f.write(line)

                del out_mask_logits
                torch.cuda.empty_cache()

        # cleanup
        del inference_state
        torch.cuda.empty_cache()

    print(f"Saved results to {out_file}")
