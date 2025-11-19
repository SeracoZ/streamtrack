import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda")

from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/seraco/Project/data/MOT/DanceTrack/train1/dancetrack0001/img1"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
img_path = os.path.join(video_dir, frame_names[frame_idx])
print("Loading:", img_path)  # Debug

plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(img_path))
plt.axis("off")   # optional, hides axes
plt.show()


det_model = YOLO('../checkpoints/yolov8x.pt')
det_results = det_model(img_path)

for result in det_results:
    # result.plot() returns an image (numpy array with bboxes + labels drawn)
    im_plot = result.plot()

    plt.figure(figsize=(10, 8))
    plt.title(f"Detection Result - frame {frame_idx}")
    plt.imshow(im_plot)
    plt.axis("off")
    plt.show()

inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)

bboxes = [
    [553, 412, 719, 628],
    [220, 374, 349, 572],
    [931, 378, 1072, 584],
    [728, 423, 899, 596],
    [734, 353, 876, 549],
    [411, 403, 601, 602],
    [350, 363, 496, 549],
]

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

all_out_obj_ids = []
all_out_masks = []

# Loop through bboxes and add them
for obj_id, box in enumerate(bboxes, start=1):  # obj_id starts from 1
    box_np = np.array(box, dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,   # unique id for each object
        box=box_np,
    )
    all_out_obj_ids.extend(out_obj_ids)
    all_out_masks.append((out_mask_logits[0] > 0.0).cpu().numpy())

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
'''
box = np.array([553, 412, 719, 629], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)


# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.axis("off")
plt.show()
'''

# Visualize results on the first frame
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title(f"frame {ann_frame_idx}")
ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))


for obj_id, box, mask in zip(all_out_obj_ids, bboxes, all_out_masks):
    show_box(box, ax)
    show_mask(mask, ax, obj_id=obj_id)

ax.axis("off")
plt.show()

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    #plt.figure(figsize=(6, 4))
    #plt.title(f"frame {out_frame_idx}")
    #plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"frame {out_frame_idx}")
    img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
    ax.imshow(img)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        show_mask(out_mask, ax, obj_id=out_obj_id)

    ax.axis("off")
    plt.show()