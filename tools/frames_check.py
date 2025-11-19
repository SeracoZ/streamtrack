import os

dancetrack_root = "/home/seraco/Project/data/MOT/dancetrack"
split = "test"   # change to "train" or "test" as needed
split_dir = os.path.join(dancetrack_root, split)

frame_counts = []

for subfolder in os.listdir(split_dir):
    sub_path = os.path.join(split_dir, subfolder, "img1")  # DanceTrack stores frames in "img1"
    if not os.path.isdir(sub_path):
        continue

    # Count number of frames (jpg files)
    frame_count = len([f for f in os.listdir(sub_path) if f.endswith(".jpg")])
    frame_counts.append((subfolder, frame_count))

# Sort by frame count (ascending)
frame_counts.sort(key=lambda x: x[1])

print("5 subfolders with the least frames:")
for subfolder, count in frame_counts[:5]:
    print(f"{subfolder}: {count} frames")
