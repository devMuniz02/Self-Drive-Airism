import airsim
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Connect to AirSim ---
client = airsim.VehicleClient()
client.confirmConnection()

airsim.wait_key('Press any key to assign IDs (Roads 1..N, others 0)')

# === 1) List all scene objects ===
all_objects = client.simListSceneObjects()
print(f"Found {len(all_objects)} objects in the scene")

# === 2) Split into Road vs non-Road (case-insensitive) ===
road_objects = []
non_road_objects = []

for name in all_objects:
    if "road" in name.lower():
        road_objects.append(name)
    else:
        non_road_objects.append(name)

print(f"Found {len(road_objects)} objects containing 'Road'")
print(f"Found {len(non_road_objects)} non-Road objects")

# === 3) Set ALL non-Road objects to ID 0 ===
for name in non_road_objects:
    client.simSetSegmentationObjectID(name, 0, False)

# === 4) Assign IDs 1..N to each Road object ===
# Sort for deterministic order
road_objects_sorted = sorted(road_objects)

for idx, name in enumerate(road_objects_sorted, start=1):
    seg_id = idx  # IDs 1,2,3,...
    client.simSetSegmentationObjectID(name, seg_id, False)
    print(f"Set {name} -> ID {seg_id}")

print("Done assigning segmentation IDs.")

airsim.wait_key('Press any key to capture images')

# --- Get ORIGINAL (Scene) + SEGMENTATION from front camera ("0") ---
responses = client.simGetImages([
    # original RGB scene
    airsim.ImageRequest("0", airsim.ImageType.Scene,       False, False),
    # segmentation image (RGB where ID is encoded mainly in R channel)
    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
])

print("Retrieved images:", len(responses))

# --- Convert responses to numpy arrays ---
# Scene (original)
resp_scene = responses[0]
img1d_scene = np.frombuffer(resp_scene.image_data_uint8, dtype=np.uint8)
img_scene = img1d_scene.reshape(resp_scene.height, resp_scene.width, 3)

# Segmentation
resp_seg = responses[1]
img1d_seg = np.frombuffer(resp_seg.image_data_uint8, dtype=np.uint8)
img_seg_rgb = img1d_seg.reshape(resp_seg.height, resp_seg.width, 3)

# AirSim pone el ID en el canal R (R=ID, G=B=0 normalmente)
seg_ids = img_seg_rgb[:, :, 0].astype(np.int32)

# --- Info de qué IDs hay en la imagen ---
unique_ids, counts = np.unique(seg_ids, return_counts=True)
print("Unique segmentation IDs and counts (0 = non-Road, 1..N = different roads):")
for uid, c in zip(unique_ids, counts):
    print(f"ID {uid}: {c} pixels")

# --- Show ORIGINAL + SEGMENTED with COLORBAR ---
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Original RGB (frontal view)
axs[0].imshow(cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Scene (Frontal)")
axs[0].axis("off")

# Segmentation IDs (single-channel)
im = axs[1].imshow(seg_ids, cmap="tab20", vmin=0, vmax=255)
axs[1].set_title("Segmentation IDs (Roads = 1..N, others = 0)")
axs[1].axis("off")

# Add colorbar showing the ID values
cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
cbar.set_label("Segmentation ID")

plt.tight_layout()
plt.show()
