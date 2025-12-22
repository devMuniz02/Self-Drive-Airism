import airsim
import re

# --- Connect to AirSim ---
client = airsim.VehicleClient()
client.confirmConnection()

# --- 1) List all scene objects ---
print("Listing all scene objects...")
object_names = client.simListSceneObjects()
print(f"Found {len(object_names)} objects")

# --- 2) For each object, get its segmentation ID ---
name_id_list = []
for name in object_names:
    obj_id = client.simGetSegmentationObjectID(name)
    name_id_list.append((name, obj_id))

# --- 3) Remove numbers from names and collect unique base names ---
clean_names = []
for name, obj_id in name_id_list:
    # Remove all digits from the name
    clean_name = re.sub(r'\d+', '', name)
    clean_names.append(clean_name)

unique_clean_names = sorted(set(clean_names))

# --- 4) Print detailed mapping and unique names ---
print("\n=== Object Name → Segmentation ID ===")
for name, obj_id in name_id_list:
    print(f"{name:40s}  ID: {obj_id}")

print("\n=== Unique Object Names (numbers removed) ===")
for cname in unique_clean_names:
    print(cname)

# --- 5) NEW: List only objects that contain 'Road' in the name ---
road_name_id_list = [
    (name, obj_id)
    for name, obj_id in name_id_list
    if re.search(r"Road", name, re.IGNORECASE)
]

print("\n=== Objects whose name contains 'Road' (case-insensitive) ===")
if not road_name_id_list:
    print("No objects with 'Road' in the name were found.")
else:
    for name, obj_id in road_name_id_list:
        print(f"{name:40s}  ID: {obj_id}")

    # Optionally, also show unique cleaned base names just for Road objects
    road_clean_names = [
        re.sub(r'\d+', '', name) for name, _ in road_name_id_list
    ]
    unique_road_clean_names = sorted(set(road_clean_names))

    print("\n=== Unique 'Road*' base names (numbers removed) ===")
    for cname in unique_road_clean_names:
        print(cname)
