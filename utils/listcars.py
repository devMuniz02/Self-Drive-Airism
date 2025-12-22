import airsim
import numpy as np
import cv2
import time
import argparse
import warnings
# Note: YOLO is not needed for this diagnostic script

# --- Main Execution (MODIFIED FOR DIAGNOSTICS) ---
if __name__ == "__main__":
    
    # warnings.simplefilter('ignore', np.RankWarning) 
    
    # --- NEW: List of keywords to search for ---
    KEYWORDS = ['car', 'bus', 'ambulance', 'vehicle', 'bicycle', 'motorcycle']
    
    try:
        # --- Connect only the CarClient ---
        client = airsim.CarClient() 
        client.confirmConnection()
        print("CarClient connected!")
        
        # --- NEW: Get ALL scene objects ---
        print("Finding all scene objects...")
        
        all_object_names = client.simListSceneObjects(".*") 
        
        if not all_object_names:
            print("Could not find any objects.")
        else:
            print(f"Found {len(all_object_names)} total objects. Filtering for keywords...")
            
            # --- NEW: Process for unique names containing keywords ---
            found_vehicle_names = set()
            
            for name in all_object_names:
                name_lower = name.lower() # Check in lowercase
                for keyword in KEYWORDS:
                    if keyword in name_lower:
                        found_vehicle_names.add(name) # Add the original name
                        break # Move to the next object name
            
            if not found_vehicle_names:
                print("--- NO OBJECTS FOUND MATCHING YOUR KEYWORDS ---")
            else:
                print(f"--- FOUND {len(found_vehicle_names)} UNIQUE VEHICLE NAMES ---")
                print("Please copy this list and paste it in our chat.")
                print("-" * 30)
                
                # Print every unique name found
                for vehicle_name in sorted(found_vehicle_names): # Sort them alphabetically
                    print(vehicle_name)
                
                print("-" * 30)
                print("--- END OF LIST ---")

        # --- END NEW ---

    except Exception as e:
        print(f"Error connecting to AirSim or listing objects: {e}")
        exit()

    print("Diagnostic complete. Exiting.")
    exit()

# --- The rest of your code is below this line but will not run ---
# --- (You can leave it or delete it, it doesn't matter for this test) ---

# --- Configuration Mappings ---
CAMERA_MAP = {
    "frontal": "0",
    "driver": "3",
    "reverse": "4",
    "follow": "follow_cam"
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True),
    "depth": (airsim.ImageType.DepthPlanar, True, False),
    "segmented": (airsim.ImageType.Segmentation, False, False)
}

# (All your other functions like get_images, process_image_for_lines, etc., would be here)