import airsim
import numpy as np
import cv2
import time

# --- Configuration Mappings ---
CAMERA_MAP = {
    "frontal": "0",   # Frontal Camera ID
    "driver": "3",    # Driver's POV Camera ID
    "reverse": "4",   # Back/Reverse Camera ID
}
TYPE_MAP = {
    # Using compression for better FPS
    "original": (airsim.ImageType.Scene, False, True), 
    "depth": (airsim.ImageType.DepthPlanar, True, False),
    "segmented": (airsim.ImageType.Segmentation, False, False)
}

# --- Image Processing Function (Handles Compression) ---
def get_and_process_images(client, selected_cameras, selected_type_tuple):
    """
    Retrieves the specified single image type from all selected cameras.
    NOW HANDLES COMPRESSED IMAGES.
    """
    airsim_type, is_float, is_compressed = selected_type_tuple
    
    requests = []
    for camera_friendly_name in selected_cameras:
        camera_id = CAMERA_MAP[camera_friendly_name]
        requests.append(
            airsim.ImageRequest(camera_id, airsim_type, is_float, is_compressed)
        )

    responses = client.simGetImages(requests)
    
    processed_images = []
    for response in responses:
        img_to_show = None 
        
        if is_compressed:
            try:
                img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_to_show = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decoding compressed image: {e}")
        
        elif response.image_data_uint8 and not is_float:
            img_width, img_height = response.width, response.height
            img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(img_height, img_width, 3)
            img_to_show = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        elif response.image_data_float and is_float:
            depth_data = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
            max_dist = 100.0 
            depth_data[depth_data > max_dist] = max_dist 
            depth_normalized = (depth_data / max_dist) * 255.0
            depth_normalized = depth_normalized.astype(np.uint8)
            img_to_show = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        if img_to_show is None:
            img_to_show = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(img_to_show, "No Image", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        processed_images.append(img_to_show)
        
    return processed_images

# --- Dashboard Function (Simplified) ---
def create_dashboard(processed_images, selected_cameras, selected_type, car_state):
    """
    Arranges the processed images into a single horizontal row with labels
    and adds a header with car state.
    """
    if not processed_images:
        return None

    labeled_img = processed_images[0]
    
    if labeled_img.shape[0] < 2 or labeled_img.shape[1] < 2:
         labeled_img = np.zeros((240, 320, 3), dtype=np.uint8)
         cv2.putText(labeled_img, "No Image", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    h, w, _ = labeled_img.shape
    label_height = 30 
    
    footer = np.zeros((label_height, w, 3), dtype=np.uint8) 
    cam_name = selected_cameras[0].upper()
    cv2.putText(footer, cam_name, (w // 2 - len(cam_name)*7, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    final_dashboard = np.vstack([labeled_img, footer])
    
    header_height = 40
    header = np.zeros((header_height, final_dashboard.shape[1], 3), dtype=np.uint8) 
    
    # Show image type
    header_text = f"Type: {selected_type.upper()}"
    cv2.putText(header, header_text, (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show "VIEW ONLY" status
    status_text = "VIEW ONLY"
    status_color = (0, 255, 255) # Yellow
    (text_width_control, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(header, status_text, (final_dashboard.shape[1] // 2 - text_width_control // 2, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show speed and gear
    speed_mph = car_state.speed * 2.23694 
    gear = car_state.gear
    handbrake = car_state.handbrake
    stats_text = f"Speed: {speed_mph:.0f} MPH | Gear: {gear}"
    
    (text_width_stats, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = final_dashboard.shape[1] - text_width_stats - 20
    
    color = (0, 0, 255) if handbrake else (255, 255, 255)
    cv2.putText(header, stats_text, (text_x, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if handbrake:
         cv2.putText(header, "HBK", (text_x - 50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    dashboard_with_labels = np.vstack([header, final_dashboard])
    
    return dashboard_with_labels


# --- Main Execution (Simplified) ---
if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- Hardcoded settings ---
    selected_cameras = ["frontal"]
    selected_type = "segmented"  # Options: "original", "depth", "segmented"
    selected_type_tuple = TYPE_MAP[selected_type]
    
    try:
        client = airsim.CarClient() 
        client.confirmConnection()
        # --- NOTE: client.enableApiControl(True) is REMOVED ---
        print("Connected to AirSim in VIEW-ONLY mode.")
        print("Press 'Q' in the window to quit.")
        
    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running in Car mode: {e}")
        exit()

    WINDOW_NAME = f"AirSim Dashboard - {selected_type.upper()}"

    while True:
        try:
            # --- NO COLLISION OR CONTROL LOGIC ---

            # Get the pressed key *only* to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # --- Data Gathering and Display ---
            car_state = client.getCarState()
            images = get_and_process_images(client, selected_cameras, selected_type_tuple)
            dashboard = create_dashboard(images, selected_cameras, selected_type, car_state)
            
            if dashboard is not None:
                cv2.imshow(WINDOW_NAME, dashboard)
            
            time.sleep(0.01) 
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            if "Connection was closed" in str(e):
                print("Connection lost. Attempting to reconnect...")
                try:
                    client.confirmConnection()
                    # --- NOTE: No enableApiControl() here ---
                    print("Reconnected!")
                except Exception as recon_e:
                    print(f"Reconnect failed: {recon_e}. Exiting.")
                    break
            else:
                print(f"An error occurred in the loop: {e}")
                break 

    # --- Clean up ---
    # --- NOTE: No enableApiControl(False) here ---
    cv2.destroyAllWindows()
    print("Exited.")