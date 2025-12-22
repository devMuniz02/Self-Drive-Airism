import airsim
import numpy as np
import cv2
import time
import argparse
import warnings
from ultralytics import YOLO  # For car detection

# --- Configuration Mappings (Unchanged) ---
CAMERA_MAP = {
    "frontal": "0", "driver": "3", "reverse": "4", "follow": "follow_cam"
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True),
    "segmented": (airsim.ImageType.Segmentation, False, False)
}

# --- Image Processing Function (Unchanged) ---
def get_images(client):
    cam_names = ["frontal", "driver", "reverse", "follow", "segmented"]
    requests = []
    
    # Add requests for original cameras
    for cam_name in cam_names[:4]:
        camera_id = CAMERA_MAP[cam_name]
        airsim_type, is_float, is_compressed = TYPE_MAP["original"]
        requests.append(airsim.ImageRequest(camera_id, airsim_type, is_float, is_compressed))
    
    # Add request for Segmented view
    seg_req = airsim.ImageRequest(
        CAMERA_MAP["frontal"],
        TYPE_MAP["segmented"][0],
        TYPE_MAP["segmented"][1],
        TYPE_MAP["segmented"][2]
    )
    requests.append(seg_req)

    responses = client.simGetImages(requests)
    images = {}
    
    # Process original 4 images
    for i, cam_name in enumerate(cam_names[:4]):
        response = responses[i]
        img_to_show = None
        if response.image_data_uint8:
            try:
                img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_to_show = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decoding {cam_name} image: {e}")
        if img_to_show is None:
            img_to_show = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img_to_show, f"{cam_name} Image Error", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        images[cam_name] = img_to_show
        
    # Process 5th response (segmented)
    seg_img_bgr = None
    try:
        response = responses[4]
        if response.image_data_uint8 and response.width > 0 and response.height > 0:
            img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # Fix: Check if it's 3 or 4 channel
            channels = 3
            if (response.height * response.width * 4 == len(img_1d)):
                channels = 4
                
            if channels == 4:
                img_rgba = img_1d.reshape(response.height, response.width, 4)
                seg_img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            else:
                img_rgb = img_1d.reshape(response.height, response.width, 3)
                seg_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            raise Exception("Received empty segmented image")
    except Exception as e:
        print(f"Error processing segmented image: {e}")
        seg_img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(seg_img_bgr, "Segmented Error", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    images["segmented"] = seg_img_bgr
        
    return images

# -----------------------------------------------------------------
# --- 🤖 3-STAGE (RGB -> HSV -> GRAY) ROAD DETECTION 🤖 ---
# -----------------------------------------------------------------

def calculate_roi_points(height, width, top_y_p, top_x_p, top_w_p,
                         bottom_y_p, bottom_x_p, bottom_w_p):
    """Calculates the ROI polygon vertices (Unchanged)."""
    top_y = height * (top_y_p / 100.0)
    top_x_center = width * (top_x_p / 100.0)
    top_width = width * (top_w_p / 100.0)

    bottom_y = height * (bottom_y_p / 100.0)
    bottom_x_center = width * (bottom_x_p / 100.0)
    bottom_width = width * (bottom_w_p / 100.0)

    top_left_x = top_x_center - (top_width / 2)
    top_right_x = top_x_center + (top_width / 2)
    bottom_left_x = bottom_x_center - (bottom_width / 2)
    bottom_right_x = bottom_x_center + (bottom_width / 2)

    vertices_for_poly = np.array(
        [[(bottom_left_x, bottom_y),
          (top_left_x, top_y),
          (top_right_x, top_y),
          (bottom_right_x, bottom_y)]],
        dtype=np.int32
    )
    vertices_for_warp = np.float32([
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bottom_left_x, bottom_y),
        (bottom_right_x, bottom_y)
    ])
    return vertices_for_poly, vertices_for_warp

def draw_roi(image, vertices):
    """Draws the ROI trapezoid on an image (Unchanged)."""
    img_copy = image.copy()
    cv2.polylines(img_copy, vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    return img_copy

# --- 3-Stage Filter Function (Unchanged) ---
def process_image_3_stage_filter(img, roi_vertices, blur_kernel_size, 
                                 rgb_r_low, rgb_r_high, rgb_g_low, rgb_g_high,
                                 rgb_b_low, rgb_b_high,
                                 hsv_h_low, hsv_h_high, hsv_s_low, hsv_s_high,
                                 hsv_v_low, hsv_v_high,
                                 gray_low_thresh, gray_high_thresh):
    """
    Applies a 3-stage filter (RGB -> HSV -> Grayscale) to find the road surface.
    """
    # Stage 1: RGB/BGR threshold
    bgr_low = np.array([rgb_b_low, rgb_g_low, rgb_r_low])
    bgr_high = np.array([rgb_b_high, rgb_g_high, rgb_r_high])
    rgb_mask = cv2.inRange(img, bgr_low, bgr_high)
    rgb_filtered_img = cv2.bitwise_and(img, img, mask=rgb_mask)

    # Stage 2: HSV threshold
    hsv_img = cv2.cvtColor(rgb_filtered_img, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([hsv_h_low, hsv_s_low, hsv_v_low])
    hsv_high = np.array([hsv_h_high, hsv_s_high, hsv_v_high])
    hsv_mask = cv2.inRange(hsv_img, hsv_low, hsv_high)
    hsv_filtered_debug = cv2.bitwise_and(rgb_filtered_img, rgb_filtered_img, mask=hsv_mask)

    # Stage 3: Grayscale + blur + threshold
    gray_from_filtered = cv2.cvtColor(hsv_filtered_debug, cv2.COLOR_BGR2GRAY)
    blurry_gray = cv2.GaussianBlur(gray_from_filtered, (blur_kernel_size, blur_kernel_size), 0)
    final_road_mask = cv2.inRange(blurry_gray, gray_low_thresh, gray_high_thresh)
    
    height, width = final_road_mask.shape
    roi_mask = np.zeros_like(final_road_mask)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    
    final_road_mask_for_tracker = cv2.bitwise_and(final_road_mask, roi_mask)
    
    return final_road_mask_for_tracker, hsv_filtered_debug, final_road_mask_for_tracker

# --- Blob Centering Function (Unchanged) ---
def calculate_steering_from_blob(warped_mask):
    """
    Calculates the steering offset by finding the center
    of the road blob at a "look-ahead" point.
    """
    h, w = warped_mask.shape
    y_eval = int(h * 0.75) 
    look_ahead_row = warped_mask[y_eval, :]
    
    indices = look_ahead_row.nonzero()[0]
    
    if len(indices) > 0:
        lane_center = np.mean(indices)
        car_center = w / 2
        offset = car_center - lane_center
        return offset, (int(lane_center), y_eval)
    else:
        histogram = np.sum(warped_mask[h//2:, :], axis=0)
        indices = histogram.nonzero()[0]
        if len(indices) > 0:
            lane_center = np.mean(indices)
            car_center = w / 2
            offset = car_center - lane_center
            return offset, None 
        
        return 0.0, None

def perspective_warp(img, src_points):
    height, width = img.shape[:2]
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix_inv

# --- Visualization Function (Draws Red Dot) ---
def draw_center_visuals(original_img, warped_mask, center_point, matrix_inv):
    """
    Draws a red dot at the calculated center point.
    """
    if center_point is None:
        return original_img 

    h, w = warped_mask.shape
    warp_zero = np.zeros_like(warped_mask).astype(np.uint8)
    color_warp = cv2.cvtColor(warp_zero, cv2.COLOR_GRAY2BGR)

    cv2.circle(color_warp, center_point, 10, (0, 0, 255), -1)

    unwarped_overlay = cv2.warpPerspective(color_warp, matrix_inv, (w, h))
    result = cv2.addWeighted(original_img, 1, unwarped_overlay, 0.8, 0)
    return result

def draw_steering_lines(image, steering_value):
    if steering_value is None:
        return image
    image_copy = image.copy()
    h, w, _ = image_copy.shape

    start_left = (w // 2 - 40, h - 20)
    start_right = (w // 2 + 40, h - 20)
    line_length = h // 4
    max_angle_deg = 30

    angle_rad = steering_value * np.deg2rad(max_angle_deg)
    end_y = int(start_left[1] - line_length * np.cos(angle_rad))
    end_x_offset = int(line_length * np.sin(angle_rad))

    end_left_x = start_left[0] + end_x_offset
    end_right_x = start_right[0] + end_x_offset

    cv2.line(image_copy, start_left, (end_left_x, end_y), (0, 255, 255), 3)
    cv2.line(image_copy, start_right, (end_right_x, end_y), (0, 255, 255), 3)
    return image_copy

# --- Dashboard Function (Unchanged) ---
def create_dashboard(frontal_img, driver_img, reverse_img, follow_img, 
                     segmented_debug, final_road_mask_debug,
                     car_state, control_mode, steering_value=None):
    
    STD_HEIGHT = 200
    STD_WIDTH = 250  
    
    try:
        front_resized = cv2.resize(frontal_img, (STD_WIDTH, STD_HEIGHT))
        driver_resized = cv2.resize(driver_img, (STD_WIDTH, STD_HEIGHT))
        reverse_resized = cv2.resize(reverse_img, (STD_WIDTH, STD_HEIGHT))
        follow_resized = cv2.resize(follow_img, (STD_WIDTH, STD_HEIGHT))
        segmented_resized = cv2.resize(segmented_debug, (STD_WIDTH, STD_HEIGHT))
        road_mask_resized = cv2.resize(final_road_mask_debug, (STD_WIDTH, STD_HEIGHT))
        road_mask_resized = cv2.cvtColor(road_mask_resized, cv2.COLOR_GRAY2BGR)

    except Exception as e:
        print(f"Error resizing images: {e}")
        blank_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        front_resized = driver_resized = reverse_resized = segmented_resized = road_mask_resized = follow_resized = blank_img
    
    cv2.putText(driver_resized, "DRIVER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(front_resized, "FRONTAL (ROI & Target)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(segmented_resized, "SEGMENTED (AirSim)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(road_mask_resized, "FINAL ROAD MASK", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(reverse_resized, "REVERSE", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(follow_resized, "FOLLOW (YOLO)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined_image_row1 = np.hstack([driver_resized, front_resized, segmented_resized])
    combined_image_row2 = np.hstack([road_mask_resized, reverse_resized, follow_resized])
    combined_image = np.vstack([combined_image_row1, combined_image_row2]) 

    # --- Footer (Unchanged) ---
    h, w, _ = combined_image.shape 
    footer_height = 100
    footer = np.zeros((footer_height, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20
    padding = 15

    mode_text = f"MODE: {control_mode.upper()} (Press 'K' to toggle)"
    mode_color = (0, 255, 0) if control_mode == 'auto' else (0, 255, 255)
    cv2.putText(footer, mode_text, (padding, padding + line_height * 0),
                font, font_scale, mode_color, font_thickness)

    speed_mph = car_state.speed * 2.23694 
    speed_text = f"Speed: {speed_mph:.0f} MPH"
    cv2.putText(footer, speed_text, (padding, padding + line_height * 1),
                font, font_scale, (255, 255, 255), font_thickness)

    gear_text = f"Gear: {car_state.gear}"
    cv2.putText(footer, gear_text, (padding, padding + line_height * 2),
                font, font_scale, (255, 255, 255), font_thickness)

    if steering_value is not None:
        steer_text = f"Steer: {steering_value:.2f}"
        steer_color = (0, 0, 255)
        cv2.putText(footer, steer_text, (padding, padding + line_height * 3),
                    font, font_scale, steer_color, font_thickness)

    if car_state.handbrake:
        hb_text = "HANDBRAKE ON"
        (text_width, _), _ = cv2.getTextSize(hb_text, font, font_scale, font_thickness)
        hb_x = w - text_width - padding
        cv2.putText(footer, hb_text, (hb_x, padding + line_height * 0),
                    font, font_scale, (0, 0, 255), font_thickness)

    dashboard = np.vstack([combined_image, footer])
    
    return dashboard

# --- *** NEW: Color Preview UI Function *** ---
def create_color_preview_ui(h, w, 
                            rgb_r_low, rgb_r_high, rgb_g_low, rgb_g_high,
                            rgb_b_low, rgb_b_high,
                            hsv_h_low, hsv_h_high, hsv_s_low, hsv_s_high,
                            hsv_v_low, hsv_v_high,
                            gray_low_thresh, gray_high_thresh):
    """
    Creates a separate UI image to visualize the slider color selections.
    """
    ui_image = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- RGB ---
    cv2.putText(ui_image, "RGB Filter", (10, 30), font, 0.7, (255, 255, 255), 2)
    # Low Color
    rgb_low_color = (rgb_b_low, rgb_g_low, rgb_r_low)
    cv2.rectangle(ui_image, (10, 40), (w//2 - 10, 100), rgb_low_color, -1)
    cv2.putText(ui_image, "LOW", (15, 60), font, 0.5, (255,255,255), 1)
    # High Color
    rgb_high_color = (rgb_b_high, rgb_g_high, rgb_r_high)
    cv2.rectangle(ui_image, (w//2 + 10, 40), (w - 10, 100), rgb_high_color, -1)
    cv2.putText(ui_image, "HIGH", (w//2 + 15, 60), font, 0.5, (0,0,0), 1)

    # --- HSV ---
    cv2.putText(ui_image, "HSV Filter", (10, 140), font, 0.7, (255, 255, 255), 2)
    # Low Color
    hsv_low_color = np.uint8([[[hsv_h_low, hsv_s_low, hsv_v_low]]])
    bgr_low_color = cv2.cvtColor(hsv_low_color, cv2.COLOR_HSV2BGR)[0][0]
    cv2.rectangle(ui_image, (10, 150), (w//2 - 10, 210),
                  (int(bgr_low_color[0]), int(bgr_low_color[1]), int(bgr_low_color[2])), -1)
    cv2.putText(ui_image, "LOW", (15, 170), font, 0.5, (255,255,255), 1)
    # High Color
    hsv_high_color = np.uint8([[[hsv_h_high, hsv_s_high, hsv_v_high]]])
    bgr_high_color = cv2.cvtColor(hsv_high_color, cv2.COLOR_HSV2BGR)[0][0]
    cv2.rectangle(ui_image, (w//2 + 10, 150), (w - 10, 210),
                  (int(bgr_high_color[0]), int(bgr_high_color[1]), int(bgr_high_color[2])), -1)
    cv2.putText(ui_image, "HIGH", (w//2 + 15, 170), font, 0.5, (0,0,0), 1)

    # --- Grayscale ---
    cv2.putText(ui_image, "Grayscale Filter", (10, 250), font, 0.7, (255, 255, 255), 2)
    # Create gradient
    gradient = np.linspace(0, 255, w - 20, dtype=np.uint8).reshape(1, w - 20)
    gradient_img = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    ui_image[260:320, 10:w-10] = gradient_img
    
    # Draw markers
    low_pos = int((gray_low_thresh / 255.0) * (w - 20)) + 10
    high_pos = int((gray_high_thresh / 255.0) * (w - 20)) + 10
    
    cv2.line(ui_image, (low_pos, 250), (low_pos, 330), (0, 255, 0), 2) # Low = Green
    cv2.putText(ui_image, str(gray_low_thresh), (low_pos - 10, 350),
                font, 0.5, (0, 255, 0), 1)
    
    cv2.line(ui_image, (high_pos, 250), (high_pos, 330), (0, 0, 255), 2) # High = Red
    cv2.putText(ui_image, str(gray_high_thresh), (high_pos - 10, 370),
                font, 0.5, (0, 0, 255), 1)

    return ui_image

# --- Dummy callback for trackbars (Unchanged) ---
def on_trackbar_change(val):
    pass

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Argparse (Unchanged) ---
    parser = argparse.ArgumentParser(description="AirSim Car Dashboard")
    parser.add_argument("--mode", choices=['manual', 'auto'],
                        default='manual', help="Initial control mode (default: manual)")
    parser.add_argument('--use_yolo', action='store_true',
                        help="Enable YOLOv8 object detection (default: False)")
    args = parser.parse_args()

    lane_ids = set()  # will be filled after connection
    
    try:
        client = airsim.CarClient() 
        client.confirmConnection()
        print("Connected to AirSim.")
        
        # Set API control based on mode
        if args.mode == 'auto':
            client.enableApiControl(True)
            print("API Control Enabled for Auto mode.")
        else:
            client.enableApiControl(False)
            print("API Control Disabled for Manual mode. (Will enable if you press 'K')")
            
        print("Setting segmentation ID for 'Road' meshes to 42...")
        success = client.simSetSegmentationObjectID("Road[\\w]*", 42, True)
        if not success:
            print("Could not find any 'Road' meshes to set ID.")

        # --- NEW: Collect lane IDs only from Road_2L* and Road_Large* meshes ---
        lane_objects = []
        for pattern in ["Road_2L[\\w]*", "Road_Large[\\w]*"]:
            objs = client.simListSceneObjects(pattern)
            lane_objects.extend(objs)

        for name in lane_objects:
            lane_id = client.simGetSegmentationObjectID(name)
            if lane_id != -1:
                lane_ids.add(lane_id)

        print(f"Lane objects (Road_2L / Road_Large): {lane_objects}")
        print(f"Lane segmentation IDs used for lane mask: {lane_ids}")

    except Exception as e:
        print(f"Error connecting to AirSim: {e}")
        exit()

    print("--- Controls ---")
    print("K: Toggle control mode (Manual/Auto)")
    print("R: Reset Environment")
    print("Q: Quit")
    if args.mode == 'manual':
        print("W/S: Throttle/Brake | A/D: Steering | Space: Handbrake")

    # --- *** MODIFIED: CV2 Windows & Trackbars *** ---
    
    DASHBOARD_WINDOW = "AirSim Dashboard"
    CONTROLS_WINDOW = "AirSim Controls"
    PREVIEW_WINDOW = "Color Preview"  # NEW WINDOW
    
    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.namedWindow(PREVIEW_WINDOW)
    
    # Resize the CONTROLS window to be tall enough for all 21 sliders
    cv2.resizeWindow(CONTROLS_WINDOW, 400, 700) 
    
    # --- ROI Position Trackbars ---
    cv2.createTrackbar("Top Y %", CONTROLS_WINDOW, 60, 100, on_trackbar_change)
    cv2.createTrackbar("Top X Center %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Top Width %", CONTROLS_WINDOW, 10, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Y %", CONTROLS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom X Center %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Width %", CONTROLS_WINDOW, 100, 100, on_trackbar_change)

    # --- 3-Stage Filter Trackbars ---
    cv2.createTrackbar("Blur Kernel Size", CONTROLS_WINDOW, 7, 21, on_trackbar_change)
    cv2.createTrackbar("R Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("R High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("G Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("G High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("B Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("B High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("H Low", CONTROLS_WINDOW, 0, 179, on_trackbar_change)
    cv2.createTrackbar("H High", CONTROLS_WINDOW, 179, 179, on_trackbar_change)
    cv2.createTrackbar("S Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("S High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("V Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("V High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("Gray Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("Gray High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    
    # --- Control & PID Vars (Unchanged) ---
    car_controls = airsim.CarControls()
    throttle = 0.0
    steering = 0.0
    THROTTLE_INC = 0.1
    STEERING_INC = 0.05
    DECAY = 0.1 
    control_mode = args.mode
    was_collided = False 
    BASE_THROTTLE = 0.35
    PID_P_GAIN = 0.0028
    PID_I_GAIN = 0.0002
    PID_D_GAIN = 0.0015
    pid_integral = 0.0
    pid_previous_error = 0.0
    last_timestamp = time.time()
    
    print(f"\nInitial control mode: {control_mode.upper()}")

    # (Assume yolo_model is defined elsewhere if use_yolo is enabled)

    while True:
        try:
            # --- Collision Handling ---
            if control_mode == 'auto':  # Only check collisions if we are driving
                collision_info = client.simGetCollisionInfo()
                is_collided_now = collision_info.has_collided
                if is_collided_now and not was_collided:
                    print("Collision detected! Resetting PID...")
                    throttle = 0.0
                    steering = 0.0
                    car_controls.throttle = 0.0
                    car_controls.steering = 0.0
                    client.setCarControls(car_controls)
                    client.enableApiControl(True) 
                    pid_integral = 0.0
                    pid_previous_error = 0.0
                    print("WARNING: If car is seriously stuck, control may not be regained.")
                was_collided = is_collided_now
            
            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                if control_mode == 'manual': 
                    control_mode = 'auto'
                    client.enableApiControl(True)
                else: 
                    control_mode = 'manual'
                    client.enableApiControl(False)  # Relinquish control
                pid_integral = 0.0
                pid_previous_error = 0.0
                last_timestamp = time.time()
                print(f"Control mode set to: {control_mode.upper()}")
            elif key == ord('r'):
                print("--- ENVIRONMENT RESET ---")
                client.reset()
                time.sleep(1)  # Give sim time to reset
                if control_mode == 'auto':
                    client.enableApiControl(True)
                throttle = 0.0
                steering = 0.0
                car_controls.throttle = 0.0
                car_controls.steering = 0.0
                car_controls.handbrake = False
                pid_integral = 0.0
                pid_previous_error = 0.0
                last_timestamp = time.time()
                continue 

            # --- Get ALL slider values ---
            top_y_p = cv2.getTrackbarPos("Top Y %", CONTROLS_WINDOW)
            top_x_p = cv2.getTrackbarPos("Top X Center %", CONTROLS_WINDOW)
            top_w_p = cv2.getTrackbarPos("Top Width %", CONTROLS_WINDOW)
            bottom_y_p = cv2.getTrackbarPos("Bottom Y %", CONTROLS_WINDOW)
            bottom_x_p = cv2.getTrackbarPos("Bottom X Center %", CONTROLS_WINDOW)
            bottom_w_p = cv2.getTrackbarPos("Bottom Width %", CONTROLS_WINDOW)

            blur_kernel_size = cv2.getTrackbarPos("Blur Kernel Size", CONTROLS_WINDOW)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            if blur_kernel_size < 1:
                blur_kernel_size = 1

            rgb_r_low = cv2.getTrackbarPos("R Low", CONTROLS_WINDOW)
            rgb_r_high = cv2.getTrackbarPos("R High", CONTROLS_WINDOW)
            rgb_g_low = cv2.getTrackbarPos("G Low", CONTROLS_WINDOW)
            rgb_g_high = cv2.getTrackbarPos("G High", CONTROLS_WINDOW)
            rgb_b_low = cv2.getTrackbarPos("B Low", CONTROLS_WINDOW)
            rgb_b_high = cv2.getTrackbarPos("B High", CONTROLS_WINDOW)
            hsv_h_low = cv2.getTrackbarPos("H Low", CONTROLS_WINDOW)
            hsv_h_high = cv2.getTrackbarPos("H High", CONTROLS_WINDOW)
            hsv_s_low = cv2.getTrackbarPos("S Low", CONTROLS_WINDOW)
            hsv_s_high = cv2.getTrackbarPos("S High", CONTROLS_WINDOW)
            hsv_v_low = cv2.getTrackbarPos("V Low", CONTROLS_WINDOW)
            hsv_v_high = cv2.getTrackbarPos("V High", CONTROLS_WINDOW)
            gray_low = cv2.getTrackbarPos("Gray Low", CONTROLS_WINDOW)
            gray_high = cv2.getTrackbarPos("Gray High", CONTROLS_WINDOW)
            
            # --- Get Images & Run YOLO ---
            images = get_images(client) 
            raw_frontal = images['frontal']
            follow_cam_image = images['follow']
            
            if args.use_yolo and 'yolo_model' in globals() and yolo_model is not None:
                yolo_results = yolo_model.predict(follow_cam_image, classes=2, verbose=False)
                if yolo_results:
                    result = yolo_results[0]
                    for box in result.boxes:
                        x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                        conf = float(box.conf[0])
                        cv2.rectangle(follow_cam_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(follow_cam_image, f"Car {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if raw_frontal is None or raw_frontal.size == 0:
                print("Failed to get frontal image, skipping frame.")
                continue
            
            # --- Image Processing ---
            h, w, _ = raw_frontal.shape
            roi_poly_vertices, roi_warp_src = calculate_roi_points(
                h, w, 
                top_y_p, top_x_p, top_w_p,
                bottom_y_p, bottom_x_p, bottom_w_p
            )
            frontal_with_roi = draw_roi(raw_frontal, roi_poly_vertices)
            
            final_road_mask, hsv_filtered_debug, road_mask_debug_for_dash = \
                process_image_3_stage_filter(
                    raw_frontal, roi_poly_vertices, 
                    blur_kernel_size, 
                    rgb_r_low, rgb_r_high,
                    rgb_g_low, rgb_g_high,
                    rgb_b_low, rgb_b_high,
                    hsv_h_low, hsv_h_high,
                    hsv_s_low, hsv_s_high,
                    hsv_v_low, hsv_v_high,
                    gray_low, gray_high
                )

            # --- NEW: Restrict final_road_mask to lane IDs (Road_2L / Road_Large) from segmentation ---
            try:
                seg_img = images["segmented"]  # BGR segmentation image from get_images()
                if seg_img is not None and seg_img.size > 0 and lane_ids:
                    # AirSim encodes object ID in the RED channel of original RGB.
                    # After RGB->BGR conversion, the ID ends up in channel 2 (R in BGR).
                    seg_ids_img = seg_img[:, :, 2].astype(np.uint8)

                    lane_mask_ids = np.isin(seg_ids_img, list(lane_ids)).astype(np.uint8) * 255

                    # Keep only pixels that are BOTH in your 3-stage mask AND in those lane IDs
                    final_road_mask = cv2.bitwise_and(final_road_mask, lane_mask_ids)
                    road_mask_debug_for_dash = final_road_mask.copy()
            except Exception as e:
                print(f"Lane-ID mask error (Road_2L / Road_Large): {e}")

            final_frontal_image = frontal_with_roi
            current_steering_value = None 
            
            # --- Manual Mode ---
            if control_mode == 'manual':
                car_state = client.getCarState()
                # Just something to display; you might want steering from car_controls instead.
                current_steering_value = car_state.kinematics_estimated.angular_velocity.z_val
            
            # --- AUTO MODE LOGIC ---
            else:
                warped_mask, matrix_inv = perspective_warp(final_road_mask, roi_warp_src)
                offset, center_point_viz = calculate_steering_from_blob(warped_mask)
                
                if center_point_viz is None:
                    auto_steering = 0.0
                    auto_throttle = BASE_THROTTLE
                    pid_integral = 0.0
                    pid_previous_error = 0.0
                    final_frontal_image = frontal_with_roi 
                else:
                    current_time = time.time()
                    delta_time = current_time - last_timestamp
                    last_timestamp = current_time

                    error = offset 
                    p_term = PID_P_GAIN * error
                    pid_integral += error * delta_time
                    pid_integral = np.clip(pid_integral, -500.0, 500.0) 
                    i_term = PID_I_GAIN * pid_integral
                    if delta_time > 0:
                        derivative = (error - pid_previous_error) / delta_time
                    else:
                        derivative = 0
                    d_term = PID_D_GAIN * derivative
                    pid_previous_error = error

                    auto_steering = np.clip(p_term + i_term + d_term, -1.0, 1.0)
                    throttle_reduction = abs(auto_steering) * 0.4
                    auto_throttle = max(0.15, BASE_THROTTLE - throttle_reduction)
                    final_frontal_image = draw_center_visuals(
                        frontal_with_roi, warped_mask, center_point_viz, matrix_inv
                    )

                car_controls.throttle = auto_throttle
                car_controls.steering = auto_steering
                car_controls.handbrake = False
                client.setCarControls(car_controls)
                current_steering_value = auto_steering
            
            # --- Draw Steering Lines ---
            if current_steering_value is not None:
                final_frontal_image = draw_steering_lines(final_frontal_image, current_steering_value)

            # --- Dashboard Update ---
            car_state = client.getCarState()
            
            dashboard = create_dashboard(
                final_frontal_image,
                images['driver'],
                images['reverse'], 
                follow_cam_image, 
                images["segmented"],
                road_mask_debug_for_dash,
                car_state,
                control_mode,
                current_steering_value
            )
            
            if dashboard is not None:
                cv2.imshow(DASHBOARD_WINDOW, dashboard)
            
            # --- Color Preview Window ---
            preview_ui = create_color_preview_ui(
                400, 300,  # Height and Width of this window
                rgb_r_low, rgb_r_high, rgb_g_low, rgb_g_high,
                rgb_b_low, rgb_b_high,
                hsv_h_low, hsv_h_high, hsv_s_low, hsv_s_high,
                hsv_v_low, hsv_v_high,
                gray_low, gray_high
            )
            cv2.imshow(PREVIEW_WINDOW, preview_ui)

            time.sleep(0.01) 
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            if "Connection was closed" in str(e):
                print("Connection lost. Attempting to reconnect...")
                try:
                    client.confirmConnection()
                    if control_mode == 'auto':
                        client.enableApiControl(True)
                    print("Reconnected!")
                except Exception as recon_e:
                    print(f"Reconnect failed: {recon_e}. Exiting.")
                    break
            else:
                if ("polyfit" in str(e) or
                    "concatenate" in str(e) or
                    "nonzero" in str(e) or
                    "mean of empty" in str(e)):
                    print(f"CV Error (likely no road found): {e}")
                    if control_mode == 'auto':
                        pid_previous_error = 0.0
                        car_controls.steering = 0.0
                        car_controls.throttle = 0.1 
                        client.setCarControls(car_controls)
                else:
                    print(f"An error occurred in the loop: {e}")
                    break 

    # --- Cleanup ---
    if client.isApiControlEnabled():
        car_controls.throttle = 0.0
        car_controls.steering = 0.0
        car_controls.handbrake = True
        client.setCarControls(car_controls)
        client.enableApiControl(False)
    
    print("API Control Disabled. Exiting.")
    cv2.destroyAllWindows()
