import airsim
import numpy as np
import cv2
import time
import argparse
import warnings
from ultralytics import YOLO  # <-- NEW: Import YOLO

# --- Configuration Mappings ---
CAMERA_MAP = {
    "frontal": "0",   # Frontal Camera ID
    "driver": "3",    # Driver's POV Camera ID
    "reverse": "4",   # Back/Reverse Camera ID
    "follow": "follow_cam" # "Eye View" Camera
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True), # Compressed
    "depth": (airsim.ImageType.DepthPlanar, True, False),
    "segmented": (airsim.ImageType.Segmentation, False, False)
}

# --- Image Processing Function ---
def get_images(client):
    """
    Gets compressed, CV2-decoded images from all defined cameras.
    Returns a dictionary of images.
    """
    cam_names = ["frontal", "driver", "reverse", "follow"]
    requests = []
    
    for cam_name in cam_names:
        camera_id = CAMERA_MAP[cam_name]
        airsim_type, is_float, is_compressed = TYPE_MAP["original"]
        requests.append(airsim.ImageRequest(camera_id, airsim_type, is_float, is_compressed))
    
    responses = client.simGetImages(requests)
    images = {}
    
    for i, cam_name in enumerate(cam_names):
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
            cv2.putText(img_to_show, f"{cam_name} Image Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        images[cam_name] = img_to_show
        
    return images

# -----------------------------------------------------------------
# --- 🤖 LANE DETECTION & FOLLOWING FUNCTIONS 🤖 ---
# -----------------------------------------------------------------

def calculate_roi_points(height, width, top_y_p, top_x_p, top_w_p, bottom_y_p, bottom_x_p, bottom_w_p):
    """
    Calculates the ROI polygon vertices and the warp source points
    based on percentage values from trackbars.
    """
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
    
    vertices_for_poly = np.array([
        [(bottom_left_x, bottom_y), (top_left_x, top_y), (top_right_x, top_y), (bottom_right_x, bottom_y)]
    ], dtype=np.int32)
    
    vertices_for_warp = np.float32([
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bottom_left_x, bottom_y),
        (bottom_right_x, bottom_y)
    ])
    
    return vertices_for_poly, vertices_for_warp

def draw_roi(image, vertices):
    img_copy = image.copy()
    cv2.polylines(img_copy, vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    return img_copy

def process_image_for_lines(img, roi_vertices, blur_kernel_size, canny_low_thresh, canny_high_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurry = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    canny = cv2.Canny(blurry, canny_low_thresh, canny_high_thresh)
    
    height, width = canny.shape
    mask = np.zeros_like(canny)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(canny, mask)
    
    return masked_edges, blurry, canny

def perspective_warp(img, src_points):
    height, width = img.shape[:2]
    
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix_inv

def calculate_steering_angle(warped_img):
    height, width = warped_img.shape
    
    histogram = np.sum(warped_img[height // 2:, :], axis=0)
    midpoint = width // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    n_windows = 9
    window_height = height // n_windows
    margin = 100
    min_pixels = 50
    
    left_lane_inds = []
    right_lane_inds = []
    
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    current_leftx = left_base
    current_rightx = right_base
    
    for window in range(n_windows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = current_leftx - margin
        win_xleft_high = current_leftx + margin
        win_xright_low = current_rightx - margin
        win_xright_high = current_rightx + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > min_pixels:
            current_leftx = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pixels:
            current_rightx = int(np.mean(nonzerox[good_right_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    try:
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
    except (np.linalg.LinAlgError, TypeError):
        return 0.0, 0, None, None
    
    y_eval = height
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    lane_center = (left_x + right_x) / 2
    car_center = width / 2
    
    offset_pixels = car_center - lane_center
    
    P_CONSTANT = 0.0025 
    steering_angle = offset_pixels * P_CONSTANT
    steering_angle = np.clip(steering_angle, -1.0, 1.0)
    
    return steering_angle, offset_pixels, left_fit, right_fit

def draw_lane_visuals(original_img, warped_img, left_fit, right_fit, matrix_inv):
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = cv2.cvtColor(warp_zero, cv2.COLOR_GRAY2BGR)
    
    height, width = warped_img.shape
    ploty = np.linspace(0, height - 1, height)
    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        return original_img
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    unwarped_overlay = cv2.warpPerspective(color_warp, matrix_inv, (width, height))
    
    result = cv2.addWeighted(original_img, 1, unwarped_overlay, 0.3, 0)
    return result

# -----------------------------------------------------------------
# --- 🤖 END OF LANE DETECTION FUNCTIONS 🤖 ---
# -----------------------------------------------------------------


def create_dashboard(frontal_img, driver_img, reverse_img, follow_img, frontal_blurry, frontal_canny, 
                     car_state, control_mode, steering_value=None):
    """
    Creates a dashboard with multiple camera views and a text footer.
    """
    
    STD_HEIGHT = 200 
    STD_WIDTH = 250  
    
    try:
        front_resized = cv2.resize(frontal_img, (STD_WIDTH, STD_HEIGHT))
        driver_resized = cv2.resize(driver_img, (STD_WIDTH, STD_HEIGHT))
        reverse_resized = cv2.resize(reverse_img, (STD_WIDTH, STD_HEIGHT))
        follow_resized = cv2.resize(follow_img, (STD_WIDTH, STD_HEIGHT))
        
        blurry_resized = cv2.resize(frontal_blurry, (STD_WIDTH, STD_HEIGHT))
        blurry_resized = cv2.cvtColor(blurry_resized, cv2.COLOR_GRAY2BGR)
        
        canny_resized = cv2.resize(frontal_canny, (STD_WIDTH, STD_HEIGHT))
        canny_resized = cv2.cvtColor(canny_resized, cv2.COLOR_GRAY2BGR)

    except Exception as e:
        print(f"Error resizing images: {e}")
        blank_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        front_resized = driver_resized = reverse_resized = blurry_resized = canny_resized = follow_resized = blank_img
    
    cv2.putText(driver_resized, "DRIVER", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(front_resized, "FRONTAL (ROI & Lane)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(blurry_resized, "BLURRY", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canny_resized, "CANNY EDGES", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(reverse_resized, "REVERSE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(follow_resized, "FOLLOW (YOLO)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined_image_row1 = np.hstack([driver_resized, front_resized, blurry_resized])
    combined_image_row2 = np.hstack([canny_resized, reverse_resized, follow_resized])
    combined_image = np.vstack([combined_image_row1, combined_image_row2]) 

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

    if control_mode == 'auto' and steering_value is not None:
        steer_text = f"Steer: {steering_value:.2f}"
        steer_color = (0, 0, 255)
        cv2.putText(footer, steer_text, (padding, padding + line_height * 3), 
                    font, font_scale, steer_color, font_thickness)
    
    if car_state.handbrake:
        hb_text = "HANDBRAKE ON"
        (text_width, _), _ = cv2.getTextSize(hb_text, font, font_scale, font_thickness)
        hb_x = w - text_width - padding
        cv2.putText(footer, hb_text, (hb_x, padding + line_height * 1), 
                    font, font_scale, (0, 0, 255), font_thickness)

    dashboard = np.vstack([combined_image, footer])
    
    return dashboard

# --- Dummy callback for trackbars ---
def on_trackbar_change(val):
    pass

# --- Main Execution ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AirSim Car Dashboard")
    parser.add_argument(
        "--mode", 
        choices=['manual', 'auto'], 
        default='manual', 
        help="Initial control mode (default: manual)"
    )
    args = parser.parse_args()
    
    try:
        client = airsim.CarClient() 
        client.confirmConnection()
        client.enableApiControl(True)
        print("Connected to AirSim Car! API Control Enabled.")

    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running: {e}")
        exit()

    # --- NEW: Load YOLO Model ---
    print("Loading YOLOv8 model... This may take a moment (first-time download).")
    try:
        yolo_model = YOLO('yolov8n.pt') # Use the 'nano' model for speed
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model. Have you run 'pip install ultralytics'? Error: {e}")
        exit()
    # --- END NEW ---

    print("--- Controls ---")
    print("K: Toggle control mode (Manual/Auto)")
    print("R: Reset Environment")
    print("Q: Quit")
    if args.mode == 'manual':
         print("W/S: Throttle/Brake | A/D: Steering | Space: Handbrake")

    DASHBOARD_WINDOW = "AirSim Lane Follower"
    ROI_POS_WINDOW = "ROI - Position"
    ROI_FILTER_WINDOW = "ROI - Filters"
    
    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ROI_POS_WINDOW)
    cv2.namedWindow(ROI_FILTER_WINDOW)
    
    cv2.createTrackbar("Top Y %", ROI_POS_WINDOW, 60, 100, on_trackbar_change)
    cv2.createTrackbar("Top X Center %", ROI_POS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Top Width %", ROI_POS_WINDOW, 10, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Y %", ROI_POS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom X Center %", ROI_POS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Width %", ROI_POS_WINDOW, 100, 100, on_trackbar_change)

    cv2.createTrackbar("Canny Lower Threshold", ROI_FILTER_WINDOW, 50, 255, on_trackbar_change)
    cv2.createTrackbar("Canny Upper Threshold", ROI_FILTER_WINDOW, 150, 255, on_trackbar_change)
    cv2.createTrackbar("Blur Kernel Size", ROI_FILTER_WINDOW, 5, 21, on_trackbar_change)
    
    car_controls = airsim.CarControls()
    throttle = 0.0
    steering = 0.0
    THROTTLE_INC = 0.1
    STEERING_INC = 0.05
    DECAY = 0.1 
    
    control_mode = args.mode
    was_collided = False 
    
    AUTO_THROTTLE = 0.3 
    
    print(f"\nInitial control mode: {control_mode.upper()}")

    while True:
        try:
            collision_info = client.simGetCollisionInfo()
            is_collided_now = collision_info.has_collided
            
            if is_collided_now and not was_collided:
                print("Collision detected! Attempting to re-enable API control...")
                throttle = 0.0
                steering = 0.0
                car_controls.throttle = 0.0
                car_controls.steering = 0.0
                client.setCarControls(car_controls)
                client.enableApiControl(True) 
                print("WARNING: If car is seriously stuck, control may not be regained.")
            
            was_collided = is_collided_now

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            elif key == ord('k'):
                if control_mode == 'manual':
                    control_mode = 'auto'
                    throttle = 0.0
                    steering = 0.0
                else:
                    control_mode = 'manual'
                print(f"Control mode set to: {control_mode.upper()}")

            elif key == ord('r'):
                print("--- ENVIRONMENT RESET ---")
                client.reset()
                client.enableApiControl(True)
                throttle = 0.0
                steering = 0.0
                car_controls.throttle = 0.0
                car_controls.steering = 0.0
                car_controls.handbrake = False
                continue 

            # --- Get ALL slider values ---
            top_y_p = cv2.getTrackbarPos("Top Y %", ROI_POS_WINDOW)
            top_x_p = cv2.getTrackbarPos("Top X Center %", ROI_POS_WINDOW)
            top_w_p = cv2.getTrackbarPos("Top Width %", ROI_POS_WINDOW)
            bottom_y_p = cv2.getTrackbarPos("Bottom Y %", ROI_POS_WINDOW)
            bottom_x_p = cv2.getTrackbarPos("Bottom X Center %", ROI_POS_WINDOW)
            bottom_w_p = cv2.getTrackbarPos("Bottom Width %", ROI_POS_WINDOW)

            canny_low_thresh = cv2.getTrackbarPos("Canny Lower Threshold", ROI_FILTER_WINDOW)
            canny_high_thresh = cv2.getTrackbarPos("Canny Upper Threshold", ROI_FILTER_WINDOW)
            
            blur_kernel_size = cv2.getTrackbarPos("Blur Kernel Size", ROI_FILTER_WINDOW)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            if blur_kernel_size < 1:
                blur_kernel_size = 1

            images = get_images(client)
            raw_frontal = images['frontal']
            
            # --- NEW: YOLO Detection on 'follow' camera ---
            follow_cam_image = images['follow']
            # 'classes=2' tells YOLO to only detect 'car' (in the COCO dataset)
            # 'verbose=False' stops it from printing a ton of text
            yolo_results = yolo_model.predict(follow_cam_image, classes=2, verbose=False)
            
            if yolo_results:
                result = yolo_results[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                    conf = float(box.conf[0])
                    # Draw the box and confidence
                    cv2.rectangle(follow_cam_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(follow_cam_image, f"Car {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # --- END NEW YOLO LOGIC ---
            
            if raw_frontal is None or raw_frontal.size == 0:
                print("Failed to get frontal image, skipping frame.")
                continue
            
            h, w, _ = raw_frontal.shape
            
            roi_poly_vertices, roi_warp_src = calculate_roi_points(h, w, 
                                                                   top_y_p, top_x_p, top_w_p,
                                                                   bottom_y_p, bottom_x_p, bottom_w_p)
            
            frontal_with_roi = draw_roi(raw_frontal, roi_poly_vertices)

            masked_edges, frontal_blurry_debug, frontal_canny_debug = \
                process_image_for_lines(raw_frontal, roi_poly_vertices, 
                                        blur_kernel_size, canny_low_thresh, canny_high_thresh)

            final_frontal_image = frontal_with_roi
            current_steering_value = None 
            
            if control_mode == 'manual':
                handbrake_toggled = False
                
                if key == ord('w'):
                    throttle = min(1.0, throttle + THROTTLE_INC)
                elif key == ord('s'):
                    throttle = max(-1.0, throttle - THROTTLE_INC)
                elif key == ord('a'):
                    steering = max(-1.0, steering - STEERING_INC)
                elif key == ord('d'):
                    steering = min(1.0, steering + STEERING_INC)
                elif key == ord(' '):
                    car_controls.handbrake = not car_controls.handbrake
                    handbrake_toggled = True
                else:
                    if not handbrake_toggled: 
                        if throttle > 0: throttle = max(0.0, throttle - DECAY)
                        elif throttle < 0: throttle = min(0.0, throttle + DECAY)
                        if steering > 0: steering = max(0.0, steering - DECAY)
                        elif steering < 0: steering = min(0.0, steering + DECAY)

                car_controls.throttle = throttle
                car_controls.steering = steering
                client.setCarControls(car_controls)
            
            else:
                # --- AUTO MODE ---
                warped_img, matrix_inv = perspective_warp(masked_edges, roi_warp_src)
                
                auto_steering, offset, l_fit, r_fit = calculate_steering_angle(warped_img)
                current_steering_value = auto_steering
                
                car_controls.throttle = AUTO_THROTTLE
                car_controls.steering = auto_steering
                car_controls.handbrake = False
                client.setCarControls(car_controls)
                
                final_frontal_image = draw_lane_visuals(frontal_with_roi, warped_img, l_fit, r_fit, matrix_inv)

            car_state = client.getCarState()
            
            dashboard = create_dashboard(final_frontal_image, images['driver'], images['reverse'], 
                                         follow_cam_image, # <-- Pass the modified 'follow' image
                                         frontal_blurry_debug, frontal_canny_debug,
                                         car_state, control_mode, current_steering_value)
            
            if dashboard is not None:
                cv2.imshow(DASHBOARD_WINDOW, dashboard)
            
            time.sleep(0.01) 
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            if "Connection was closed" in str(e):
                print("Connection lost. Attempting to reconnect...")
                try:
                    client.confirmConnection()
                    client.enableApiControl(True)
                    print("Reconnected!")
                except Exception as recon_e:
                    print(f"Reconnect failed: {recon_e}. Exiting.")
                    break
            else:
                print(f"An error occurred in the loop: {e}")
                if "polyfit" in str(e) or "concatenate" in str(e) or "nonzero" in str(e):
                    print(f"CV Error (likely no lines found): {e}")
                    if control_mode == 'auto':
                        car_controls.throttle = AUTO_THROTTLE
                        car_controls.steering = 0.0
                        client.setCarControls(car_controls)
                else:
                    break 

    car_controls.throttle = 0.0
    car_controls.steering = 0.0
    car_controls.handbrake = True
    client.setCarControls(car_controls)
    client.enableApiControl(False)
    
    print("API Control Disabled. Exiting.")
    cv2.destroyAllWindows()