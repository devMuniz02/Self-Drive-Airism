import airsim
import numpy as np
import cv2
import time
import argparse
import warnings

# --- Configuration Mappings ---
CAMERA_MAP = {
    "frontal": "0",   # Frontal Camera ID
    "driver": "3",    # Driver's POV Camera ID
    "reverse": "4",   # Back/Reverse Camera ID
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True), # Compressed
    "depth": (airsim.ImageType.DepthPlanar, True, False),
    "segmented": (airsim.ImageType.Segmentation, False, False)
}

# --- Image Processing Function (Handles Compression) ---
def get_image(client):
    """
    Gets a single, compressed, CV2-decoded image from the frontal camera.
    """
    camera_id = CAMERA_MAP["frontal"]
    airsim_type, is_float, is_compressed = TYPE_MAP["original"]
    
    request = airsim.ImageRequest(camera_id, airsim_type, is_float, is_compressed)
    response = client.simGetImages([request])[0]
    
    img_to_show = None
    if is_compressed:
        try:
            img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_to_show = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding compressed image: {e}")
    
    if img_to_show is None:
        img_to_show = np.zeros((response.height, response.width, 3), dtype=np.uint8)
        cv2.putText(img_to_show, "No Image", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    return img_to_show

# -----------------------------------------------------------------
# --- 🤖 LANE DETECTION & FOLLOWING FUNCTIONS 🤖 ---
# -----------------------------------------------------------------

def process_image_for_lines(img):
    """Applies Canny edge detection and masks a region of interest."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    height, width = canny.shape
    mask = np.zeros_like(canny)
    
    roi_vertices = [
        (0, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width, height)
    ]
    cv2.fillPoly(mask, [np.array(roi_vertices, dtype=np.int32)], 255)
    
    masked_edges = cv2.bitwise_and(canny, mask)
    return masked_edges

def perspective_warp(img):
    """Applies a bird's-eye view perspective warp."""
    height, width = img.shape[:2]
    
    src = np.float32([
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [0, height],
        [width, height]
    ])
    
    dst = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img

def calculate_steering_angle(warped_img):
    """
    Calculates the steering angle based on the lane center.
    Returns: steering_angle (-1 to 1), lane_center_offset
    """
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

def draw_lane_visuals(original_img, warped_img, left_fit, right_fit):
    """Draws the detected lane onto the original image."""
    
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
    
    src = np.float32([
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [0, height],
        [width, height]
    ])
    dst = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    matrix_inv = cv2.getPerspectiveTransform(dst, src) 
    unwarped_overlay = cv2.warpPerspective(color_warp, matrix_inv, (width, height))
    
    result = cv2.addWeighted(original_img, 1, unwarped_overlay, 0.3, 0)
    
    return result

# -----------------------------------------------------------------
# --- 🤖 END OF LANE DETECTION FUNCTIONS 🤖 ---
# -----------------------------------------------------------------


# --- MODIFIED: This function is completely rewritten ---
def create_dashboard(image_to_display, car_state, control_mode, steering_value=None):
    """
    Creates a dashboard with a text footer below the main image.
    """
    
    if image_to_display.shape[0] < 2 or image_to_display.shape[1] < 2:
         image_to_display = np.zeros((240, 320, 3), dtype=np.uint8)
         cv2.putText(image_to_display, "No Image", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    h, w, _ = image_to_display.shape
    
    # --- Create the Footer ---
    footer_height = 100  # Height of the text box
    footer = np.zeros((footer_height, w, 3), dtype=np.uint8)
    
    # --- Text Settings ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20  # Pixels between each line
    padding = 15      # Pixels from left edge
    
    # --- Define Text Lines ---
    
    # Line 1: Mode
    mode_text = f"MODE: {control_mode.upper()} (Press 'K' to toggle)"
    mode_color = (0, 255, 0) if control_mode == 'auto' else (0, 255, 255)
    cv2.putText(footer, mode_text, (padding, padding + line_height * 0), 
                font, font_scale, mode_color, font_thickness)
    
    # Line 2: Speed
    speed_mph = car_state.speed * 2.23694 
    speed_text = f"Speed: {speed_mph:.0f} MPH"
    cv2.putText(footer, speed_text, (padding, padding + line_height * 1), 
                font, font_scale, (255, 255, 255), font_thickness)

    # Line 3: Gear
    gear_text = f"Gear: {car_state.gear}"
    cv2.putText(footer, gear_text, (padding, padding + line_height * 2), 
                font, font_scale, (255, 255, 255), font_thickness)

    # Line 4: Steering (only in auto mode)
    if control_mode == 'auto' and steering_value is not None:
        steer_text = f"Steer: {steering_value:.2f}"
        steer_color = (0, 0, 255) # Red for steering
        cv2.putText(footer, steer_text, (padding, padding + line_height * 3), 
                    font, font_scale, steer_color, font_thickness)
    
    # Handbrake Status (on the right side)
    if car_state.handbrake:
        hb_text = "HANDBRAKE ON"
        (text_width, _), _ = cv2.getTextSize(hb_text, font, font_scale, font_thickness)
        hb_x = w - text_width - padding # Position on the right
        cv2.putText(footer, hb_text, (hb_x, padding + line_height * 1), 
                    font, font_scale, (0, 0, 255), font_thickness)

    # Stack the image and the footer vertically
    dashboard = np.vstack([image_to_display, footer])
    
    return dashboard


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
        print("--- Controls ---")
        print("K: Toggle control mode (Manual/Auto)")
        print("Q: Quit")
        if args.mode == 'manual':
             print("W/S: Throttle/Brake | A/D: Steering | Space: Handbrake")
        
    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running in Car mode: {e}")
        exit()

    WINDOW_NAME = "AirSim Lane Follower"
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    

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

            raw_image = get_image(client)
            if raw_image is None or raw_image.size == 0:
                print("Failed to get image, skipping frame.")
                continue

            final_image_to_display = raw_image
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
                final_image_to_display = raw_image
            
            else:
                edge_img = process_image_for_lines(raw_image)
                warped_img = perspective_warp(edge_img)
                auto_steering, offset, l_fit, r_fit = calculate_steering_angle(warped_img)
                current_steering_value = auto_steering # Store for dashboard
                
                car_controls.throttle = AUTO_THROTTLE
                car_controls.steering = auto_steering
                car_controls.handbrake = False
                client.setCarControls(car_controls)
                
                final_image_to_display = draw_lane_visuals(raw_image, warped_img, l_fit, r_fit)

            car_state = client.getCarState()
            # MODIFIED: Passed current_steering_value to dashboard
            dashboard = create_dashboard(final_image_to_display, car_state, control_mode, current_steering_value)
            
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
                    client.enableApiControl(True)
                    print("Reconnected!")
                except Exception as recon_e:
                    print(f"Reconnect failed: {recon_e}. Exiting.")
                    break
            else:
                print(f"An error occurred in the loop: {e}")
                if "polyfit" in str(e) or "concatenate" in str(e):
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