import airsim
import numpy as np
import cv2
import time
import argparse
from ultralytics import YOLO  # Optional: For car detection on follow cam

# =============================
#  CONFIG & IMAGE ACQUISITION
# =============================

CAMERA_MAP = {
    "frontal": "0", "driver": "3", "reverse": "4", "follow": "follow_cam"
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True),
}

def get_images(client):
    """
    Get images from frontal, driver, reverse and follow cameras (compressed).
    Returns dict with BGR images.
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
            cv2.putText(img_to_show, f"{cam_name} Image Error", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        images[cam_name] = img_to_show
        
    return images

# =============================
#  LANE / ROAD DETECTION
# =============================

def calculate_roi_points(height, width,
                         top_y_p, top_x_p, top_w_p,
                         bottom_y_p, bottom_x_p, bottom_w_p):
    """
    Calculate trapezoid ROI vertices (for drawing & warp).
    All parameters are in percent of image size.
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
    """Draw trapezoid ROI on image."""
    img_copy = image.copy()
    cv2.polylines(img_copy, vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    return img_copy

def process_image_3_stage_filter(img, roi_vertices, blur_kernel_size, 
                                 rgb_r_low, rgb_r_high, rgb_g_low, rgb_g_high, rgb_b_low, rgb_b_high,
                                 hsv_h_low, hsv_h_high, hsv_s_low, hsv_s_high, hsv_v_low, hsv_v_high,
                                 gray_low_thresh, gray_high_thresh):
    """
    3-stage road detection: RGB -> HSV -> Gray -> Threshold + ROI.
    Returns:
        final_road_mask_for_tracker : binary mask (uint8) for driving
        hsv_filtered_debug          : BGR debug image
        final_road_mask_for_tracker : same as mask, for dashboard
    """

    # Stage 1: RGB filter (using BGR order in OpenCV)
    bgr_low = np.array([rgb_b_low, rgb_g_low, rgb_r_low])
    bgr_high = np.array([rgb_b_high, rgb_g_high, rgb_r_high])
    rgb_mask = cv2.inRange(img, bgr_low, bgr_high)
    rgb_filtered_img = cv2.bitwise_and(img, img, mask=rgb_mask)

    # Stage 2: HSV filter
    hsv_img = cv2.cvtColor(rgb_filtered_img, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([hsv_h_low, hsv_s_low, hsv_v_low])
    hsv_high = np.array([hsv_h_high, hsv_s_high, hsv_v_high])
    hsv_mask = cv2.inRange(hsv_img, hsv_low, hsv_high)
    hsv_filtered_debug = cv2.bitwise_and(rgb_filtered_img, rgb_filtered_img, mask=hsv_mask)

    # Stage 3: Gray & threshold
    gray_from_filtered = cv2.cvtColor(hsv_filtered_debug, cv2.COLOR_BGR2GRAY)
    blurry_gray = cv2.GaussianBlur(gray_from_filtered, (blur_kernel_size, blur_kernel_size), 0)
    final_road_mask = cv2.inRange(blurry_gray, gray_low_thresh, gray_high_thresh)
    
    # Apply ROI
    height, width = final_road_mask.shape
    roi_mask = np.zeros_like(final_road_mask)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    final_road_mask_for_tracker = cv2.bitwise_and(final_road_mask, roi_mask)
    
    return final_road_mask_for_tracker, hsv_filtered_debug, final_road_mask_for_tracker

def calculate_steering_from_blob(warped_mask):
    """
    Compute horizontal offset of the road blob w.r.t. image center.

    Returns:
        offset (pixels): lane_center - car_center
        center_point   : (x, y) on warped image (for visualization), or None
    """
    h, w = warped_mask.shape
    y_eval = int(h * 0.75)  # Look-ahead row

    look_ahead_row = warped_mask[y_eval, :]
    indices = look_ahead_row.nonzero()[0]
    
    if len(indices) > 0:
        lane_center = np.mean(indices)
        car_center = w / 2.0
        offset = lane_center - car_center   # >0 => blob on right, <0 => blob on left
        return offset, (int(lane_center), y_eval)
    else:
        # Fallback: use histogram on bottom half
        histogram = np.sum(warped_mask[h//2:, :], axis=0)
        indices = histogram.nonzero()[0]
        if len(indices) > 0:
            lane_center = np.mean(indices)
            car_center = w / 2.0
            offset = lane_center - car_center
            return offset, None
        
        # No blob at all
        return 0.0, None

def perspective_warp(img, src_points):
    """
    Warp trapezoid ROI to a bird's-eye view rectangle.
    """
    height, width = img.shape[:2]
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix_inv

def draw_center_visuals(original_img, warped_mask, center_point, matrix_inv):
    """
    Draw red dot where we think the lane center is (in original perspective).
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
    """
    Visual steering lines at the bottom of the frontal image.
    """
    if steering_value is None:
        return image
    image_copy = image.copy()
    h, w, _ = image_copy.shape

    start_left = (w // 2 - 40, h - 20)
    start_right = (w // 2 + 40, h - 20)
    line_length = h // 4
    max_angle_deg = 30  # purely visual

    angle_rad = steering_value * np.deg2rad(max_angle_deg)
    end_y = int(start_left[1] - line_length * np.cos(angle_rad))
    end_x_offset = int(line_length * np.sin(angle_rad))

    end_left_x = start_left[0] + end_x_offset
    end_right_x = start_right[0] + end_x_offset

    cv2.line(image_copy, start_left, (end_left_x, end_y), (0, 255, 255), 3)
    cv2.line(image_copy, start_right, (end_right_x, end_y), (0, 255, 255), 3)
    return image_copy

def steering_from_offset(offset, frame_width, gain=1.0):
    """
    Map pixel offset (lane_center - car_center) to steering in [-1, 1].

    offset > 0  -> lane on RIGHT  -> steer right (positive)
    offset < 0  -> lane on LEFT   -> steer left  (negative)
    """
    max_offset = frame_width / 2.0
    if max_offset == 0:
        return 0.0
    norm_offset = offset / max_offset   # [-1 .. 1]
    steering = norm_offset * gain
    steering = float(np.clip(steering, -1.0, 1.0))
    return steering

# =============================
#  DASHBOARD / UI
# =============================

def create_dashboard(frontal_img, driver_img, reverse_img, follow_img, 
                     hsv_filtered_debug, final_road_mask_debug,
                     car_state, control_mode, steering_value=None):
    
    STD_HEIGHT = 200
    STD_WIDTH = 250
    
    try:
        front_resized = cv2.resize(frontal_img, (STD_WIDTH, STD_HEIGHT))
        driver_resized = cv2.resize(driver_img, (STD_WIDTH, STD_HEIGHT))
        reverse_resized = cv2.resize(reverse_img, (STD_WIDTH, STD_HEIGHT))
        follow_resized = cv2.resize(follow_img, (STD_WIDTH, STD_HEIGHT))
        hsv_filtered_resized = cv2.resize(hsv_filtered_debug, (STD_WIDTH, STD_HEIGHT))
        road_mask_resized = cv2.resize(final_road_mask_debug, (STD_WIDTH, STD_HEIGHT))
        road_mask_resized = cv2.cvtColor(road_mask_resized, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Error resizing images: {e}")
        blank_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        front_resized = driver_resized = reverse_resized = hsv_filtered_resized = road_mask_resized = follow_resized = blank_img
    
    cv2.putText(driver_resized, "DRIVER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(front_resized, "FRONTAL (ROI & Target)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(hsv_filtered_resized, "HSV FILTERED", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(road_mask_resized, "FINAL ROAD MASK", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(reverse_resized, "REVERSE", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(follow_resized, "FOLLOW (YOLO)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined_image_row1 = np.hstack([driver_resized, front_resized, hsv_filtered_resized])
    combined_image_row2 = np.hstack([road_mask_resized, reverse_resized, follow_resized])
    combined_image = np.vstack([combined_image_row1, combined_image_row2])

    # Footer
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
        cv2.putText(footer, steer_text, (padding, padding + line_height * 3),
                    font, font_scale, (0, 0, 255), font_thickness)
    
    if car_state.handbrake:
        hb_text = "HANDBRAKE ON"
        (text_width, _), _ = cv2.getTextSize(hb_text, font, font_scale, font_thickness)
        hb_x = w - text_width - padding
        cv2.putText(footer, hb_text, (hb_x, padding + line_height * 0),
                    font, font_scale, (0, 0, 255), font_thickness)

    dashboard = np.vstack([combined_image, footer])
    return dashboard

def on_trackbar_change(val):
    pass

# =============================
#  MAIN
# =============================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AirSim Road-Following Car")
    parser.add_argument("--mode", choices=['manual', 'auto'], default='auto',
                        help="Initial control mode (default: auto)")
    parser.add_argument('--use_yolo', action='store_true',
                        help="Enable YOLOv8 object detection on the follow camera")
    args = parser.parse_args()
    
    try:
        client = airsim.CarClient() 
        client.confirmConnection()
        client.enableApiControl(True)
        print("Connected to AirSim Car! API Control Enabled.")
    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running: {e}")
        exit()

    # YOLO (optional)
    yolo_model = None
    if args.use_yolo:
        print("Loading YOLOv8 model...")
        try:
            yolo_model = YOLO('yolov8n.pt')
            print("YOLO model loaded.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Disabling YOLO.")
            args.use_yolo = False
    else:
        print("YOLO detection is OFF. Use --use_yolo to enable.")

    print("--- Controls ---")
    print("K: Toggle control mode (Manual/Auto)")
    print("R: Reset Environment")
    print("Q: Quit")
    print("Manual mode: W/S = Throttle/Brake, A/D = Steering, SPACE = Handbrake")
    
    DASHBOARD_WINDOW = "AirSim Road-Following Dashboard"
    CONTROLS_WINDOW = "AirSim Road-Following Controls"
    
    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.resizeWindow(CONTROLS_WINDOW, 400, 800)

    # Filter sliders
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
    cv2.createTrackbar("S High", CONTROLS_WINDOW, 50, 255, on_trackbar_change)
    cv2.createTrackbar("V Low", CONTROLS_WINDOW, 0, 255, on_trackbar_change)
    cv2.createTrackbar("V High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("Gray Low", CONTROLS_WINDOW, 50, 255, on_trackbar_change)
    cv2.createTrackbar("Gray High", CONTROLS_WINDOW, 255, 255, on_trackbar_change)

    # ROI sliders
    cv2.createTrackbar("Top Y %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Y %", CONTROLS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Top Width %", CONTROLS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Width %", CONTROLS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Top X Center %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom X Center %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    
    car_controls = airsim.CarControls()
    throttle = 0.0
    steering = 0.0
    THROTTLE_INC = 0.1
    STEERING_INC = 0.05
    DECAY = 0.1
    
    control_mode = args.mode
    was_collided = False
    print(f"\nInitial control mode: {control_mode.upper()}")

    while True:
        try:
            collision_info = client.simGetCollisionInfo()
            is_collided_now = collision_info.has_collided
            if is_collided_now and not was_collided:
                print("Collision detected! Stopping car.")
                throttle = 0.0
                steering = 0.0
                car_controls.throttle = 0.0
                car_controls.steering = 0.0
                client.setCarControls(car_controls)
            was_collided = is_collided_now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                control_mode = 'auto' if control_mode == 'manual' else 'manual'
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
                time.sleep(1)
                continue

            # Sliders
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
            
            images = get_images(client)
            raw_frontal = images['frontal']
            follow_cam_image = images['follow']

            # YOLO on follow cam
            if args.use_yolo and yolo_model is not None:
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
                    rgb_r_low, rgb_r_high, rgb_g_low, rgb_g_high, rgb_b_low, rgb_b_high,
                    hsv_h_low, hsv_h_high, hsv_s_low, hsv_s_high, hsv_v_low, hsv_v_high,
                    gray_low, gray_high
                )

            final_frontal_image = frontal_with_roi
            current_steering_value = None 
            
            # ========== MANUAL MODE ==========
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
                        if throttle > 0:
                            throttle = max(0.0, throttle - DECAY)
                        elif throttle < 0:
                            throttle = min(0.0, throttle + DECAY)
                        if steering > 0:
                            steering = max(0.0, steering - DECAY)
                        elif steering < 0:
                            steering = min(0.0, steering + DECAY)

                car_controls.throttle = throttle
                car_controls.steering = steering
                client.setCarControls(car_controls)
                current_steering_value = steering
            
            # ========== AUTO MODE: FOLLOW BLOB CENTER ==========
            else:
                warped_mask, matrix_inv = perspective_warp(final_road_mask, roi_warp_src)
                offset, center_point_viz = calculate_steering_from_blob(warped_mask)
                h_warp, w_warp = warped_mask.shape

                if center_point_viz is None:
                    # No road blob: go slow and straight
                    auto_steering = 0.0
                    auto_throttle = 0.1
                    final_frontal_image = frontal_with_roi
                else:
                    # Map offset -> steering [-1,1]
                    auto_steering = steering_from_offset(offset, w_warp, gain=1.0)
                    # Throttle: faster when centered, slower on big turns
                    turn_factor = min(abs(auto_steering), 1.0)
                    auto_throttle = max(0.15, 0.4 - 0.25 * turn_factor)

                    final_frontal_image = draw_center_visuals(
                        frontal_with_roi, warped_mask, center_point_viz, matrix_inv
                    )

                car_controls.throttle = auto_throttle
                car_controls.steering = auto_steering
                car_controls.handbrake = False
                client.setCarControls(car_controls)
                current_steering_value = auto_steering
            
            if current_steering_value is not None:
                final_frontal_image = draw_steering_lines(final_frontal_image, current_steering_value)

            car_state = client.getCarState()
            dashboard = create_dashboard(
                final_frontal_image, images['driver'], images['reverse'], 
                follow_cam_image,
                hsv_filtered_debug, road_mask_debug_for_dash,
                car_state, control_mode, current_steering_value
            )
            
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
                break 

    # Cleanup
    car_controls.throttle = 0.0
    car_controls.steering = 0.0
    car_controls.handbrake = True
    client.setCarControls(car_controls)
    client.enableApiControl(False)
    print("API Control Disabled. Exiting.")
    cv2.destroyAllWindows()
