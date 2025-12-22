import airsim
import numpy as np
import cv2
import time
import argparse
import warnings
from ultralytics import YOLO  # Optional car detection

# --- 💡 NEW IMPORTS for YOLOP ---
import torch
import torchvision.transforms as transforms

# =========================================================
#   CONFIGURATION
# =========================================================

CAMERA_MAP = {
    "frontal": "0", "driver": "3", "reverse": "4", "follow": "follow_cam"
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True),
}

# =========================================================
#   IMAGE ACQUISITION
# =========================================================

def get_images(client):
    """
    Grab images from AirSim cameras, return dict of cv2 BGR images.
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

        # --- Safety: handle None or empty ---
        if img_to_show is None or img_to_show.size == 0:
            img_to_show = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                img_to_show, f"{cam_name} Image Error",
                (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

        images[cam_name] = img_to_show

    return images

# =========================================================
#   ROI / PERSPECTIVE WARP HELPERS
# =========================================================

def calculate_roi_points(height, width,
                         top_y_p, top_x_p, top_w_p,
                         bottom_y_p, bottom_x_p, bottom_w_p):
    """
    Calculates the ROI polygon vertices (trapezoid) and
    corresponding source points for perspective warp.
    """
    top_y         = height * (top_y_p / 100.0)
    top_x_center  = width  * (top_x_p / 100.0)
    top_width     = width  * (top_w_p / 100.0)

    bottom_y      = height * (bottom_y_p / 100.0)
    bottom_x_center = width * (bottom_x_p / 100.0)
    bottom_width  = width  * (bottom_w_p / 100.0)

    top_left_x     = top_x_center    - (top_width / 2)
    top_right_x    = top_x_center    + (top_width / 2)
    bottom_left_x  = bottom_x_center - (bottom_width / 2)
    bottom_right_x = bottom_x_center + (bottom_width / 2)

    vertices_for_poly = np.array(
        [[
            (bottom_left_x,  bottom_y),
            (top_left_x,     top_y),
            (top_right_x,    top_y),
            (bottom_right_x, bottom_y)
        ]],
        dtype=np.int32
    )

    vertices_for_warp = np.float32([
        (top_left_x,     top_y),
        (top_right_x,    top_y),
        (bottom_left_x,  bottom_y),
        (bottom_right_x, bottom_y)
    ])

    return vertices_for_poly, vertices_for_warp


def draw_roi(image, vertices):
    """Draws the ROI trapezoid on an image."""
    img_copy = image.copy()
    cv2.polylines(img_copy, vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    return img_copy

# =========================================================
#   YOLOP DRIVABLE-AREA SEGMENTATION
# =========================================================

yolop_img_size = (640, 640)
yolop_preprocessor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(yolop_img_size),
    transforms.ToTensor(),
])

def get_yolop_mask(cv2_img, yolop_model, device):
    """
    Runs YOLOP on a CV2 BGR image and returns a 0/255 drivable-area mask.
    Defensive checks added so we never call cv2.resize with 0 size.
    """
    # --- Safety check: empty or None frame ---
    if cv2_img is None or not hasattr(cv2_img, "shape") or cv2_img.size == 0:
        print("[WARN] get_yolop_mask received empty image, returning blank mask.")
        return np.zeros((480, 640), dtype=np.uint8)  # fallback size

    original_h, original_w = cv2_img.shape[:2]

    # --- Safety check: invalid dimensions ---
    if original_h <= 0 or original_w <= 0:
        print(f"[WARN] get_yolop_mask got invalid size ({original_w}x{original_h}), returning blank mask.")
        return np.zeros((480, 640), dtype=np.uint8)

    # BGR -> RGB for PyTorch / PIL
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img_tensor = yolop_preprocessor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # YOLOP returns: det_out, da_seg_out, ll_seg_out
        _, da_seg_out, _ = yolop_model(img_tensor)

    # da_seg_out is typically (1, 1, H, W) or (1, C, H, W)
    da_mask_tensor = da_seg_out[0]  # -> (C, H, W) or (1, H, W)

    if da_mask_tensor.dim() == 3 and da_mask_tensor.shape[0] > 1:
        # If multiple channels, take argmax to get drivable class
        da_mask_tensor = torch.argmax(da_mask_tensor, dim=0)  # (H, W)
    else:
        # Single channel: squeeze channel dim
        da_mask_tensor = da_mask_tensor.squeeze(0)  # (H, W)

    da_mask_tensor = da_mask_tensor.cpu()
    da_mask_np = da_mask_tensor.numpy().astype(np.float32)

    # If it's logits or class labels, normalize/scale safely
    if da_mask_np.max() > 1.0:
        # assume labels 0/1 or 0/2 etc.
        da_mask_np = (da_mask_np > 0).astype(np.float32)

    da_mask_cv = (da_mask_np * 255).astype(np.uint8)

    # Resize mask back to original frame size (guaranteed > 0 here)
    da_mask_resized = cv2.resize(da_mask_cv, (original_w, original_h))

    # Binarize
    _, final_mask = cv2.threshold(da_mask_resized, 127, 255, cv2.THRESH_BINARY)
    return final_mask

# =========================================================
#   STEERING & WARP
# =========================================================

def calculate_steering_from_blob(warped_mask):
    """
    Calculates the steering offset by finding the center
    of the drivable blob at a "look-ahead" line.
    offset > 0  -> blob center is to the RIGHT of image center
    offset < 0  -> blob center is to the LEFT
    """
    h, w = warped_mask.shape
    y_eval = int(h * 0.75)  # 75% down

    look_ahead_row = warped_mask[y_eval, :]
    indices = look_ahead_row.nonzero()[0]

    if len(indices) > 0:
        lane_center = np.mean(indices)
        car_center = w / 2
        offset = lane_center - car_center
        return offset, (int(lane_center), y_eval)
    else:
        # Fallback: bottom half histogram
        histogram = np.sum(warped_mask[h // 2:, :], axis=0)
        indices = histogram.nonzero()[0]
        if len(indices) > 0:
            lane_center = np.mean(indices)
            car_center = w / 2
            offset = lane_center - car_center
            return offset, None

        return 0.0, None


def perspective_warp(img, src_points):
    height, width = img.shape[:2]
    dst_points = np.float32([
        [0,       0],
        [width,   0],
        [0,    height],
        [width, height]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix_inv

# =========================================================
#   VISUALIZATION
# =========================================================

def draw_center_visuals(original_img, warped_mask, center_point, matrix_inv):
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
    start_left  = (w // 2 - 40, h - 20)
    start_right = (w // 2 + 40, h - 20)
    line_length = h // 4
    max_angle_deg = 30

    angle_rad = steering_value * np.deg2rad(max_angle_deg)
    end_y = int(start_left[1] - line_length * np.cos(angle_rad))
    end_x_offset = int(line_length * np.sin(angle_rad))

    end_left_x  = start_left[0]  + end_x_offset
    end_right_x = start_right[0] + end_x_offset

    cv2.line(image_copy, start_left,  (end_left_x,  end_y), (0, 255, 255), 3)
    cv2.line(image_copy, start_right, (end_right_x, end_y), (0, 255, 255), 3)

    return image_copy

# =========================================================
#   DASHBOARD
# =========================================================

def create_dashboard(frontal_img, driver_img, reverse_img, follow_img,
                     debug_image_1, debug_image_2,
                     car_state, control_mode, steering_value=None):

    STD_HEIGHT = 200
    STD_WIDTH  = 250

    try:
        front_resized   = cv2.resize(frontal_img,   (STD_WIDTH, STD_HEIGHT))
        driver_resized  = cv2.resize(driver_img,    (STD_WIDTH, STD_HEIGHT))
        reverse_resized = cv2.resize(reverse_img,   (STD_WIDTH, STD_HEIGHT))
        follow_resized  = cv2.resize(follow_img,    (STD_WIDTH, STD_HEIGHT))

        debug_1_resized = cv2.resize(debug_image_1, (STD_WIDTH, STD_HEIGHT))
        debug_2_resized = cv2.resize(debug_image_2, (STD_WIDTH, STD_HEIGHT))

        if len(debug_1_resized.shape) == 2:
            debug_1_resized = cv2.cvtColor(debug_1_resized, cv2.COLOR_GRAY2BGR)
        if len(debug_2_resized.shape) == 2:
            debug_2_resized = cv2.cvtColor(debug_2_resized, cv2.COLOR_GRAY2BGR)

    except Exception as e:
        print(f"Error resizing images: {e}")
        blank_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        front_resized = driver_resized = reverse_resized = debug_1_resized = debug_2_resized = follow_resized = blank_img

    cv2.putText(driver_resized, "DRIVER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(front_resized, "FRONTAL (Warp & Target)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_1_resized, "YOLOP MASK (RAW)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_2_resized, "WARPED MASK", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(reverse_resized, "REVERSE", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(follow_resized, "FOLLOW (YOLO)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined_image_row1 = np.hstack([driver_resized, front_resized, debug_1_resized])
    combined_image_row2 = np.hstack([debug_2_resized, reverse_resized, follow_resized])
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

# =========================================================
#   MISC
# =========================================================

def on_trackbar_change(val):
    pass

# =========================================================
#   MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AirSim Car Dashboard with YOLOP road following")
    parser.add_argument(
        "--mode", choices=['manual', 'auto'], default='manual',
        help="Initial control mode (default: manual)"
    )
    parser.add_argument(
        '--use_yolo',
        action='store_true',
        help="Enable YOLOv8 object detection on the follow camera (default: False)"
    )
    args = parser.parse_args()

    # Connect to AirSim Car
    try:
        print("Connected!")
        client = airsim.CarClient()
        client.confirmConnection()
        client.enableApiControl(True)
        print("Connected to AirSim Car! API Control Enabled.")
    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running: {e}")
        exit()

    # Load YOLOP
    print("Loading YOLOP model... This will download weights on first run.")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        yolop_model = yolop_model.to(device)
        yolop_model.eval()

        print("YOLOP model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOP model: {e}")
        print("Please ensure you have PyTorch and an internet connection.")
        client.enableApiControl(False)
        exit()

    # Optional YOLOv8 for cars (follow cam)
    yolo_model = None
    if args.use_yolo:
        print("Loading YOLOv8 model... This may take a moment (first-time download).")
        try:
            yolo_model = YOLO('yolov8n.pt')
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Disabling YOLO.")
            args.use_yolo = False
    else:
        print("YOLO detection is OFF. To enable, use the --use_yolo flag.")

    print("--- Controls ---")
    print("K: Toggle control mode (Manual/Auto)")
    print("R: Reset Environment")
    print("Q: Quit")
    print("W/S: Throttle/Brake | A/D: Steering | Space: Handbrake")

    # Windows & trackbars
    DASHBOARD_WINDOW = "AirSim Dashboard"
    CONTROLS_WINDOW = "AirSim Controls"

    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.resizeWindow(CONTROLS_WINDOW, 400, 300)

    # ROI trackbars (for perspective warp)
    cv2.createTrackbar("Top Y %",        CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Top X Center %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Top Width %",    CONTROLS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Y %",     CONTROLS_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom X Center %", CONTROLS_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("Bottom Width %", CONTROLS_WINDOW, 100, 100, on_trackbar_change)

    # Control variables & PID
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

    # Blank mask for debug init
    blank_debug_mask = np.zeros((200, 250), dtype=np.uint8)

    while True:
        try:
            # Collision check
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
            was_collided = is_collided_now

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                if control_mode == 'manual':
                    control_mode = 'auto'
                else:
                    control_mode = 'manual'
                pid_integral = 0.0
                pid_previous_error = 0.0
                last_timestamp = time.time()
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
                pid_integral = 0.0
                pid_previous_error = 0.0
                last_timestamp = time.time()
                continue

            # ROI sliders
            top_y_p        = cv2.getTrackbarPos("Top Y %",        CONTROLS_WINDOW)
            top_x_p        = cv2.getTrackbarPos("Top X Center %", CONTROLS_WINDOW)
            top_w_p        = cv2.getTrackbarPos("Top Width %",    CONTROLS_WINDOW)
            bottom_y_p     = cv2.getTrackbarPos("Bottom Y %",     CONTROLS_WINDOW)
            bottom_x_p     = cv2.getTrackbarPos("Bottom X Center %", CONTROLS_WINDOW)
            bottom_w_p     = cv2.getTrackbarPos("Bottom Width %", CONTROLS_WINDOW)

            # Get images
            images = get_images(client)
            raw_frontal     = images['frontal']
            follow_cam_image = images['follow']

            # Optional YOLOv8 on follow cam
            if args.use_yolo and yolo_model is not None:
                try:
                    yolo_results = yolo_model.predict(follow_cam_image, classes=2, verbose=False)
                    if yolo_results:
                        result = yolo_results[0]
                        for box in result.boxes:
                            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                            conf = float(box.conf[0])
                            cv2.rectangle(follow_cam_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(follow_cam_image, f"Car {conf:.2f}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 2)
                except Exception as y_e:
                    print(f"[WARN] YOLOv8 inference error: {y_e}")

            # Validate frontal image
            if raw_frontal is None or raw_frontal.size == 0:
                print("[WARN] raw_frontal is empty, skipping frame.")
                continue

            h, w, _ = raw_frontal.shape
            if h <= 0 or w <= 0:
                print(f"[WARN] raw_frontal has invalid size ({w}x{h}), skipping frame.")
                continue

            # ROI / warp points
            roi_poly_vertices, roi_warp_src = calculate_roi_points(
                h, w,
                top_y_p, top_x_p, top_w_p,
                bottom_y_p, bottom_x_p, bottom_w_p
            )

            frontal_with_roi = draw_roi(raw_frontal, roi_poly_vertices)

            # YOLOP mask
            final_road_mask = get_yolop_mask(raw_frontal, yolop_model, device)

            yolop_mask_debug_for_dash   = final_road_mask.copy()
            warped_mask_debug_for_dash  = blank_debug_mask
            final_frontal_image         = frontal_with_roi
            current_steering_value      = None

            # MANUAL MODE
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

            # AUTO MODE
            else:
                warped_mask, matrix_inv = perspective_warp(final_road_mask, roi_warp_src)
                warped_mask_debug_for_dash = warped_mask

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
                        derivative = 0.0
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

            # Steering lines overlay
            if current_steering_value is not None:
                final_frontal_image = draw_steering_lines(final_frontal_image, current_steering_value)

            # Dashboard
            car_state = client.getCarState()
            dashboard = create_dashboard(
                final_frontal_image,
                images['driver'],
                images['reverse'],
                follow_cam_image,
                yolop_mask_debug_for_dash,
                warped_mask_debug_for_dash,
                car_state,
                control_mode,
                current_steering_value
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
                if "mean of empty" in str(e):
                    print(f"CV Error (YOLOP mask likely empty or warp failed): {e}")
                    if control_mode == 'auto':
                        pid_previous_error = 0.0
                        car_controls.steering = 0.0
                        car_controls.throttle = 0.1
                        client.setCarControls(car_controls)
                else:
                    print(f"An error occurred in the loop: {e}")
                    import traceback
                    traceback.print_exc()
                    break

    # Cleanup
    car_controls.throttle = 0.0
    car_controls.steering = 0.0
    car_controls.handbrake = True
    client.setCarControls(car_controls)
    client.enableApiControl(False)

    print("API Control Disabled. Exiting.")
    cv2.destroyAllWindows()
