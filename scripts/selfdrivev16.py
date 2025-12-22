import airsim
import numpy as np
import cv2
import time
import argparse
from ultralytics import YOLO  # For car detection

# --- Configuration Mappings ---
CAMERA_MAP = {
    "frontal": "0", "driver": "3", "reverse": "4", "follow": "follow_cam"
}
TYPE_MAP = {
    "original": (airsim.ImageType.Scene, False, True),
}

# --- Target NED Coordinates (in meters) ---
TARGET_X = 595.63  # North
TARGET_Y = -258.19 # East (Negative is West)
TARGET_Z = -0.68   # Down (Negative is Up/Above ground plane)
VEHICLE_NAME = "" 


def teleport_car_to_start_pose(client):
    """Sets the car's position to the predefined TARGET_X, Y, Z coordinates."""
    try:
        target_position = airsim.Vector3r(TARGET_X, TARGET_Y, TARGET_Z)
        current_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
        current_orientation = current_pose.orientation
        target_pose = airsim.Pose(target_position, current_orientation)
        client.simSetVehiclePose(target_pose, ignore_collision=True, vehicle_name=VEHICLE_NAME)
        print(f"✅ Car teleported to: X={TARGET_X:.2f}m, Y={TARGET_Y:.2f}m, Z={TARGET_Z:.2f}m")
        return True
    except Exception as e:
        print(f"❌ Error during car teleport: {e}")
        return False


# --- Get AirSim images ---
def get_images(client):
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

# -----------------------------------------------------------------
# ROI
# -----------------------------------------------------------------

def calculate_roi_points(height, width,
                         top_y_p, top_x_p, top_w_p,
                         bottom_y_p, bottom_x_p, bottom_w_p):
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
        [(bottom_left_x, bottom_y),
         (top_left_x,   top_y),
         (top_right_x,  top_y),
         (bottom_right_x, bottom_y)]
    ], dtype=np.int32)
    
    vertices_for_warp = np.float32([
        (top_left_x,   top_y),
        (top_right_x,  top_y),
        (bottom_left_x, bottom_y),
        (bottom_right_x, bottom_y)
    ])
    
    return vertices_for_poly, vertices_for_warp

def draw_roi(image, vertices):
    img_copy = image.copy()
    cv2.polylines(img_copy, vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    return img_copy

# -----------------------------------------------------------------
# BLOB STEERING & VISUALIZATION
# -----------------------------------------------------------------

def calculate_steering_from_blob(warped_mask, max_angle_deg=30.0):
    """
    warped_mask: bird-eye binary image (0-255). We compute centroid and angle to bottom-center.
    """
    if len(warped_mask.shape) == 3:
        gray = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_mask

    h, w = gray.shape
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    M = cv2.moments(binary)
    if M["m00"] == 0:
        return 0.0, None

    viz_cX = int(M["m10"] / M["m00"])
    viz_cY = int(M["m01"] / M["m00"])
    viz_point = (viz_cX, viz_cY)

    car_center_x = w / 2.0
    bottom_y = h - 1

    dx = viz_cX - car_center_x
    dy = bottom_y - viz_cY

    if dy == 0:
        angle_rad = 0.0
    else:
        angle_rad = np.arctan2(dx, dy)

    max_angle_rad = np.deg2rad(max_angle_deg)
    steering = float(np.clip(angle_rad / max_angle_rad, -1.0, 1.0))

    return steering, viz_point


def perspective_warp(img, src_points):
    """
    Warps img using src_points to a full-rect bird-eye. Also returns inverse matrix.
    """
    height, width = img.shape[:2]
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix_inv


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

# -----------------------------------------------------------------
# Simplified lane binary: yellow + vertical gradient (with GUI params)
# -----------------------------------------------------------------

def combined_lane_threshold(img,
                            h_low, h_high,
                            s_low, s_high,
                            v_low, v_high,
                            gradx_low):
    """
    Returns a binary image (0/1) of lane pixels:
      - Color: yellow in HSV in [h_low,h_high], [s_low,s_high], [v_low,v_high]
      - Gradient: strong vertical edges (Sobel x) above gradx_low
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv,
                              (h_low,  s_low, v_low),
                              (h_high, s_high, v_high))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    max_val = np.max(abs_sobelx)
    if max_val < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)

    scaled_sobel = np.uint8(255 * abs_sobelx / (max_val + 1e-6))
    gradx_bin = np.zeros_like(scaled_sobel, dtype=np.uint8)
    gradx_bin[(scaled_sobel >= gradx_low) & (scaled_sobel <= 255)] = 1

    yellow_bin = (yellow_mask > 0).astype(np.uint8)

    combined = np.zeros_like(gradx_bin, dtype=np.uint8)
    combined[(yellow_bin == 1) & (gradx_bin == 1)] = 1

    return combined  # 0/1


# -----------------------------------------------------------------
# Sliding-window + polynomial lane fits
# -----------------------------------------------------------------

def find_lane_pixels(binary_warped):
    """
    Sliding window lane detection in bird's-eye binary image.
    Returns left/right lane pixel coordinates and an output debug image.
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    leftx = nonzerox[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if right_lane_inds.size > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if right_lane_inds.size > 0 else np.array([])

    return leftx, lefty, rightx, righty, out_img


def fit_lane_polynomials(binary_warped):
    """
    Returns left_fit, right_fit (2nd order) and a bird-eye debug image.
    """
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    if leftx.size == 0 or rightx.size == 0:
        return None, None, out_img

    left_fit = np.polyfit(lefty, leftx, 2)   # x = f(y)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, out_img


def draw_lane_region(original_img, binary_warped, left_fit, right_fit, Minv):
    """
    Dibuja el área del carril en bird-eye y la proyecta de vuelta a la vista frontal.
    """
    h, w = binary_warped.shape[:2]
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if left_fit is None or right_fit is None:
        frontal = original_img.copy()
        lane_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
        bird_eye_lane = color_warp
        return frontal, lane_mask, bird_eye_lane

    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]
                         )
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (original_img.shape[1], original_img.shape[0]))
    frontal_with_lane = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

    lane_mask_unwarped = cv2.cvtColor(newwarp, cv2.COLOR_BGR2GRAY)

    return frontal_with_lane, lane_mask_unwarped, color_warp


def lane_detection_pipeline(img,
                            roi_vertices,
                            roi_warp_src,
                            h_low, h_high,
                            s_low, s_high,
                            v_low, v_high,
                            gradx_low):
    """
    Returns:
      - frontal_with_lane (BGR)
      - lane_mask_unwarped (0-255, frontal)
      - bird_eye_lane (BGR)
      - binary_warped_255 (bird-eye lane mask 0-255)
      - lane_bin_orig_255 (frontal bin before ROI, 0-255)
    """
    lane_bin_orig = combined_lane_threshold(
        img, h_low, h_high, s_low, s_high, v_low, v_high, gradx_low
    )
    lane_bin_orig_255 = (lane_bin_orig * 255).astype(np.uint8)

    lane_bin_roi = lane_bin_orig_255.copy()
    if roi_vertices is not None:
        roi_mask = np.zeros_like(lane_bin_roi)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        lane_bin_roi = cv2.bitwise_and(lane_bin_roi, roi_mask)

    binary_warped, Minv = perspective_warp(lane_bin_roi, roi_warp_src)
    binary_warped_bin = (binary_warped > 0).astype(np.uint8)

    left_fit, right_fit, _ = fit_lane_polynomials(binary_warped_bin)
    frontal_with_lane, lane_mask_unwarped, bird_eye_lane = \
        draw_lane_region(img, binary_warped_bin, left_fit, right_fit, Minv)

    binary_warped_255 = (binary_warped_bin * 255).astype(np.uint8)

    return frontal_with_lane, lane_mask_unwarped, bird_eye_lane, binary_warped_255, lane_bin_orig_255

# -----------------------------------------------------------------
# Resize with aspect ratio (letterbox into cell)
# -----------------------------------------------------------------

def resize_keep_aspect(img, target_width, target_height):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=resized.dtype)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

# -----------------------------------------------------------------
# Dashboard: 2 × 4 grid + footer
# -----------------------------------------------------------------

def create_dashboard(frontal_img_with_lanes,
                     driver_img,
                     raw_frontal,
                     follow_img,
                     lane_bin_orig_255,
                     lane_mask_unwarped,
                     canny_bgr,
                     bird_eye_lane,
                     car_state,
                     control_mode,
                     steering_value=None):
    
    STD_HEIGHT = 240
    STD_WIDTH  = 320  
    
    try:
        driver_resized   = resize_keep_aspect(driver_img,  STD_WIDTH, STD_HEIGHT)
        front_resized    = resize_keep_aspect(frontal_img_with_lanes, STD_WIDTH, STD_HEIGHT)
        raw_resized      = resize_keep_aspect(raw_frontal, STD_WIDTH, STD_HEIGHT)
        follow_resized   = resize_keep_aspect(follow_img,  STD_WIDTH, STD_HEIGHT)

        lane_bin_bgr     = cv2.cvtColor(lane_bin_orig_255, cv2.COLOR_GRAY2BGR)
        lane_bin_resized = resize_keep_aspect(lane_bin_bgr, STD_WIDTH, STD_HEIGHT)

        lane_mask_bgr    = cv2.cvtColor(lane_mask_unwarped, cv2.COLOR_GRAY2BGR)
        lane_mask_resized= resize_keep_aspect(lane_mask_bgr, STD_WIDTH, STD_HEIGHT)

        canny_resized    = resize_keep_aspect(canny_bgr, STD_WIDTH, STD_HEIGHT)
        bird_resized     = resize_keep_aspect(bird_eye_lane, STD_WIDTH, STD_HEIGHT)

    except Exception as e:
        print(f"Error resizing images: {e}")
        blank_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        driver_resized = front_resized = raw_resized = follow_resized = \
            lane_bin_resized = lane_mask_resized = canny_resized = bird_resized = blank_img
    
    cv2.putText(driver_resized, "DRIVER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(front_resized, "FRONTAL (LANES + ROI)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(raw_resized, "FRONTAL RAW", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(follow_resized, "FOLLOW (YOLO)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(lane_bin_resized, "LANE BINARY (pre-ROI)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(lane_mask_resized, "LANE MASK (frontal)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canny_resized, "CANNY (FRONTAL)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bird_resized, "BIRD-EYE LANES", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    combined_image_row1 = np.hstack([driver_resized, front_resized, raw_resized, follow_resized])
    combined_image_row2 = np.hstack([lane_bin_resized, lane_mask_resized, canny_resized, bird_resized])
    combined_image = np.vstack([combined_image_row1, combined_image_row2])

    cam_h, cam_w, _ = combined_image.shape
    footer_height = int(cam_h * (30.0 / 70.0))

    footer = np.zeros((footer_height, cam_w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20
    padding = 15
    mode_text = f"MODE: {control_mode.upper()} (K: toggle)"
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
        cv2.putText(footer, steer_text, (padding + 220, padding + line_height * 1),
                    font, font_scale, steer_color, font_thickness)
    if car_state.handbrake:
        hb_text = "HANDBRAKE ON"
        (text_width, _), _ = cv2.getTextSize(hb_text, font, font_scale, font_thickness)
        hb_x = cam_w - text_width - padding
        cv2.putText(footer, hb_text, (hb_x, padding + line_height * 0),
                    font, font_scale, (0, 0, 255), font_thickness)

    dashboard = np.vstack([combined_image, footer])
    
    return dashboard

def on_trackbar_change(val):
    pass

# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AirSim Car Dashboard (Advanced Lane Detection + Debug Views)")
    parser.add_argument("--mode", choices=['manual', 'auto'], default='manual')
    parser.add_argument('--use_yolo', action='store_true')
    args = parser.parse_args()
    
    try:
        client = airsim.CarClient() 
        client.confirmConnection()
        client.enableApiControl(True)
        print("Connected to AirSim Car! API Control Enabled.")

        teleport_car_to_start_pose(client)

    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running: {e}")
        exit()

    yolo_model = None
    if args.use_yolo:
        print("Loading YOLOv8 model...")
        try:
            yolo_model = YOLO('yolov8n.pt')
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Disabling YOLO.")
            args.use_yolo = False
    else:
        print("YOLO detection is OFF. Use --use_yolo to enable.")

    print("--- Controls ---")
    print("K: Toggle control mode (Manual/Auto)")
    print("R: Reset Environment")
    print("Q: Quit")
    if args.mode == 'manual':
        print("W/S: Throttle/Brake | A/D: Steering | Space: Handbrake")

    DASHBOARD_WINDOW = "AirSim Dashboard"
    CTRL_WINDOW      = "AirSim Controls"

    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DASHBOARD_WINDOW, 1280, 720)

    cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WINDOW, 450, 720)

    # --- ROI sliders ---
    cv2.createTrackbar("ROI_TopY(%)",  CTRL_WINDOW, 60, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_BotY(%)",  CTRL_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_TopW(%)",  CTRL_WINDOW, 40, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_BotW(%)",  CTRL_WINDOW, 80, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_TopX(%)",  CTRL_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_BotX(%)",  CTRL_WINDOW, 50, 100, on_trackbar_change)

    # --- Auto throttle ---
    cv2.createTrackbar("MaxThrAuto(%)", CTRL_WINDOW, 35, 100, on_trackbar_change)

    # --- Canny sliders (for frontal debug) ---
    cv2.createTrackbar("CannyLow",   CTRL_WINDOW, 50, 255, on_trackbar_change)
    cv2.createTrackbar("CannyHigh",  CTRL_WINDOW, 150, 255, on_trackbar_change)

    # --- Yellow HSV + GradX sliders for lane binary ---
    cv2.createTrackbar("Y_H_low",  CTRL_WINDOW, 15, 179, on_trackbar_change)
    cv2.createTrackbar("Y_H_high", CTRL_WINDOW, 35, 179, on_trackbar_change)
    cv2.createTrackbar("Y_S_low",  CTRL_WINDOW, 80, 255, on_trackbar_change)
    cv2.createTrackbar("Y_S_high", CTRL_WINDOW, 255,255, on_trackbar_change)
    cv2.createTrackbar("Y_V_low",  CTRL_WINDOW, 150,255, on_trackbar_change)
    cv2.createTrackbar("Y_V_high", CTRL_WINDOW, 255,255, on_trackbar_change)
    cv2.createTrackbar("GradX_low", CTRL_WINDOW, 30, 255, on_trackbar_change)

    car_controls = airsim.CarControls()
    throttle = 0.0
    steering = 0.0
    THROTTLE_INC = 0.1
    STEERING_INC = 0.05
    DECAY = 0.1 
    control_mode = args.mode
    was_collided = False 
    
    BASE_THROTTLE_MAX_PERCENT = 35 
    BASE_THROTTLE_VALUE = BASE_THROTTLE_MAX_PERCENT / 100.0
    
    print(f"\nInitial control mode: {control_mode.upper()}")

    while True:
        try:
            collision_info = client.simGetCollisionInfo()
            is_collided_now = collision_info.has_collided
            if is_collided_now and not was_collided:
                print("Collision detected! Resetting controls...")
                throttle = 0.0
                steering = 0.0
                car_controls.throttle = 0.0
                car_controls.steering = 0.0
                client.setCarControls(car_controls)
                client.enableApiControl(True)
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

                teleport_car_to_start_pose(client)
                
                throttle = steering = 0.0
                car_controls.throttle = 0.0
                car_controls.steering = 0.0
                car_controls.handbrake = False
                continue 

            # --- Read sliders ---
            top_y_p     = cv2.getTrackbarPos("ROI_TopY(%)", CTRL_WINDOW)
            bottom_y_p  = cv2.getTrackbarPos("ROI_BotY(%)", CTRL_WINDOW)
            top_w_p     = cv2.getTrackbarPos("ROI_TopW(%)", CTRL_WINDOW)
            bottom_w_p  = cv2.getTrackbarPos("ROI_BotW(%)", CTRL_WINDOW)
            top_x_p     = cv2.getTrackbarPos("ROI_TopX(%)", CTRL_WINDOW)
            bottom_x_p  = cv2.getTrackbarPos("ROI_BotX(%)", CTRL_WINDOW)

            throttle_max_percent = cv2.getTrackbarPos("MaxThrAuto(%)", CTRL_WINDOW)
            BASE_THROTTLE_VALUE = throttle_max_percent / 100.0

            canny_low   = cv2.getTrackbarPos("CannyLow",  CTRL_WINDOW)
            canny_high  = cv2.getTrackbarPos("CannyHigh", CTRL_WINDOW)
            if canny_high <= canny_low:
                canny_high = min(canny_low + 1, 255)

            h_low  = cv2.getTrackbarPos("Y_H_low",  CTRL_WINDOW)
            h_high = cv2.getTrackbarPos("Y_H_high", CTRL_WINDOW)
            s_low  = cv2.getTrackbarPos("Y_S_low",  CTRL_WINDOW)
            s_high = cv2.getTrackbarPos("Y_S_high", CTRL_WINDOW)
            v_low  = cv2.getTrackbarPos("Y_V_low",  CTRL_WINDOW)
            v_high = cv2.getTrackbarPos("Y_V_high", CTRL_WINDOW)
            gradx_low = cv2.getTrackbarPos("GradX_low", CTRL_WINDOW)
            
            images = get_images(client)
            raw_frontal = images['frontal']
            follow_cam_image = images['follow']

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

            # --- Lane detection pipeline with GUI params ---
            (final_frontal_image,
             lane_mask_unwarped,
             bird_eye_lane,
             warped_binary_for_steering,
             lane_bin_orig_255) = lane_detection_pipeline(
                raw_frontal, roi_poly_vertices, roi_warp_src,
                h_low, h_high, s_low, s_high, v_low, v_high, gradx_low
            )

            # Canny edges on raw frontal (debug)
            gray_frontal = cv2.cvtColor(raw_frontal, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_frontal, canny_low, canny_high)
            canny_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

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
            
            else:
                auto_steering, center_point_viz = calculate_steering_from_blob(warped_binary_for_steering)

                if center_point_viz is None:
                    auto_steering = 0.0

                auto_steering = float(np.clip(auto_steering, -1.0, 1.0))

                throttle_reduction = abs(auto_steering) * 0.4
                auto_throttle = max(0.15, BASE_THROTTLE_VALUE - throttle_reduction)

                car_controls.throttle = auto_throttle
                car_controls.steering = auto_steering
                car_controls.handbrake = False
                client.setCarControls(car_controls)
                current_steering_value = auto_steering
            
            # Draw ROI and steering lines
            final_frontal_image = draw_roi(final_frontal_image, roi_poly_vertices)
            if current_steering_value is not None:
                final_frontal_image = draw_steering_lines(final_frontal_image, current_steering_value)

            car_state = client.getCarState()
            
            dashboard = create_dashboard(
                final_frontal_image,
                images['driver'],
                raw_frontal,
                follow_cam_image,
                lane_bin_orig_255,
                lane_mask_unwarped,
                canny_bgr,
                bird_eye_lane,
                car_state,
                control_mode,
                current_steering_value
            )
            
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
                if ("polyfit" in str(e) or
                    "concatenate" in str(e) or
                    "nonzero" in str(e) or
                    "mean of empty" in str(e)):
                    print(f"CV Error (likely no lane found): {e}")
                    if control_mode == 'auto':
                        car_controls.steering = 0.0
                        car_controls.throttle = 0.1 
                        client.setCarControls(car_controls)
                else:
                    print(f"An error occurred in the loop: {e}")
                    break 

    car_controls.throttle = 0.0
    car_controls.steering = 0.0
    car_controls.handbrake = True
    client.setCarControls(car_controls)
    client.enableApiControl(False)
    
    print("API Control Disabled. Exiting.")
    cv2.destroyAllWindows()
