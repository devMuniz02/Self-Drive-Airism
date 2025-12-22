import airsim
import numpy as np
import cv2
import time
import argparse
from ultralytics import YOLO 

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
        # 1. Create a Vector3r object for the target position
        target_position = airsim.Vector3r(TARGET_X, TARGET_Y, TARGET_Z)
        
        # 2. Get the current orientation to maintain the car's direction
        current_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
        current_orientation = current_pose.orientation

        # 3. Create the target pose
        target_pose = airsim.Pose(target_position, current_orientation)
        
        # 4. Set the vehicle's pose
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
# 3-STAGE (RGB -> HSV -> GRAY) ROAD DETECTION
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
         (top_left_x,  top_y),
         (top_right_x,  top_y),
         (bottom_right_x, bottom_y)]
    ], dtype=np.int32)
    
    vertices_for_warp = np.float32([
        (top_left_x,  top_y),
        (top_right_x,  top_y),
        (bottom_left_x, bottom_y),
        (bottom_right_x, bottom_y)
    ])
    
    return vertices_for_poly, vertices_for_warp

def draw_roi(image, vertices):
    img_copy = image.copy()
    cv2.polylines(img_copy, vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    return img_copy

def process_image_3_stage_filter(img, roi_vertices, blur_kernel_size, 
                                 rgb_r_low, rgb_r_high,
                                 rgb_g_low, rgb_g_high,
                                 rgb_b_low, rgb_b_high,
                                 hsv_h_low, hsv_h_high,
                                 hsv_s_low, hsv_s_high,
                                 hsv_v_low, hsv_v_high,
                                 gray_low_thresh, gray_high_thresh):
    
    # Stage 1: RGB Filtering
    bgr_low = np.array([rgb_b_low, rgb_g_low, rgb_r_low])
    bgr_high = np.array([rgb_b_high, rgb_g_high, rgb_r_high])
    rgb_mask = cv2.inRange(img, bgr_low, bgr_high)
    rgb_filtered_img = cv2.bitwise_and(img, img, mask=rgb_mask) # <-- Intermediate Image 1

    # Stage 2: HSV Filtering (applied to RGB filtered image)
    hsv_img = cv2.cvtColor(rgb_filtered_img, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([hsv_h_low, hsv_s_low, hsv_v_low])
    hsv_high = np.array([hsv_h_high, hsv_s_high, hsv_v_high])
    hsv_mask = cv2.inRange(hsv_img, hsv_low, hsv_high)
    hsv_filtered_img = cv2.bitwise_and(rgb_filtered_img, rgb_filtered_img, mask=hsv_mask) # <-- Intermediate Image 2

    # Stage 3: Grayscale Filtering (applied to HSV filtered image)
    gray_from_filtered = cv2.cvtColor(hsv_filtered_img, cv2.COLOR_BGR2GRAY)
    blurry_gray = cv2.GaussianBlur(gray_from_filtered, (blur_kernel_size, blur_kernel_size), 0)
    final_road_mask_grayscale = cv2.inRange(blurry_gray, gray_low_thresh, gray_high_thresh) # <-- Intermediate Image 3 (Mask)
    
    # Apply ROI
    height, width = final_road_mask_grayscale.shape
    roi_mask = np.zeros_like(final_road_mask_grayscale)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    
    # Final mask for tracking (ROI applied to Grayscale mask)
    final_road_mask_for_tracker = cv2.bitwise_and(final_road_mask_grayscale, roi_mask)
    
    # Return 5 images: Tracker Mask, RGB, HSV, Grayscale Mask, Final Mask (for dashboard clarity)
    return final_road_mask_for_tracker, rgb_filtered_img, hsv_filtered_img, final_road_mask_grayscale, final_road_mask_for_tracker

# -----------------------------------------------------------------
# BLOB STEERING & VISUALIZATION
# -----------------------------------------------------------------

def calculate_steering_from_blob(warped_mask, cX_override=None, max_angle_deg=30.0):
    """
    Calculates steering using the center of mass, optionally using an overridden cX value.
    Returns: steering, viz_point (actual center or overridden point for visualization)
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

    # Calculate actual center of mass (CoM)
    actual_cX = int(M["m10"] / M["m00"])
    actual_cY = int(M["m01"] / M["m00"])

    # Determine X-coordinate to use for steering calculation (CoM or Clamped value)
    steering_cX = cX_override if cX_override is not None else actual_cX
    
    # Visualization point uses the steering_cX but the actual cY
    viz_point = (steering_cX, actual_cY)

    car_center_x = w / 2.0
    bottom_y = h - 1

    dx = steering_cX - car_center_x
    dy = bottom_y - actual_cY

    if dy == 0:
        angle_rad = 0.0
    else:
        angle_rad = np.arctan2(dx, dy)

    max_angle_rad = np.deg2rad(max_angle_deg)
    steering = float(np.clip(angle_rad / max_angle_rad, -1.0, 1.0))

    return steering, viz_point

def perspective_warp(img, src_points):
    height, width = img.shape[:2]
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_img = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix_inv

def draw_center_visuals(original_img, warped_mask, center_point, matrix_inv):
    if center_point is None:
        return original_img

    h, w = warped_mask.shape
    warp_zero = np.zeros_like(warped_mask).astype(np.uint8)
    color_warp = cv2.cvtColor(warp_zero, cv2.COLOR_GRAY2BGR)

    # Draw the calculated/clamped center point
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

# -----------------------------------------------------------------
# LANE BOUNDARY LOGIC 
# -----------------------------------------------------------------

def get_x_at_y(coeffs, y, w):
    """
    Solves the linear equation for x at a given y: y = m*x + c
    Coefficients are [m, c].
    Returns the x value.
    """
    m, c = coeffs
    
    if abs(m) < 1e-6: # Near-horizontal line
        return None 
        
    # x = (y - c) / m
    x = (y - c) / m
    return x


def adjust_blob_center_to_lane_boundary(warped_mask_h, warped_mask_w, original_cX,
                                         left_coeffs, right_coeffs):
    """
    Clamps the blob's center-of-mass (cX) to the line boundaries if it crosses them.
    This generates a strong self-correcting steering signal.
    """
    if left_coeffs is None or right_coeffs is None:
        return original_cX
    
    # Check close to the car (bottom of the warped image)
    y_check = warped_mask_h - 1 

    # Calculate Lane X Positions at Y_Check
    left_x = get_x_at_y(left_coeffs, y_check, warped_mask_w)
    right_x = get_x_at_y(right_coeffs, y_check, warped_mask_w)

    if left_x is None or right_x is None:
        return original_cX
        
    # Ensure left_x is indeed to the left of right_x, swap if necessary
    if left_x > right_x:
        left_x, right_x = right_x, left_x
        
    # Apply Clamping:
    
    # 1. Too far left: Clamp to the left line's X position
    if original_cX < left_x:
        return int(left_x)

    # 2. Too far right: Clamp to the right line's X position
    elif original_cX > right_x:
        return int(right_x)

    # 3. Within bounds: Return original X
    return original_cX


# -----------------------------------------------------------------
# DRAW LINES: LINEAR REGRESSION 
# -----------------------------------------------------------------

def draw_long_road_lines(base_image,
                         hsv_filtered_img,
                         canny_low,
                         canny_high,
                         min_line_length=120,
                         max_line_gap=20,
                         min_angle_from_horizontal_deg=10,
                         angle_merge_eps_deg=5.0,
                         endpoint_dist_thresh=40.0,
                         roi_vertices=None):
    """
    Uses linear regression (deg=1) for line fitting.
    Identifies and draws lane lines. Returns the coefficients of the leftmost and 
    rightmost linear curves found: (m, c).
    """

    if hsv_filtered_img is None or hsv_filtered_img.size == 0:
        return base_image, None, None # Return None for left/right coeffs

    # --- Canny sobre HSV filtrado ---
    if len(hsv_filtered_img.shape) == 3:
        gray = cv2.cvtColor(hsv_filtered_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = hsv_filtered_img.copy()

    gray = gray.astype(np.uint8)

    # Aplicar ROI también a Canny si existe
    if roi_vertices is not None:
        roi_mask = np.zeros_like(gray)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

    if canny_high <= canny_low:
        canny_high = min(canny_low + 1, 255)

    edges = cv2.Canny(gray, canny_low, canny_high)

    # --- HoughLinesP ---
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=max_line_gap
    )

    img_out = base_image.copy()
    if lines is None:
        return img_out, None, None

    segments = []  # (x1, y1, x2, y2, angle_deg)

    # --- Filtrar segmentos por longitud y ángulo ---
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_line_length:
            continue

        dx = x2 - x1
        dy = y2 - y1
        angle_deg = np.degrees(np.arctan2(dy, dx))  # -180..180

        # distancia al horizontal (0º o 180º)
        angle_from_horizontal = min(abs(angle_deg),
                                    abs(abs(angle_deg) - 180))

        if angle_from_horizontal < min_angle_from_horizontal_deg:
            # muy horizontal → no lo usamos para las líneas de carretera
            continue

        # Dibuja segmento original en VERDE
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        segments.append((x1, y1, x2, y2, angle_deg))

    if not segments:
        return img_out, None, None

    # --- Agrupar segmentos por ángulo y cercanía ---
    groups = []  # cada grupo: {"angle_mean": float, "points": [(x,y), ...], "coeffs": None, "x_min": float, "x_max": float}

    def min_dist_to_group(points_list, p):
        px, py = p
        dmin = float("inf")
        for gx, gy in points_list:
            d = np.hypot(px - gx, py - gy)
            if d < dmin:
                dmin = d
        return dmin

    for (x1, y1, x2, y2, angle_deg) in segments:
        p1 = (x1, y1)
        p2 = (x2, y2)
        assigned = False
        current_segment_min_x = min(x1, x2)
        current_segment_max_x = max(x1, x2)


        for g in groups:
            # comprobar ángulo con el promedio del grupo
            if abs(angle_deg - g["angle_mean"]) <= angle_merge_eps_deg:
                d1 = min_dist_to_group(g["points"], p1)
                d2 = min_dist_to_group(g["points"], p2)
                # solo se une si ALGUNO de los extremos está cerca del grupo
                if min(d1, d2) <= endpoint_dist_thresh:
                    g["points"].append(p1)
                    g["points"].append(p2)
                    # actualizar ángulo promedio sencillo
                    old_n = len(g["points"]) - 2
                    g["angle_mean"] = (g["angle_mean"] * old_n + 2 * angle_deg) / (old_n + 2)
                    
                    # Update min/max x for the assigned group
                    g["x_min"] = min(g["x_min"], current_segment_min_x)
                    g["x_max"] = max(g["x_max"], current_segment_max_x)
                    
                    assigned = True
                    break

        if not assigned:
            groups.append({
                "angle_mean": angle_deg,
                "points": [p1, p2],
                "coeffs": None, 
                "x_min": current_segment_min_x,
                "x_max": current_segment_max_x,
            })

    h, w = gray.shape

    # Keep track of the leftmost and rightmost line coefficients
    left_lane_coeffs = None
    right_lane_coeffs = None
    
    # Sort groups by the mean X coordinate of their points for reliable left/right identification
    sorted_groups = sorted(groups, key=lambda g: np.mean(np.array(g["points"])[:, 0]) if len(g["points"]) >= 2 else w/2)

    # --- Para cada grupo, ajustar LINEAL y dibujarlo en ROJO ---
    for i, g in enumerate(sorted_groups):
        pts = np.array(g["points"], dtype=np.float32)
        if pts.shape[0] < 3:
            continue

        xs = pts[:, 0]
        ys = pts[:, 1]

        # Ajuste lineal: y = m x + c (deg=1)
        try:
            # Fit x vs y because line-finding works better on the y-coordinate range
            coeffs = np.polyfit(xs, ys, deg=1) # coeffs = [m, c] where y = m*x + c
            g["coeffs"] = coeffs # STORE COEFFS IN GROUP
        except Exception as e:
            print(f"[WARN] Linear fit failed for group: {e}")
            continue

        # If this is the first group with valid coefficients, it's the leftmost
        if left_lane_coeffs is None:
            left_lane_coeffs = coeffs
            
        # The last valid coefficients will be the rightmost one
        right_lane_coeffs = coeffs

        m, c = coeffs

        # Generar línea suave dentro del rango de x de ese grupo
        x_min = int(g["x_min"])
        x_max = int(g["x_max"])
        
        # Extend the line to the bottom of the ROI or image
        x_vals_start = int(max(0, x_min - 20))
        x_vals_end = int(min(w, x_max + 20))

        if x_vals_end <= x_vals_start:
            continue

        x_vals = np.linspace(x_vals_start, x_vals_end, num=80)
        y_vals = m * x_vals + c

        x_vals = x_vals.astype(np.int32)
        y_vals = y_vals.astype(np.int32)

        # Limitar a los bordes de la imagen
        valid = (x_vals >= 0) & (x_vals < w) & (y_vals >= 0) & (y_vals < h)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]

        if len(x_vals) < 2:
            continue

        curve_points = np.stack((x_vals, y_vals), axis=1).reshape(-1, 1, 2)

        # Dibujar línea en ROJO
        cv2.polylines(img_out, [curve_points], isClosed=False, color=(0, 0, 255), thickness=4)

    return img_out, left_lane_coeffs, right_lane_coeffs


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
# Dashboard: camera grid + footer
# -----------------------------------------------------------------

def create_dashboard(frontal_img, driver_img, follow_img, 
                     rgb_filtered, hsv_filtered, gray_mask,
                     canny_img, final_road_mask_debug,
                     car_state, control_mode, steering_value=None):
    
    STD_HEIGHT = 240
    STD_WIDTH  = 320  
    
    try:
        # Top Row (Processing Pipeline)
        front_resized   = resize_keep_aspect(frontal_img, STD_WIDTH, STD_HEIGHT)
        rgb_resized     = resize_keep_aspect(rgb_filtered, STD_WIDTH, STD_HEIGHT)
        hsv_resized     = resize_keep_aspect(hsv_filtered, STD_WIDTH, STD_HEIGHT)
        
        # Convert grayscale mask to BGR for concatenation
        gray_bgr = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        gray_resized    = resize_keep_aspect(gray_bgr, STD_WIDTH, STD_HEIGHT)

        # Bottom Row (Debug/Other Cams)
        mask_bgr = cv2.cvtColor(final_road_mask_debug, cv2.COLOR_GRAY2BGR)
        mask_resized    = resize_keep_aspect(mask_bgr, STD_WIDTH, STD_HEIGHT)
        canny_resized   = resize_keep_aspect(canny_img, STD_WIDTH, STD_HEIGHT)
        follow_resized  = resize_keep_aspect(follow_img,  STD_WIDTH, STD_HEIGHT)
        driver_resized  = resize_keep_aspect(driver_img,  STD_WIDTH, STD_HEIGHT)

    except Exception as e:
        print(f"Error resizing images: {e}")
        blank_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        driver_resized = front_resized = rgb_resized = hsv_resized = gray_resized = mask_resized = canny_resized = follow_resized = blank_img
    
    # Add labels to the processing pipeline
    cv2.putText(front_resized, "FRONTAL (Lines/Blob)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(rgb_resized, "1. RGB FILTERED", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(hsv_resized, "2. HSV FILTERED", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(gray_resized, "3. GRAYSCALE MASK", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add labels to the debug/other cam row
    cv2.putText(mask_resized, "FINAL MASK (with ROI)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canny_resized, "CANNY/EDGES", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(follow_resized, "FOLLOW (YOLO)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(driver_resized, "DRIVER CAM", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined_image_row1 = np.hstack([front_resized, rgb_resized, hsv_resized, gray_resized])
    combined_image_row2 = np.hstack([mask_resized, canny_resized, follow_resized, driver_resized])
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
    
    parser = argparse.ArgumentParser(description="AirSim Car Dashboard")
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

    # Trackbars for ROI & filters
    cv2.createTrackbar("ROI_TopY(%)",   CTRL_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_BotY(%)",   CTRL_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_TopW(%)",   CTRL_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_BotW(%)",   CTRL_WINDOW, 100, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_TopX(%)",   CTRL_WINDOW, 50, 100, on_trackbar_change)
    cv2.createTrackbar("ROI_BotX(%)",   CTRL_WINDOW, 50, 100, on_trackbar_change)

    cv2.createTrackbar("MaxThrAuto(%)", CTRL_WINDOW, 35, 100, on_trackbar_change)
    cv2.createTrackbar("BlurKSize",     CTRL_WINDOW, 7,  21,  on_trackbar_change)

    cv2.createTrackbar("R_Low",   CTRL_WINDOW, 0,   255, on_trackbar_change)
    cv2.createTrackbar("R_High",  CTRL_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("G_Low",   CTRL_WINDOW, 0,   255, on_trackbar_change)
    cv2.createTrackbar("G_High",  CTRL_WINDOW, 255, 255, on_trackbar_change)
    cv2.createTrackbar("B_Low",   CTRL_WINDOW, 0,   255, on_trackbar_change)
    cv2.createTrackbar("B_High",  CTRL_WINDOW, 255, 255, on_trackbar_change)

    cv2.createTrackbar("H_Low",   CTRL_WINDOW, 0,   179, on_trackbar_change)
    cv2.createTrackbar("H_High",  CTRL_WINDOW, 179, 179, on_trackbar_change)
    cv2.createTrackbar("S_Low",   CTRL_WINDOW, 0,   255, on_trackbar_change)
    cv2.createTrackbar("S_High",  CTRL_WINDOW, 50,  255, on_trackbar_change)
    cv2.createTrackbar("V_Low",   CTRL_WINDOW, 0,   255, on_trackbar_change)
    cv2.createTrackbar("V_High",  CTRL_WINDOW, 255, 255, on_trackbar_change)

    cv2.createTrackbar("GrayLow",   CTRL_WINDOW, 50,  255, on_trackbar_change)
    cv2.createTrackbar("GrayHigh",  CTRL_WINDOW, 255, 255, on_trackbar_change)

    # Canny controls (sobre HSV filtrado)
    cv2.createTrackbar("CannyLow",    CTRL_WINDOW, 50, 255, on_trackbar_change)
    cv2.createTrackbar("CannyHigh",   CTRL_WINDOW, 150, 255, on_trackbar_change)

    # Angle, min line length & angle merge epsilon
    cv2.createTrackbar("MinAngFromHor", CTRL_WINDOW, 10, 89, on_trackbar_change)
    cv2.createTrackbar("MinLineLen",    CTRL_WINDOW, 120, 500, on_trackbar_change)
    cv2.createTrackbar("MergeAngEps",   CTRL_WINDOW, 5, 45,  on_trackbar_change)  # slider para ángulo entre líneas
    
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

            # Read sliders
            top_y_p     = cv2.getTrackbarPos("ROI_TopY(%)", CTRL_WINDOW)
            bottom_y_p  = cv2.getTrackbarPos("ROI_BotY(%)", CTRL_WINDOW)
            top_w_p     = cv2.getTrackbarPos("ROI_TopW(%)", CTRL_WINDOW)
            bottom_w_p  = cv2.getTrackbarPos("ROI_BotW(%)", CTRL_WINDOW)
            top_x_p     = cv2.getTrackbarPos("ROI_TopX(%)", CTRL_WINDOW)
            bottom_x_p  = cv2.getTrackbarPos("ROI_BotX(%)", CTRL_WINDOW)

            throttle_max_percent = cv2.getTrackbarPos("MaxThrAuto(%)", CTRL_WINDOW)
            BASE_THROTTLE_VALUE = throttle_max_percent / 100.0

            blur_kernel_size = cv2.getTrackbarPos("BlurKSize", CTRL_WINDOW)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            if blur_kernel_size < 1:
                blur_kernel_size = 1

            rgb_r_low   = cv2.getTrackbarPos("R_Low",   CTRL_WINDOW)
            rgb_r_high  = cv2.getTrackbarPos("R_High", CTRL_WINDOW)
            rgb_g_low   = cv2.getTrackbarPos("G_Low",   CTRL_WINDOW)
            rgb_g_high  = cv2.getTrackbarPos("G_High", CTRL_WINDOW)
            rgb_b_low   = cv2.getTrackbarPos("B_Low",   CTRL_WINDOW)
            rgb_b_high  = cv2.getTrackbarPos("B_High", CTRL_WINDOW)

            hsv_h_low   = cv2.getTrackbarPos("H_Low",   CTRL_WINDOW)
            hsv_h_high  = cv2.getTrackbarPos("H_High",  CTRL_WINDOW)
            hsv_s_low   = cv2.getTrackbarPos("S_Low",   CTRL_WINDOW)
            hsv_s_high  = cv2.getTrackbarPos("S_High",  CTRL_WINDOW)
            hsv_v_low   = cv2.getTrackbarPos("V_Low",   CTRL_WINDOW)
            hsv_v_high  = cv2.getTrackbarPos("V_High",  CTRL_WINDOW)

            gray_low    = cv2.getTrackbarPos("GrayLow",   CTRL_WINDOW)
            gray_high   = cv2.getTrackbarPos("GrayHigh",  CTRL_WINDOW)

            canny_low   = cv2.getTrackbarPos("CannyLow",  CTRL_WINDOW)
            canny_high  = cv2.getTrackbarPos("CannyHigh", CTRL_WINDOW)
            if canny_high <= canny_low:
                canny_high = min(canny_low + 1, 255)

            min_angle_from_horizontal = cv2.getTrackbarPos("MinAngFromHor", CTRL_WINDOW)
            min_line_len              = cv2.getTrackbarPos("MinLineLen",    CTRL_WINDOW)
            if min_line_len < 10:
                min_line_len = 10

            merge_angle_eps           = cv2.getTrackbarPos("MergeAngEps",   CTRL_WINDOW)
            if merge_angle_eps < 1:
                merge_angle_eps = 1
            
            images = get_images(client)
            raw_frontal = images['frontal']
            follow_cam_image = images['follow']
            driver_cam_image = images['driver'] # Get driver cam for dashboard

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
            
            # Unpack the 5 returned images
            final_road_mask, rgb_filtered_img, hsv_filtered_img, final_road_mask_grayscale, road_mask_debug_for_dash = \
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

            # Canny using HSV filtered image + ROI (for display)
            # The hsv_filtered_img is already in BGR format, use it for Canny base.
            hsv_gray = cv2.cvtColor(hsv_filtered_img, cv2.COLOR_BGR2GRAY)
            roi_mask_canny = np.zeros_like(hsv_gray)
            cv2.fillPoly(roi_mask_canny, roi_poly_vertices, 255)
            hsv_gray_roi = cv2.bitwise_and(hsv_gray, hsv_gray, mask=roi_mask_canny)

            edges = cv2.Canny(hsv_gray_roi, canny_low, canny_high)
            canny_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # For dashboard display

            # --- LINE DETECTION & COEFFICIENT EXTRACTION (LINEAR) ---
            final_frontal_image, left_coeffs, right_coeffs = draw_long_road_lines(
                frontal_with_roi,
                hsv_filtered_img,
                canny_low,
                canny_high,
                min_line_len,
                20, 
                min_angle_from_horizontal,
                merge_angle_eps,
                40.0, 
                roi_poly_vertices
            )
            
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
            
            else: # control_mode == 'auto'
                warped_mask, matrix_inv = perspective_warp(final_road_mask, roi_warp_src)
                
                # --- STEP 1: Get the actual CoM of the blob ---
                _, unadjusted_center_point = calculate_steering_from_blob(warped_mask, cX_override=None)
                
                if unadjusted_center_point is None:
                    adjusted_cX = None
                else:
                    unadjusted_cX = unadjusted_center_point[0]
                    
                    # --- STEP 2: Clamp/Adjust the CoM based on lane lines ---
                    adjusted_cX = adjust_blob_center_to_lane_boundary(
                        warped_mask.shape[0], warped_mask.shape[1], unadjusted_cX, 
                        left_coeffs, right_coeffs
                    )
                
                # --- STEP 3: Calculate final steering using the Adjusted cX ---
                auto_steering, viz_point = calculate_steering_from_blob(
                    warped_mask, cX_override=adjusted_cX
                )
                
                if viz_point is None:
                    auto_steering = 0.0

                auto_steering = float(np.clip(auto_steering, -1.0, 1.0))

                # Since the input cX is already clamped, the steering calculation naturally handles the boundary.
                
                throttle_reduction = abs(auto_steering) * 0.4
                auto_throttle = max(0.15, BASE_THROTTLE_VALUE - throttle_reduction)

                car_controls.throttle = auto_throttle
                car_controls.steering = auto_steering
                car_controls.brake = 0.0
                car_controls.handbrake = False
                client.setCarControls(car_controls)
                
                current_steering_value = auto_steering

                if viz_point is not None:
                    final_frontal_image = draw_center_visuals(
                        final_frontal_image, warped_mask, viz_point, matrix_inv
                    )
            
            # Draw steering lines on the frontal image
            final_frontal_image = draw_steering_lines(final_frontal_image, current_steering_value)

            # Get current car state
            car_state = client.getCarState()
            
            # Create and show dashboard
            dashboard = create_dashboard(
                final_frontal_image, driver_cam_image, follow_cam_image, 
                rgb_filtered_img, hsv_filtered_img, final_road_mask_grayscale,
                canny_bgr, road_mask_debug_for_dash,
                car_state, control_mode, current_steering_value
            )
            cv2.imshow(DASHBOARD_WINDOW, dashboard)
            
        except Exception as e:
            print(f"[LOOP ERROR] {e}")
            break

    client.enableApiControl(False)
    client.reset()
    cv2.destroyAllWindows()
    print("Exiting.")