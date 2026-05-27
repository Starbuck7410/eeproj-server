import cv2
import numpy as np
import config




def detect_colored_balls(frame):
    # Blurring & HSV (Only done once)
    # blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Simplified color ranges for lookup
    color_ranges = {
        "red":    [(np.array([0,   100, 100]), np.array([10,  255, 255])),
                   (np.array([150, 100, 100]), np.array([180, 255, 255]))],
        "orange": [(np.array([10,  100, 100]), np.array([20,  255, 255]))],
        "yellow": [(np.array([20,  100, 100]), np.array([40,  255, 255]))],
        "green":  [(np.array([40,  100, 100]), np.array([80,  255, 255]))],
        "cyan":   [(np.array([80,  150, 200]), np.array([100, 255, 255]))],
        "blue":   [(np.array([100, 100, 100]), np.array([110, 255, 255]))],
        "purple": [(np.array([110, 100, 100]), np.array([150, 180, 180]))],
    }

    detected_circles = []

    # Create one master mask for all colors to find blobs
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in color_ranges.values():
        for lower, upper in ranges:
            full_mask |= cv2.inRange(hsv, lower, upper)

    # Fast Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    # Use Contours instead of HoughCircles (Much Faster)
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detection_id = 0
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < config.MIN_AERA or perimeter == 0:  # Filter noise
            continue
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity < config.CIRCULARITY_THRESH: continue # Ignore non-round objects

        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        if radius > 5:
            # Classify color by sampling the center pixel in HSV
            cx, cy, cr = int(x), int(y), int(radius)
            sample_pixel = hsv[cy, cx]
            
            detected_color = "unknown"
            for name, ranges in color_ranges.items():
                for lower, upper in ranges:
                    if np.all(sample_pixel >= lower) and np.all(sample_pixel <= upper):
                        detected_color = name
                        break
            
            if detected_color == "unknown": 
                continue
            
            detected_circles.append({
                "color": detected_color,
                "x": cx, "y": cy, "radius": cr, "id": detection_id
            })
            # Drawing on original frame
            cv2.circle(frame, (cx, cy), cr, (0, 255, 0), 2)
            cv2.putText(frame, f"{detected_color}, {detection_id}", (cx, cy - cr), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(frame, f"{detected_color}, {detection_id}", (cx, cy - cr), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            detection_id += 1

    return detected_circles, frame

def get_tag_center(tag_corners):
    """
    Returns the center pixel coordinate of an AprilTag.
    
    tag_corners format:
    [[x1, y1], [x2, y2],
     [x3, y3], [x4, y4]]
    """
    corners = tag_corners.reshape(4, 2)
    center = np.mean(corners, axis=0)
    return center

def get_relative_pos(ball_pixel_coords, tag_corners):
    # Define the real-world coordinates of the tag corners (in mm)
    # Assuming tag is centered at (0,0), corners are half-size away
    s = config.TAG_SIZE_MM / 2
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    world_pts = np.array([
        [-s,  s], [ s,  s], 
        [ s, -s], [-s, -s]
    ], dtype=np.float32)

    # Get the detected pixel corners from AprilTag
    # Ensure tag_corners is shaped (4, 2)
    pixel_pts = tag_corners.reshape(4, 2).astype(np.float32)

    # Calculate the Homography Matrix
    # This matrix 'M' translates any pixel (u, v) on that plane to real (x, y)
    M, _ = cv2.findHomography(pixel_pts, world_pts)

    # Transform the ball pixel coordinate
    ball_pt = np.array([[ball_pixel_coords]], dtype=np.float32)
    real_world_pt = cv2.perspectiveTransform(ball_pt, M)

    # Result is [[x, y]]
    rel_x = real_world_pt[0][0][0]
    rel_y = real_world_pt[0][0][1]
    
    return (round(rel_x, 2), round(rel_y, 2))


def get_relative_pos_between_tags(reference_tag_corners, target_tag_corners):
    """
    Computes the relative position of one tag with respect to another tag.

    Returns:
        (rel_x_mm, rel_y_mm)
    """
    # print(f'Reference corners: {reference_tag_corners}, Target corners: {target_tag_corners}')
    # Real-world coordinates of the reference tag corners
    s = config.TAG_SIZE_MM / 2

    # Corner order:
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    world_pts = np.array([
        [-s,  s], [ s,  s],
        [ s, -s], [-s, -s]
    ], dtype=np.float32)

    # Reference tag pixel corners
    ref_pixel_pts = reference_tag_corners.reshape(4, 2).astype(np.float32)

    # Homography from image -> reference tag coordinate system
    M, _ = cv2.findHomography(ref_pixel_pts, world_pts)

    # Get center of target tag in pixel coordinates
    target_center = get_tag_center(target_tag_corners)

    # Transform target center into reference tag world coordinates
    target_pt = np.array([[target_center]], dtype=np.float32)
    transformed_pt = cv2.perspectiveTransform(target_pt, M)

    rel_x = transformed_pt[0][0][0]
    rel_y = transformed_pt[0][0][1]

    return (round(rel_x, 2), round(rel_y, 2))