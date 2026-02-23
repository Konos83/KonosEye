"""
KonosEye - Critical Zone Editor (mask.py)
A graphical UI tool to easily draw and save detection polygons 
directly onto the RTSP camera stream.
"""

import cv2
import json
import numpy as np
import os

# --- CONSTANTS ---
SETTINGS_FILE = "settings.json"
WINDOW_NAME = "KonosEye - Critical Zone Editor"
DISPLAY_WIDTH = 1280  
MAX_ZONES = 4         

ZOOM_LEVEL = 3.0      
PAN_SPEED = 40        

# Colors for different zones (BGR format)
ZONE_COLORS = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (255, 0, 255)] 

# --- GLOBALS ---
polygons = [[] for _ in range(MAX_ZONES)]
current_zone_idx = 0
rtsp_url = ""

# Viewport & Zoom tracking
is_zoomed = False
view_x, view_y = 0, 0
view_w, view_h = 0, 0
img_w, img_h = 0, 0

def load_config():
    """Loads the RTSP URL and existing zones from settings.json."""
    global rtsp_url, polygons
    
    if not os.path.exists(SETTINGS_FILE):
        print(f"[ERROR] {SETTINGS_FILE} not found! Please ensure it exists in the directory.")
        return False

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        profile_name = data.get('active_profile', 'default')
        if profile_name not in data.get('profiles', {}):
            print(f"[ERROR] Profile '{profile_name}' not found in settings!")
            return False
            
        p = data['profiles'][profile_name]
        rtsp_url = p.get('RTSP_URL', "")
        
        # Load existing zones if they exist
        raw_zones = p.get('CRITICAL_ZONES', [])
        for i, zone in enumerate(raw_zones):
            if i >= MAX_ZONES: 
                break
            
            # Handle both dictionary formats and raw lists
            points = zone.get('points', []) if isinstance(zone, dict) else zone
            
            clean_pts = []
            for pt in points:
                if len(pt) >= 2:
                    clean_pts.append((int(pt[0]), int(pt[1])))
            polygons[i] = clean_pts
            
        print(f"[SUCCESS] Configuration loaded. Target URL: {rtsp_url}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        return False

def save_to_json():
    """Saves the drawn polygons back into settings.json without altering other settings."""
    if not os.path.exists(SETTINGS_FILE):
        print("[ERROR] settings.json does not exist!")
        return

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        profile_name = data.get('active_profile', 'default')
        
        zones_to_save = []
        for i, poly in enumerate(polygons):
            if len(poly) < 3: 
                continue # A valid polygon needs at least 3 points
            
            zone_obj = {
                "id": i + 1,
                "name": f"Zone_{i+1}",
                "points": [[int(p[0]), int(p[1])] for p in poly]
            }
            zones_to_save.append(zone_obj)
        
        # Update the specific profile's zones
        data['profiles'][profile_name]['CRITICAL_ZONES'] = zones_to_save
        
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"\n[SUCCESS] Successfully saved {len(zones_to_save)} zone(s) to JSON.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save to JSON: {e}")

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to draw points on the stream, compensating for zoom/pan."""
    global polygons
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Translate display coordinates back to original video coordinates
        scale = view_w / DISPLAY_WIDTH
        real_x = int(view_x + (x * scale))
        real_y = int(view_y + (y * scale))
        
        # Clamp coordinates to image boundaries
        real_x = max(0, min(real_x, img_w - 1))
        real_y = max(0, min(real_y, img_h - 1))
        
        polygons[current_zone_idx].append((real_x, real_y))

def update_viewport():
    """Ensures panning doesn't go out of frame bounds."""
    global view_x, view_y
    view_x = max(0, min(view_x, img_w - view_w))
    view_y = max(0, min(view_y, img_h - view_h))

def main():
    global img_w, img_h, view_x, view_y, view_w, view_h, is_zoomed, current_zone_idx

    if not load_config():
        input("Press Enter to exit...")
        return

    print(f"[*] Attempting to connect to stream: {rtsp_url}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERROR] Cannot connect to camera stream. Please check your RTSP URL.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Connected, but failed to receive initial frame.")
        return

    img_h, img_w = frame.shape[:2]
    
    # Initialize viewport to full image
    view_x, view_y = 0, 0
    view_w, view_h = img_w, img_h
    
    aspect = img_h / img_w
    disp_h = int(DISPLAY_WIDTH * aspect)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n" + "="*30)
    print("      EDITOR CONTROLS      ")
    print("="*30)
    print("🖱️  Left Click : Add point")
    print("💾  'E'        : Save zones to settings.json")
    print("🔄  'N'        : Switch to Next Zone")
    print("🗑️  'R'        : Reset (Clear) current zone")
    print("🔍  'Z'        : Toggle Zoom In/Out (Centers view)")
    print("🕹️  W/A/S/D    : Pan camera while zoomed in")
    print("❌  'Q'        : Quit application")
    print("="*30 + "\n")

    while True:
        ret, current_frame = cap.read()
        if ret: 
            frame = current_frame
        
        update_viewport()

        # Crop Region of Interest (ROI) if zoomed
        if is_zoomed:
            roi = frame[int(view_y):int(view_y+view_h), int(view_x):int(view_x+view_w)]
        else:
            view_x, view_y, view_w, view_h = 0, 0, img_w, img_h
            roi = frame
            
        if roi.size == 0: 
            continue
            
        display_img = cv2.resize(roi, (DISPLAY_WIDTH, disp_h))

        # Calculate display scaling
        scale_x = DISPLAY_WIDTH / view_w
        scale_y = disp_h / view_h

        # Render all drawn polygons
        for i, poly in enumerate(polygons):
            if not poly: 
                continue
            
            pts_screen = []
            for px, py in poly:
                sx = int((px - view_x) * scale_x)
                sy = int((py - view_y) * scale_y)
                pts_screen.append([sx, sy])
            
            pts_np = np.array(pts_screen, np.int32)
            color = ZONE_COLORS[i]
            thickness = 3 if i == current_zone_idx else 1
            
            # Draw outline
            cv2.polylines(display_img, [pts_np], True, color, thickness)
            
            # Add semi-transparent fill
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts_np], color)
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
            
            # Emphasize vertices for the active zone
            if i == current_zone_idx:
                for pt in pts_screen:
                    cv2.circle(display_img, tuple(pt), 4, (255, 255, 255), -1)

        # Draw UI overlay
        cv2.rectangle(display_img, (0, 0), (DISPLAY_WIDTH, 40), (30, 30, 30), -1)
        txt = f"ZONE {current_zone_idx+1} | MODE: {'ZOOM' if is_zoomed else 'FULL'} | Press [E] to SAVE"
        cv2.putText(display_img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): 
            break
        elif key == ord('e'): 
            save_to_json()
            # Flash screen green to confirm save visually
            cv2.rectangle(display_img, (0,0), (DISPLAY_WIDTH, disp_h), (0, 255, 0), 10)
            cv2.imshow(WINDOW_NAME, display_img)
            cv2.waitKey(200)
        elif key == ord('n'): 
            current_zone_idx = (current_zone_idx + 1) % MAX_ZONES
        elif key == ord('r'): 
            polygons[current_zone_idx] = []
        elif key == ord('z'):
            is_zoomed = not is_zoomed
            if is_zoomed:
                view_w, view_h = img_w / ZOOM_LEVEL, img_h / ZOOM_LEVEL
                view_x, view_y = (img_w - view_w)/2, (img_h - view_h)/2
        
        # Panning controls
        if is_zoomed:
            if key == ord('w'): view_y -= PAN_SPEED
            if key == ord('s'): view_y += PAN_SPEED
            if key == ord('a'): view_x -= PAN_SPEED
            if key == ord('d'): view_x += PAN_SPEED

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()