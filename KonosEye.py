"""
SecureGuard - Enterprise-Grade RTSP Camera NVR Pipeline
Features:
- Hardware Accelerated YOLO11 via AMD DirectML / ONNXRuntime
- Non-Maximum Suppression (NMS) for accurate counting
- Dynamic CLAHE for adaptive Night Vision
- Anti-Ghosting algorithm to filter static shadows/trees
- Asynchronous Telegram Alerts
- Zero-buffer RTSP capturing for real-time processing
"""

import cv2
import json
import time
import os
import numpy as np
import requests
import onnxruntime as ort
import threading
import datetime
import logging
import traceback
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- LOGGING SETUP ---
def setup_logger():
    logger = logging.getLogger('SecureGuard')
    logger.setLevel(logging.DEBUG)
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] %(name)s.%(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    try:
        if not os.path.exists('logs'):
            os.makedirs('logs')
        file_handler = logging.FileHandler('logs/secureguard.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except:
        pass
    
    return logger

logger = setup_logger()

# --- CONFIG CLASS ---
class Config:
    def __init__(self):
        self.rtsp_url = ""
        self.telegram_token = ""
        self.chat_id_1 = ""
        self.chat_id_2 = ""
        
        self.yolo_model = "yolo11n.onnx"
        self.yolo_conf = 0.30
        self.target_fps = 1.0
        self.alert_cooldown = 3
        self.min_pixels_area = 0
        self.max_pixels_area = 200000 
        
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75
        self.clahe_clip_limit = 3.0
        
        self.crop_size = 640
        self.critical_zones = []
        
        self.gpu_enabled = True
        self.batch_size = 1  
        self.gpu_memory_fraction = 0.8  
        
        self.alert_confirmations = 1  
        self.min_detection_frames = 2  
        
        self.motion_enabled = False
        self.motion_history_alpha = 0.02  
        self.motion_threshold = 25  
        self.motion_min_area = 500  
        
        self.match_iou_threshold = 0.1
        self.track_max_age = 5.0  
        self.track_min_hits = 1  
        
        self.person_only = True
        self.person_class_id = 0  
        
        self.track_alert_cooldown = 30  
        self.zone_sensitivities = {}  
        self.min_bbox_rel_height = 0.0  
        self.low_light_additional_hits = 1  

cfg = Config()
keep_running = True
frame_lock = threading.Lock()
latest_frame = None
has_new_frame = False
frame_counter = 0
process_every_n_frames = 1

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, dw, dh

def load_config(silent=False):
    try:
        path = os.path.join(os.getcwd(), "settings.json")
        if not os.path.exists(path):
            logger.error(f"Settings file not found at: {path}")
            return False

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        profile_name = data.get("active_profile", "default")
        if "profiles" not in data or profile_name not in data["profiles"]:
            logger.error(f"Profile '{profile_name}' not found in settings.json")
            return False
            
        p = data["profiles"][profile_name]
        
        if not p.get("RTSP_URL"):
            logger.error("RTSP_URL is empty in settings.json")
            return False
        
        cfg.rtsp_url = p.get("RTSP_URL", "")
        cfg.telegram_token = p.get("TELEGRAM_TOKEN", "")
        cfg.chat_id_1 = p.get("CHAT_ID_1", "")
        cfg.chat_id_2 = p.get("CHAT_ID_2", "")
        
        yolo = p.get("YOLO_CONFIG", {})
        cfg.yolo_model = yolo.get("MODEL_FILE", "yolo11n.onnx").replace(".xml", ".onnx")
        cfg.yolo_conf = yolo.get("CONFIDENCE", 0.30)
        cfg.target_fps = yolo.get("SCAN_FPS", 1.0)
        cfg.alert_cooldown = yolo.get("COOLDOWN_SECONDS", 3)
        cfg.min_pixels_area = yolo.get("MIN_PIXELS_AREA", 0)
        cfg.max_pixels_area = yolo.get("MAX_PIXELS_AREA", 400000)
        
        img_env = p.get("IMAGE_ENHANCEMENT", {})
        cfg.bilateral_sigma_color = img_env.get("BILATERAL_SIGMA_COLOR", 75)
        cfg.bilateral_sigma_space = img_env.get("BILATERAL_SIGMA_SPACE", 75)
        cfg.clahe_clip_limit = img_env.get("CLAHE_CLIP_LIMIT", 3.0)
        
        crop = p.get("CROP_SETTINGS", {})
        cfg.crop_size = crop.get("SIZE", 640)

        new_zones = []
        if "CRITICAL_ZONES" in p:
            for i, zone in enumerate(p["CRITICAL_ZONES"]):
                if i >= 4: break 
                try:
                    pts = np.array(zone.get("points", []), dtype=np.int32).reshape((-1, 1, 2))
                    new_zones.append(pts)
                except Exception as e:
                    logger.warning(f"Failed to parse zone {i}: {e}")
        
        cfg.critical_zones = new_zones
        if not silent: 
            logger.info(f"CONFIG LOADED | URL: {cfg.rtsp_url[:40]}... | FPS: {cfg.target_fps} | Min Area: {cfg.min_pixels_area}px | Zones: {len(new_zones)}")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in settings.json: {e}")
        return False
    except Exception as e:
        logger.error(f"Config Error: {e}\n{traceback.format_exc()}")
        return False

def send_alert(image_path, caption, chat_id, max_retries=3):
    if not chat_id: return False
    if not os.path.exists(image_path): return False
    
    file_size = os.path.getsize(image_path)
    if file_size == 0 or file_size > 50 * 1024 * 1024: return False
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.telegram.org/bot{cfg.telegram_token}/sendPhoto"
            if not cfg.telegram_token: return False
            
            with open(image_path, 'rb') as img_file:
                files = {'photo': img_file}
                data = {'chat_id': chat_id, 'caption': caption}
                response = requests.post(url, files=files, data=data, timeout=15)
                
                if response.status_code == 200:
                    logger.info(f"Alert sent successfully to {chat_id}")
                    return True
                else:
                    if attempt < max_retries - 1: time.sleep((attempt + 1) * 2)
        except:
            if attempt < max_retries - 1: time.sleep(2 * (attempt + 1))
    return False

# --- IMAGE ENHANCEMENT (Dynamic CLAHE for Night Vision) ---
def enhance_image(img):
    try:
        denoised = cv2.bilateralFilter(img, 5, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)
    except:
        denoised = img

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness > 100: gamma = 1.0
    elif avg_brightness > 80: gamma = 1.05
    elif avg_brightness > 60: gamma = 1.15
    elif avg_brightness > 40: gamma = 1.35
    elif avg_brightness > 25: gamma = 1.70
    else: gamma = 2.0

    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(denoised, table)
    else:
        enhanced = denoised

    # Apply CLAHE only in dark conditions (< 60 brightness)
    if avg_brightness < 60:
        try:
            # LAB color space to preserve color channels
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            pass

    return enhanced, gamma

def get_clean_crop(full_img, center_target, size):
    h, w = full_img.shape[:2]
    cx, cy = center_target
    
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    
    if x1 == 0: x2 = min(w, size)
    if y1 == 0: y2 = min(h, size)
    if x2 == w: x1 = max(0, w - size)
    if y2 == h: y1 = max(0, h - size)
    
    crop = full_img[y1:y2, x1:x2].copy()
    target_x = cx - x1
    target_y = cy - y1
    
    cv2.circle(crop, (target_x, target_y), 12, (255, 255, 255), 2)
    cv2.circle(crop, (target_x, target_y), 10, (0, 0, 255), -1)
    return crop

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0: return 0.0
    return interArea / denom

class YoloEngine:
    def __init__(self):
        try:
            model_path = os.path.join(os.getcwd(), "ai_model", cfg.yolo_model)
            if not os.path.exists(model_path):
                logger.critical(f"Model file not found: {model_path}")
                global keep_running
                keep_running = False
                return
            
            providers = [
                ('DmlExecutionProvider', {'device_id': 0}),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
            self.inference_times = []
            self.max_inference_time = 0
            
        except Exception as e:
            logger.critical(f"Failed to initialize YOLO engine: {e}")
            keep_running = False

    def detect_crop(self, crop_img, offset_x, offset_y):
        if crop_img is None or crop_img.size == 0: return []
        
        inference_start = time.time()
        
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)  
        except:
            pass
        
        img_letterbox, ratio, pad_w, pad_h = letterbox(crop_img, (640, 640))
        blob = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        try:
            outputs = self.session.run(None, {self.input_name: blob})
        except Exception as e:
            return []
        
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100: self.inference_times.pop(0)
        if inference_time > self.max_inference_time: self.max_inference_time = inference_time
        
        output = outputs[0][0].transpose()
        rows = output.shape[0]
        
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(rows):
            classes_scores = output[i][4:] 
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]
            
            if score > cfg.yolo_conf and class_id == cfg.person_class_id:
                cx, cy, w, h = output[i][:4]
                
                cx_orig = (cx - pad_w) / ratio
                cy_orig = (cy - pad_h) / ratio
                w_orig = w / ratio
                h_orig = h / ratio
                
                left = int(cx_orig - w_orig/2)
                top = int(cy_orig - h_orig/2)
                width = int(w_orig)
                height = int(h_orig)
                
                boxes.append([left, top, width, height])
                scores.append(float(score))
                class_ids.append(class_id)
        
        # --- NON-MAXIMUM SUPPRESSION ---
        indices = cv2.dnn.NMSBoxes(boxes, scores, cfg.yolo_conf, 0.45)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                left, top, width, height = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                
                global_l = left + offset_x
                global_t = top + offset_y
                area = width * height
                
                if area < cfg.min_pixels_area: continue
                if cfg.max_pixels_area > 0 and area > cfg.max_pixels_area: continue
                if height > 0:
                    aspect_ratio = float(width) / height
                    if aspect_ratio < 0.10 or aspect_ratio > 3.5: continue
                
                detections.append({
                    "box": [global_l, global_t, width, height],
                    "conf": score, "class_id": class_id, "area": area, "aspect_ratio": aspect_ratio,
                    "center_feet": (int(global_l + width/2), int(global_t + height)),
                    "center_body": (int(global_l + width/2), int(global_t + height/2))
                })
        return detections
    
    def get_performance_stats(self):
        if not self.inference_times: return {}
        avg_time = np.mean(self.inference_times)
        return {
            "avg_inference_ms": avg_time * 1000,
            "min_inference_ms": np.min(self.inference_times) * 1000,
            "max_inference_ms": np.max(self.inference_times) * 1000,
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0
        }

def camera_thread():
    global latest_frame, has_new_frame, keep_running
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    connection_attempts = 0
    max_attempts = 5
    backoff_time = 5
    
    while keep_running:
        try:
            cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                connection_attempts += 1
                if connection_attempts >= max_attempts:
                    connection_attempts = 0
                    backoff_time = min(backoff_time * 2, 60)
                else:
                    backoff_time = 5
                time.sleep(backoff_time)
                continue
            
            connection_attempts = 0
            backoff_time = 5
            
            while keep_running:
                ret, frame = cap.read()
                if not ret: break
                with frame_lock:
                    latest_frame = frame.copy()
                    has_new_frame = True
                time.sleep(0.005)
            cap.release()
        except:
            time.sleep(5)

def main():
    global keep_running, has_new_frame, latest_frame, frame_counter
    
    if not load_config(): return
    yolo = YoloEngine()
    if not keep_running: return

    t = threading.Thread(target=camera_thread, name="CameraThread")
    t.daemon = True
    t.start()
    
    last_alert_time = 0
    last_config_check_time = time.time()
    
    while latest_frame is None and keep_running: time.sleep(0.1)
    
    loop_iteration = 0
    prev_detections = []
    last_stats_time = time.time()
    detection_tracks = {}  
    next_track_id = 1
    bg_model = None
    is_fresh_detection = False  
    
    while keep_running:
        loop_start = time.time()
        loop_iteration += 1
        frame_counter += 1
        try:
            if (time.time() - last_config_check_time) > 5.0:
                load_config(silent=True)
                last_config_check_time = time.time()
            
            if (time.time() - last_stats_time) > 30.0:
                last_stats_time = time.time()

            detections = []
            used_gamma = 1.0
            full_res_frame = None
            is_fresh_detection = False
            
            with frame_lock:
                if has_new_frame:
                    full_res_frame = latest_frame.copy()
                    has_new_frame = False

            if full_res_frame is None:
                time.sleep(0.01)
                continue

            if frame_counter % process_every_n_frames != 0:
                detections = prev_detections.copy() if prev_detections else []
                is_fresh_detection = False
            else:
                is_fresh_detection = True
                if not cfg.critical_zones:
                    enhanced, used_gamma = enhance_image(full_res_frame)
                    detections = yolo.detect_crop(enhanced, 0, 0)
                else:
                    for poly in cfg.critical_zones:
                        x, y, w, h = cv2.boundingRect(poly)
                        pad = 100 
                        img_h, img_w = full_res_frame.shape[:2]
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(img_w, x + w + pad)
                        y2 = min(img_h, y + h + pad)
                        
                        zone_crop = full_res_frame[y1:y2, x1:x2].copy()
                        enhanced_crop, used_gamma = enhance_image(zone_crop)
                        
                        zone_dets = yolo.detect_crop(enhanced_crop, x1, y1)
                        detections.extend(zone_dets)
                
                prev_detections = detections  

            # --- MOTION PRE-FILTER ---
            now = time.time()
            motion_area = 0
            gray = cv2.cvtColor(full_res_frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            if cfg.motion_enabled:
                if bg_model is None:
                    bg_model = gray_blur.astype('float')
                    motion_area = cfg.motion_min_area + 1
                else:
                    cv2.accumulateWeighted(gray_blur, bg_model, cfg.motion_history_alpha)
                    bg_uint = cv2.convertScaleAbs(bg_model)
                    diff = cv2.absdiff(gray_blur, bg_uint)
                    _, diff_mask = cv2.threshold(diff, cfg.motion_threshold, 255, cv2.THRESH_BINARY)
                    motion_area = int(cv2.countNonZero(diff_mask))

            if cfg.motion_enabled and not cfg.critical_zones and motion_area < cfg.motion_min_area:
                detections = prev_detections
            else:
                if cfg.motion_enabled and cfg.critical_zones:
                    detections = []
                    bg_uint = cv2.convertScaleAbs(bg_model) if bg_model is not None else None
                    for poly in cfg.critical_zones:
                        x, y, w, h = cv2.boundingRect(poly)
                        x1 = max(0, x - 50); y1 = max(0, y - 50)
                        x2 = min(full_res_frame.shape[1], x + w + 50); y2 = min(full_res_frame.shape[0], y + h + 50)
                        zone_gray = gray_blur[y1:y2, x1:x2]
                        if bg_uint is not None:
                            bg_zone = bg_uint[y1:y2, x1:x2]
                            dz = cv2.absdiff(zone_gray, bg_zone)
                            _, dz_mask = cv2.threshold(dz, cfg.motion_threshold, 255, cv2.THRESH_BINARY)
                            zone_motion = int(cv2.countNonZero(dz_mask))
                        else:
                            zone_motion = cfg.motion_min_area + 1

                        if zone_motion >= cfg.motion_min_area:
                            zone_crop = full_res_frame[y1:y2, x1:x2].copy()
                            enhanced_crop, _ = enhance_image(zone_crop)
                            zone_dets = yolo.detect_crop(enhanced_crop, x1, y1)
                            detections.extend(zone_dets)

            prev_detections = detections

            # --- IOU-BASED TRACKING ---
            updated_track_ids = set()
            det_matched = [False] * len(detections)
            for di, det in enumerate(detections):
                det_box = det.get('box')
                best_iou = 0.0
                best_tid = None
                for tid, tr in detection_tracks.items():
                    if now - tr['last_seen'] > cfg.track_max_age: continue
                    i = iou(det_box, tr['bbox'])
                    if i > best_iou:
                        best_iou = i
                        best_tid = tid

                if best_iou >= cfg.match_iou_threshold and best_tid is not None:
                    tr = detection_tracks[best_tid]
                    tr['bbox'] = det_box
                    tr['last_seen'] = now
                    tr['hits'] = tr.get('hits', 1) + 1
                    tr['avg_conf'] = (tr.get('avg_conf', det['conf']) * (tr['hits'] - 1) + det['conf']) / tr['hits']
                    tr['obj'] = det
                    updated_track_ids.add(best_tid)
                    det_matched[di] = True
                else:
                    tid = next_track_id
                    next_track_id += 1
                    detection_tracks[tid] = {'bbox': det.get('box'), 'last_seen': now, 'first_seen': now, 'first_bbox': det.get('box'),  'hits': 1, 'misses': 0, 'avg_conf': det['conf'], 'obj': det}
                    updated_track_ids.add(tid)
                    det_matched[di] = True

            for tid in list(detection_tracks.keys()):
                if tid not in updated_track_ids:
                    detection_tracks[tid]['misses'] = detection_tracks[tid].get('misses', 0) + 1

            for tid in list(detection_tracks.keys()):
                if now - detection_tracks[tid]['last_seen'] > cfg.track_max_age:
                    del detection_tracks[tid]

            # --- ALERT DECISION ---
            frame_h = full_res_frame.shape[0]
            for tid, tr in detection_tracks.items():
                det_class = tr.get('obj', {}).get('class_id')
                if cfg.person_only and det_class is not None and det_class != cfg.person_class_id:
                    continue

                bx, by, bw, bh = tr['bbox']
                
                # --- ANTI-GHOSTING FILTER ---
                time_alive = now - tr.get('first_seen', now)
                if time_alive > 3.0:  
                    start_x = tr['first_bbox'][0] + tr['first_bbox'][2]/2
                    start_y = tr['first_bbox'][1] + tr['first_bbox'][3]/2
                    curr_x = bx + bw/2
                    curr_y = by + bh/2
                    movement_dist = ((curr_x - start_x)**2 + (curr_y - start_y)**2)**0.5
                    
                    if movement_dist < 15.0:  
                        continue 
                
                fx = int(bx + bw/2)
                fy = int(by + bh/2)
                in_zone = False
                zone_idx = -1
                for i, poly in enumerate(cfg.critical_zones):
                    if cv2.pointPolygonTest(poly, (fx, fy), False) >= 0:
                        in_zone = True
                        zone_idx = i + 1
                        break

                zone_cfg = cfg.zone_sensitivities.get(zone_idx, {}) if cfg.zone_sensitivities else {}
                required_hits = max(cfg.alert_confirmations, cfg.track_min_hits, zone_cfg.get('min_hits', 0))
                
                if used_gamma > 1.5: required_hits += cfg.low_light_additional_hits
                rel_h = bh / float(frame_h) if frame_h > 0 else 0
                if rel_h < cfg.min_bbox_rel_height and cfg.min_bbox_rel_height > 0: continue
                if tr.get('hits', 0) < required_hits: continue
                if tr.get('avg_conf', 0) < cfg.yolo_conf: continue
                
                last_track_alert = tr.get('last_alert', 0)
                if now - last_track_alert < cfg.track_alert_cooldown: continue
                if cfg.critical_zones and not in_zone: continue
                if (now - tr['last_seen']) > 0.5: continue
                if not is_fresh_detection: continue

                if (now - last_alert_time) > cfg.alert_cooldown:
                    stats = yolo.get_performance_stats()
                    gpu_fps = stats.get('avg_fps', 0) if stats else 0
                    
                    body_x = int(tr['bbox'][0] + tr['bbox'][2]/2)
                    body_y = int(tr['bbox'][1] + tr['bbox'][3]/2)
                    
                    clean_crop = get_clean_crop(full_res_frame, (body_x, body_y), cfg.crop_size)
                    
                    if clean_crop is not None and clean_crop.size > 0:
                        clean_crop, _ = enhance_image(clean_crop)

                    ts = int(time.time())
                    captures_dir = os.path.join(os.getcwd(), "captures")
                    try:
                        if not os.path.exists(captures_dir): os.makedirs(captures_dir)
                    except Exception as e:
                        captures_dir = None

                    fpath = None
                    if captures_dir:
                        fpath = os.path.join(captures_dir, f"sniper_alert_{ts}.jpg")
                        try:
                            if clean_crop is not None and clean_crop.size > 0:
                                success = cv2.imwrite(fpath, clean_crop)
                                if not success or not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                                    fpath = None
                        except Exception as e:
                            fpath = None

                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    msg = (f"🚨 ALERT ({now_str})\n"
                           f"📍 ZONE: {zone_idx if zone_idx!=-1 else 'FULL'}\n"
                           f"🤖 CONF: {int(tr['avg_conf']*100)}%\n"
                           f"🎮 GPU FPS: {gpu_fps:.1f}\n")

                    if fpath:
                        threading.Thread(target=send_alert, args=(fpath, msg, cfg.chat_id_1), daemon=True).start()
                        if cfg.chat_id_2: 
                            threading.Thread(target=send_alert, args=(fpath, msg, cfg.chat_id_2), daemon=True).start()
                    last_alert_time = now
                    tr['last_alert'] = now
                    tr['hits'] = 0

            elapsed = time.time() - loop_start
            wait_time = (1.0 / cfg.target_fps) - elapsed
            if wait_time > 0: time.sleep(wait_time)

        except Exception as e:
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        keep_running = False
    except Exception as e:
        pass