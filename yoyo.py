import cv2
import mediapipe as mp
import numpy as np
import time

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Ball state: 5 fingertip balls + 5 knuckle balls
balls = [
    {"attached": True, "pos": np.array([0, 0], dtype=np.float32), "vel": np.array([0, 0], dtype=np.float32), "trail": []} for _ in range(10)
]

# Global state for all balls
all_attached = True
returning = False
big_ball_trail = []

# Palm tracking for velocity
prev_palm_center = None
palm_velocity = np.array([0, 0], dtype=np.float32)

# Add before main loop:
prev_throw_openness = None

# Feature flag for fancy effect
FANCY_MODE = True

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    success, img = cap.read()
    # 1. Mirror the camera image
    img = cv2.flip(img, 1)
    cam_img = img.copy()  # Always keep the mirrored camera frame for MediaPipe
    img_rgb = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time
    
    if FANCY_MODE:
        # Create a much darker vertical gradient background for drawing
        h, w = cam_img.shape[:2]
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / h
            color = int(5 + 15 * (1 - t))  # very dark blue to black
            gradient[y, :, :] = (color, color, color//3)
        img = gradient.copy()
        # --- Canny edge detection for all outlines ---
        gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        edges_blur = cv2.GaussianBlur(edges, (7,7), 0)
        edges_color = cv2.cvtColor(edges_blur, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 1.0, edges_color, 0.7, 0)
    else:
        img = cam_img.copy()
    
    if results.multi_hand_landmarks:
        # print('Hand landmarks detected!')
        hand = results.multi_hand_landmarks[0]
        
        h, w, _ = img.shape
        landmarks = []
        
        for lm in hand.landmark:
            landmarks.append(np.array([int(lm.x * w), int(lm.y * h)], dtype=np.float32))
        # Debug: print palm center landmark coordinates
        # print('Palm center landmark (pixel):', landmarks[0])
        
        palm_center = landmarks[0]
        fingertip_indices = [4, 8, 12, 16, 20]
        knuckle_indices = [2, 6, 10, 14, 18]  # MCP joints
        all_indices = fingertip_indices + knuckle_indices
        
        # Compute palm normal: from wrist (0) to average of MCP knuckles (5, 9, 13, 17, 1)
        mcp_indices = [5, 9, 13, 17, 1]
        mcp_mean = np.mean([landmarks[j] for j in mcp_indices], axis=0)
        palm_normal = mcp_mean - landmarks[0]
        palm_normal /= (np.linalg.norm(palm_normal) + 1e-5)
        
        # For each fingertip, compute offset magnitude (distance to knuckle)
        tip_offsets = [np.linalg.norm(landmarks[tip_idx] - landmarks[knuckle_idx]) for tip_idx, knuckle_idx in zip(fingertip_indices, knuckle_indices)]
        # For knuckles, offset is 0.7 * distance to palm center
        knuckle_offsets = [np.linalg.norm(landmarks[knuckle_idx] - palm_center) * 0.7 for knuckle_idx in knuckle_indices]
        
        # Palm velocity tracking (restored)
        if prev_palm_center is not None:
            palm_velocity = (palm_center - prev_palm_center) / max(dt, 1e-5)
        speed = np.linalg.norm(palm_velocity)
        prev_palm_center = palm_center.copy()
        
        # Throw openness: average distance from fingertips to palm center
        throw_openness = np.mean([np.linalg.norm(landmarks[i] - palm_center) for i in fingertip_indices])
        # Retrieve openness: average distance between adjacent fingertips
        fingertip_pairs = list(zip(fingertip_indices, fingertip_indices[1:] + [fingertip_indices[0]]))
        retrieve_openness = np.mean([np.linalg.norm(landmarks[i] - landmarks[j]) for i, j in fingertip_pairs])
        
        # Thresholds
        throw_open_thresh = 200  # Hand must be very open to throw
        throw_speed_thresh = 400  # Palm must move fast to throw
        retrieve_close_thresh = 60  # Fingers must be close to retrieve
        min_spring_k = 8.0
        max_spring_k = 32.0
        # Map retrieve_openness to spring constant (springiness)
        spring_k = np.interp(retrieve_openness, [40, 200], [max_spring_k, min_spring_k])
        spring_k = float(np.clip(spring_k, min_spring_k, max_spring_k))
        
        # (after throw_openness and retrieve_openness are computed)
        if prev_throw_openness is None:
            prev_throw_openness = throw_openness
        
        # Track which balls are ready to catch
        ready_to_catch = [False] * len(balls)
        
        for idx, i in enumerate(all_indices):
            ball = balls[idx]
            if all_attached:
                lerp_speed = 0.7
                ball["pos"] += (landmarks[i] - ball["pos"]) * lerp_speed
                # Only detach if throw_openness crosses threshold from below (rising edge)
                if (
                    throw_openness > throw_open_thresh and
                    prev_throw_openness <= throw_open_thresh and
                    speed > throw_speed_thresh
                ):
                    all_attached = False
                    returning = False
                    for b in balls:
                        b["attached"] = False
                        direction = palm_velocity / (np.linalg.norm(palm_velocity) + 1e-5)
                        b["vel"] = direction * min(speed * 0.12, 60)
            else:
                dir_to_ref = landmarks[i] - ball["pos"]
                dist = np.linalg.norm(dir_to_ref)
                moving_toward_ref = np.dot(ball["vel"], dir_to_ref) > 0
                # If retrieve gesture is detected (fingers close together), set returning state
                if not returning and retrieve_openness < retrieve_close_thresh:
                    returning = True
                # If returning, lerp balls quickly to hand
                if returning:
                    return_lerp_speed = 0.65
                    ball["pos"] += (landmarks[i] - ball["pos"]) * return_lerp_speed
                    ball["vel"] = np.zeros(2, dtype=np.float32)
                else:
                    if dist < 100 and moving_toward_ref:
                        return_lerp_speed = 0.25
                        ball["pos"] += (landmarks[i] - ball["pos"]) * return_lerp_speed
                        ball["vel"] = np.zeros(2, dtype=np.float32)
                    else:
                        if dist > 1e-2:
                            dir_to_ref /= dist
                        damping = 0.90
                        spring_force = dir_to_ref * (dist * spring_k)
                        ball["vel"] += spring_force * dt
                        ball["vel"] *= damping
                        ball["pos"] += ball["vel"] * dt
            # Update trail
            ball["trail"].append(tuple(ball["pos"].astype(int)))
            if len(ball["trail"]) > 5:
                ball["trail"] = ball["trail"][-5:]
        # If all balls are close to hand after returning, attach all at once and switch to big red ball
        if not all_attached and returning:
            close_count = 0
            for idx, i in enumerate(all_indices):
                if np.linalg.norm(balls[idx]["pos"] - landmarks[i]) < 18:
                    close_count += 1
            if close_count == len(balls):
                all_attached = True
                returning = False
                # Save the last trail positions for the big red ball
                big_ball_trail = [balls[idx]["pos"].astype(int) for idx in range(len(balls))]
                for idx, i in enumerate(all_indices):
                    ball = balls[idx]
                    ball["attached"] = True
                    ball["pos"] = landmarks[i].copy()
                    ball["vel"] = np.zeros(2, dtype=np.float32)
        prev_throw_openness = throw_openness
        
        # Draw hand outline as glowing white lines if FANCY_MODE
        if not FANCY_MODE:
            # Draw landmarks and connections as usual
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        
        # Draw balls
        for idx, ball in enumerate(balls):
            if FANCY_MODE:
                if all_attached:
                    # Update big red ball trail every frame
                    big_pos = landmarks[9].copy()
                    # Offset toward palm center for more 'inside hand' effect
                    palm_center = landmarks[0]
                    offset_vec = palm_center - big_pos
                    big_pos += offset_vec * 0.25  # Move 25% of the way toward palm center
                    big_pos = tuple(big_pos.astype(int))
                    big_ball_trail.append(big_pos)
                    if len(big_ball_trail) > 5:
                        big_ball_trail = big_ball_trail[-5:]
                    # Render one big red ball at offset position
                    orb_layer = np.zeros_like(img)
                    color = (0, 0, 255)  # neon red (BGR)
                    # Big comet trail using last positions
                    for t, trail_pos in enumerate(reversed(big_ball_trail)):
                        radius = max(8, 24 - t * 3)
                        trail_color = tuple(int(c * 0.7) for c in color)
                        cv2.circle(orb_layer, trail_pos, radius, trail_color, -1)
                    # Extra fuzzy glow layers (smaller)
                    cv2.circle(orb_layer, big_pos, 60, color, -1)
                    orb_layer = cv2.GaussianBlur(orb_layer, (81,81), 0)
                    cv2.circle(orb_layer, big_pos, 36, color, -1)
                    orb_layer = cv2.GaussianBlur(orb_layer, (51,51), 0)
                    # Main big orb (smaller)
                    cv2.circle(orb_layer, big_pos, 20, color, -1)
                    img = cv2.addWeighted(img, 1.0, orb_layer, 0.85, 0)
                    break  # Only draw one big ball
                else:
                    orb_layer = np.zeros_like(img)
                    color = (255, 255, 255)  # much brighter neon blue (BGR)
                    pos = tuple(ball["pos"].astype(int))
                    # Comet trail (short, pointy)
                    trail_len = len(ball["trail"])
                    for t, trail_pos in enumerate(reversed(ball["trail"])):
                        radius = max(6, 16 - t * 4)
                        alpha = 0.18 * (1 - t / max(1, trail_len-1))
                        trail_color = tuple(int(c * 0.7) for c in color)
                        cv2.circle(orb_layer, trail_pos, radius, trail_color, -1)
                    orb_layer = cv2.GaussianBlur(orb_layer, (41,41), 0)
                    # Main orb
                    cv2.circle(orb_layer, pos, 16, color, -1)
                    img = cv2.addWeighted(img, 1.0, orb_layer, 0.8, 0)
            else:
                color = (0, 255, 255) if not ball["attached"] else (0, 255, 100)
                cv2.circle(img, tuple(ball["pos"].astype(int)), 14, color, -1)
    
    cv2.imshow("Yoyo Effect", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
