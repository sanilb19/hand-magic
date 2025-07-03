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
    {"attached": True, "pos": np.array([0, 0], dtype=np.float32), "vel": np.array([0, 0], dtype=np.float32), "type": "tip"} for _ in range(5)
] + [
    {"attached": True, "pos": np.array([0, 0], dtype=np.float32), "vel": np.array([0, 0], dtype=np.float32), "type": "knuckle"} for _ in range(5)
]

# Palm tracking for velocity
prev_palm_center = None
palm_velocity = np.array([0, 0], dtype=np.float32)

# Add before main loop:
prev_throw_openness = None

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    success, img = cap.read()
    # 1. Mirror the camera image
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        
        h, w, _ = img.shape
        landmarks = []
        
        for lm in hand.landmark:
            landmarks.append(np.array([int(lm.x * w), int(lm.y * h)], dtype=np.float32))
        
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
        
        for idx, i in enumerate(all_indices):
            ball = balls[idx]
            if idx < 5:
                # Fingertip balls
                finger_pos = landmarks[i]
                offset = tip_offsets[idx]
                ref_pos = finger_pos + palm_normal * offset
            else:
                # Knuckle balls
                kidx = idx - 5
                finger_pos = landmarks[i]
                offset = knuckle_offsets[kidx]
                ref_pos = finger_pos + palm_normal * offset
            # All balls now behave the same:
            if ball["attached"]:
                lerp_speed = 0.7
                ball["pos"] += (finger_pos - ball["pos"]) * lerp_speed
                # Only detach if throw_openness crosses threshold from below (rising edge)
                if (
                    throw_openness > throw_open_thresh and
                    prev_throw_openness <= throw_open_thresh and
                    speed > throw_speed_thresh
                ):
                    ball["attached"] = False
                    direction = palm_velocity / (np.linalg.norm(palm_velocity) + 1e-5)
                    ball["vel"] = direction * min(speed * 0.12, 60)
            else:
                dir_to_ref = ref_pos - ball["pos"]
                dist = np.linalg.norm(dir_to_ref)
                moving_toward_ref = np.dot(ball["vel"], dir_to_ref) > 0
                if retrieve_openness < retrieve_close_thresh and moving_toward_ref:
                    ball["attached"] = True
                    ball["pos"] = finger_pos.copy()
                    ball["vel"] = np.zeros(2, dtype=np.float32)
                else:
                    if dist < 100 and moving_toward_ref:
                        return_lerp_speed = 0.25
                        ball["pos"] += (ref_pos - ball["pos"]) * return_lerp_speed
                        ball["vel"] = np.zeros(2, dtype=np.float32)
                    else:
                        if dist > 1e-2:
                            dir_to_ref /= dist
                        damping = 0.90
                        spring_force = dir_to_ref * (dist * spring_k)
                        ball["vel"] += spring_force * dt
                        ball["vel"] *= damping
                        ball["pos"] += ball["vel"] * dt
        prev_throw_openness = throw_openness
        
        # Draw landmarks
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        
        # Draw balls
        for idx, ball in enumerate(balls):
            color = (0, 255, 255) if not ball["attached"] else (0, 255, 100)
            if idx >= 5:
                color = (255, 100, 0) if not ball["attached"] else (100, 255, 255)
            cv2.circle(img, tuple(ball["pos"].astype(int)), 14, color, -1)
    
    cv2.imshow("Yoyo Effect", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
