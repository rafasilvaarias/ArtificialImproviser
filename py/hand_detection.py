import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FINGER_TIPS = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

FINGER_NUMBERS = {
    'index': 1,
    'middle': 2,
    'ring': 3,
    'pinky': 4
}


class HandDetector:
    def __init__(self, model_path='hand_landmarker.task'):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )
        self.detector = HandLandmarker.create_from_options(options)
    
    def detect(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.detector.detect(mp_image)
        return results


def get_touching_fingers(hand_landmarks, hand_z, distance_threshold=0.1):
    thumb_tip = hand_landmarks[4]
    thumb_pos = (thumb_tip.x, thumb_tip.y)
    
    adjusted_threshold = distance_threshold - (hand_z * 0.07)
    
    touching = []
    
    for finger_name, finger_idx in FINGER_TIPS.items():
        if finger_name == 'thumb':
            continue
        
        finger_tip = hand_landmarks[finger_idx]
        finger_pos = (finger_tip.x, finger_tip.y)
        
        distance = np.sqrt((thumb_pos[0] - finger_pos[0])**2 + (thumb_pos[1] - finger_pos[1])**2)
        
        if distance < adjusted_threshold:
            touching.append(FINGER_NUMBERS[finger_name])
    
    return sorted(touching)


def get_finger_colors(hand_landmarks, hand_z, distance_threshold=0.05):
    colors = []
    thumb_tip = hand_landmarks[4]
    thumb_pos = (thumb_tip.x, thumb_tip.y)
    
    adjusted_threshold = distance_threshold - (hand_z * 0.02)
    
    for i, landmark in enumerate(hand_landmarks):
        if i == 4:
            colors.append((255, 0, 255))
        else:
            finger_pos = (landmark.x, landmark.y)
            distance = np.sqrt((thumb_pos[0] - finger_pos[0])**2 + (thumb_pos[1] - finger_pos[1])**2)
            
            if distance < adjusted_threshold:
                colors.append((0, 255, 0))
            else:
                colors.append((0, 0, 255))
    
    return colors


def get_hand_position(hand_landmarks):
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    
    avg_x = np.mean(x_coords)
    avg_y = np.mean(y_coords)
    
    lowest_point_idx = np.argmax(y_coords)
    lowest_y = hand_landmarks[lowest_point_idx].y
    
    pinky_base = hand_landmarks[17]
    
    distance_normalized = np.sqrt((pinky_base.x - hand_landmarks[lowest_point_idx].x)**2 + 
                                   (pinky_base.y - lowest_y)**2)
    
    normalized_z = (distance_normalized - 0.125) / 0.175
    normalized_z = np.clip(normalized_z, 0, 1)
    
    return avg_x, avg_y, normalized_z


def get_hand_angle(hand_landmarks):
    wrist = hand_landmarks[0]
    middle_base = hand_landmarks[9]
    
    dx = middle_base.x - wrist.x
    dy = middle_base.y - wrist.y
    angle = np.degrees(np.arctan2(dy, dx))
    
    mapped_angle = -angle / 150.0
    mapped_angle = np.clip(mapped_angle, 0, 1)
    
    return mapped_angle


def draw_hand_landmarks(img, hand_landmarks, colors):
    h, w, c = img.shape
    
    for i, landmark in enumerate(hand_landmarks):
        x, y = int(landmark.x * w), int(landmark.y * h)
        
        thumb_tip = hand_landmarks[4]
        thumb_pos = (thumb_tip.x, thumb_tip.y)
        finger_pos = (landmark.x, landmark.y)
        is_contact = np.sqrt((thumb_pos[0] - finger_pos[0])**2 + (thumb_pos[1] - finger_pos[1])**2) < 0.05
        
        size = 5 if is_contact else 1
        color = colors[i]
        
        cv2.circle(img, (x, y), size, color, cv2.FILLED)
    
    for i in range(len(hand_landmarks) - 1):
        x1, y1 = int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h)
        x2, y2 = int(hand_landmarks[i + 1].x * w), int(hand_landmarks[i + 1].y * h)
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
