import cv2
import numpy as np
from scipy import interpolate
from skimage import color
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Define facial feature landmarks
upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
cheeks = [425, 205]

# Define default colors for nail polish and eyeshadow
Rg, Gg, Bg = (207, 40, 57)  # Nail polish color
R, G, B = (0, 0, 255)  # Eyeshadow color

# Main function to apply makeup effects
def apply_makeup(src: np.ndarray, is_stream: bool, feature: str, show_landmarks: bool = False):

    landmarks = detect_landmarks(src, is_stream)
    if landmarks is None:
        raise ValueError("No face landmarks detected.")
    height, width, _ = src.shape
    
    if src.dtype != np.uint8:
        src = np.clip(src, 0, 255).astype(np.uint8)

    # Apply specific makeup feature
    if feature == 'lips':
        points = normalize_landmarks(landmarks, height, width, upper_lip + lower_lip)
        mask = lip_mask(src, points, [0, 0, 255])
        output = cv2.addWeighted(src, 1.0, mask, 0.5, 0.5)

    elif feature == 'blush':
        points = normalize_landmarks(landmarks, height, width, cheeks)
        mask = blush_mask(src, points, [102, 0, 51], 30)
        output = cv2.addWeighted(src, 1.0, mask, 0.3, 0.0)

    elif feature == 'eyeshadow':
        output = apply_eyeshadow(src, landmarks, {"EYESHADOW_LEFT": [R, G, B], "EYESHADOW_RIGHT": [R, G, B]})

    elif feature == 'eyeliner':
        output = apply_eyeliner(src, landmarks, {"EYELINER_LEFT": [139, 0, 0], "EYELINER_RIGHT": [139, 0, 0]})


    else: 
        pass

    if show_landmarks and landmarks is not None:
        plot_landmarks(src, points, True)

    return output

# Detect facial landmarks using MediaPipe FaceMesh
def detect_landmarks(src: np.ndarray, is_stream: bool = False):
    with mp_face_mesh.FaceMesh(static_image_mode=not is_stream, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

# Normalize facial landmarks to image dimensions
def normalize_landmarks(landmarks, height, width, feature_points):
    points = []
    for i in feature_points:
        keypoint = landmarks.landmark[i]
        x = int(keypoint.x * width)
        y = int(keypoint.y * height)
        points.append([x, y])
    return np.array(points)

# Create a lip mask
def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [points], color)
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask

# Create a blush mask
def blush_mask(src: np.ndarray, points: np.ndarray, color: list, radius: int):
    mask = np.zeros_like(src)
    for point in points:
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)
    return mask

# Apply eyeshadow to image
def apply_eyeshadow(src: np.ndarray, landmarks: dict, colors_map: dict):
    eyeshadow_left = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
    eyeshadow_right = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]
    
    mask = np.zeros_like(src)
    landmark_dict = get_landmark_dict(landmarks, src.shape)

    # Apply eyeshadow to left and right eyes
    for eye, color_key in zip([eyeshadow_left, eyeshadow_right], ["EYESHADOW_LEFT", "EYESHADOW_RIGHT"]):
        if all(idx in landmark_dict for idx in eye):
            eye_points = np.array([landmark_dict[idx] for idx in eye])
          cv2.fillPoly(mask, [eye_points], colors_map[color_key])
    
    mask = cv2.GaussianBlur(mask, (7, 7), 4)
    output = cv2.addWeighted(src, 1.0, mask, 0.4, 1.0)
    return output

# Apply eyeliner to image
def apply_eyeliner(src: np.ndarray, landmarks: dict, colors_map: dict):
    eyeliner_left = [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243]
    eyeliner_right = [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341, 463]

    mask = np.zeros_like(src)
    landmark_dict = get_landmark_dict(landmarks, src.shape)

    # Apply eyeliner to left and right eyes
    for eye, color_key in zip([eyeliner_left, eyeliner_right], ["EYELINER_LEFT", "EYELINER_RIGHT"]):
        if all(idx in landmark_dict for idx in eye):
            eye_points = np.array([landmark_dict[idx] for idx in eye])
            cv2.polylines(mask, [eye_points], isClosed=False, color=colors_map[color_key], thickness=4)
    
    output = cv2.addWeighted(src, 1.0, mask, 0.7, 1.0)
    return output

# Additional utility functions
def get_landmark_dict(landmarks, shape):
    height, width, _ = shape
    landmark_dict = {}
    for idx, landmark in enumerate(landmarks.landmark):
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, width, height
        )
        if landmark_px:
            landmark_dict[idx] = landmark_px
    return landmark_dict
  
def get_landmark_dict(landmarks, shape):
    """
    Convert landmarks to a dictionary of coordinates.
    """
    height, width, _ = shape
    landmark_dict = {}
    for idx, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        landmark_dict[idx] = (x, y)
    return landmark_dict
