import cv2
import mediapipe as mp
import ctypes

# Obtener resolución de pantalla (Windows)
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Inicializar módulos
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5,
                       max_num_hands=2)

cap = cv2.VideoCapture(0)

# Pantalla completa
cv2.namedWindow("Detección", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detección", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(img_rgb)
    hands_results = hands.process(img_rgb)

    # Dibujar landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Escalar frame manteniendo proporción
    h, w, _ = frame.shape
    scale = min(screen_width / w, screen_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    frame_resized = cv2.resize(frame, (new_w, new_h))

    # Crear canvas negro del tamaño de la pantalla
    canvas = cv2.resize(frame_resized, (screen_width, screen_height))

    cv2.imshow("Detección", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
