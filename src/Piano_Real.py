import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ----------------------------------
# CONFIGURACIÓN DEL PIANO (20 TECLAS)
# ----------------------------------
LEFT_KEYS = 10
RIGHT_KEYS = 10
TOTAL_KEYS = LEFT_KEYS + RIGHT_KEYS

# Frecuencias graves (mano izquierda)
LEFT_FREQS = [130, 146, 164, 174, 196, 220, 246, 261, 293, 329]  # C3–E4

# Frecuencias agudas (mano derecha)
RIGHT_FREQS = [349, 392, 440, 493, 523, 587, 659, 698, 783, 880] # F4–A5

KEY_COOLDOWN = 0.20

pygame.mixer.init(44100, -16, 1, 512)

def generate_tone(freq, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(freq * t * 2 * np.pi)
    tone = (tone * 32767).astype(np.int16)
    return pygame.mixer.Sound(tone)

# Crear sonidos
left_sounds = [generate_tone(f) for f in LEFT_FREQS]
right_sounds = [generate_tone(f) for f in RIGHT_FREQS]

last_left = [0] * LEFT_KEYS
last_right = [0] * RIGHT_KEYS

# ----------------------------------
# MEDIAPIPE
# ----------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ----------------------------------
# CÁMARA
# ----------------------------------
cap = cv2.VideoCapture(0)

print("▶ Piano de 2 manos activo ─ Presiona Q para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Dibujar 20 teclas en toda la pantalla
    key_width = w // TOTAL_KEYS
    for i in range(TOTAL_KEYS):
        x1 = i * key_width
        x2 = (i + 1) * key_width
        cv2.rectangle(frame, (x1, h-150), (x2, h), (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            label = handedness.classification[0].label  # "Left" o "Right"

            # Landmark del índice
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

            if y > h - 150:
                key_index = x // key_width

                if label == "Left":
                    # Validar rango 0–9
                    if 0 <= key_index < LEFT_KEYS:
                        now = time.time()
                        if now - last_left[key_index] >= KEY_COOLDOWN:
                            print(f"Mano IZQUIERDA → Frecuencia {LEFT_FREQS[key_index]} Hz")
                            left_sounds[key_index].play()
                            last_left[key_index] = now

                        # Resaltar tecla
                        x1 = key_index * key_width
                        x2 = (key_index + 1) * key_width
                        cv2.rectangle(frame, (x1, h-150), (x2, h), (0, 255, 0), -1)

                elif label == "Right":
                    # Ajustar índice para teclas 10–19
                    if LEFT_KEYS <= key_index < TOTAL_KEYS:
                        right_i = key_index - LEFT_KEYS
                        now = time.time()
                        if now - last_right[right_i] >= KEY_COOLDOWN:
                            print(f"Mano DERECHA → Frecuencia {RIGHT_FREQS[right_i]} Hz")
                            right_sounds[right_i].play()
                            last_right[right_i] = now

                        # Resaltar tecla
                        x1 = key_index * key_width
                        x2 = (key_index + 1) * key_width
                        cv2.rectangle(frame, (x1, h-150), (x2, h), (0, 255, 255), -1)

    cv2.imshow("Piano Virtual de 2 Manos", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
