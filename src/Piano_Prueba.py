# detector_teclado_grafo.py
import cv2
import mediapipe as mp
import time
import math
import ctypes
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ----------------- CONFIGURACIÓN -----------------
KEY_LABELS = ["hola", "este", "es", "nuestro", "trabajo", "python", "mensaje", "buenos", "dias", "final"]
KEY_COLORS = [(22,114,136),(140,218,236),(180,82,72),(212,140,132),(168,154,73),
              (214,207,162),(60,180,100),(155,221,177),(100,60,106),(131,99,148)]
DWELL_SECONDS = 3.0
L_ANGLE_MIN = 70.0
L_ANGLE_MAX = 110.0
L_PERSIST_SECONDS = 0.5  # persistencia necesaria para confirmar L posture
FPS_SMOOTH = 5  # frames window for smoothing angles if needed

# ----------------- UTILIDADES -----------------
def angle_between(v1, v2):
    # v1, v2 are tuples/lists
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return (px >= x) and (px <= x + w) and (py >= y) and (py <= y + h)

# ----------------- PANTALLA -----------------
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

# ----------------- MediaPipe init -----------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

# ----------------- Cámara -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

ret, sample_frame = cap.read()
if not ret:
    raise RuntimeError("No se pudo leer frame de la cámara.")
FRAME_H, FRAME_W = sample_frame.shape[:2]

# ----------------- Crear ventana fullscreen -----------------
WIN_NAME = "Detección - Teclado Gestual"
cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ----------------- Configurar teclado (en coordenadas del canvas) -----------------
# Ocuparemos la parte superior: keyboard_height_fraction del espacio usado por la imagen
KEYBOARD_HEIGHT_FRAC = 0.14  # 14% del alto de canvas
# Posiciones calculadas dinámicamente cuando conocemos el tamaño escalado del frame

# ----------------- Estado del sistema -----------------
dwell_state = {
    'left_index': {'key': None, 'start_ts': None, 'confirmed': False},
    'right_index': {'key': None, 'start_ts': None, 'confirmed': False}
}
recording = False
recording_start_ts = None
current_session_selections = []  # lista de (timestamp_iso, key_label)
all_sessions = []  # lista de sessions
G = nx.DiGraph()
prev_selected_key = None

# buffers para estabilidad de L detection
left_angles = deque(maxlen=FPS_SMOOTH)
right_angles = deque(maxlen=FPS_SMOOTH)
left_L_start = None
right_L_start = None

# ----------------- Funciones de grafo y métricas -----------------
def add_selection_to_graph(key_label, ts):
    global prev_selected_key, G
    if not G.has_node(key_label):
        G.add_node(key_label)
    if prev_selected_key is not None:
        u = prev_selected_key
        v = key_label
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
            G[u][v].setdefault('timestamps', []).append(ts)
        else:
            G.add_edge(u, v, weight=1, timestamps=[ts])
    prev_selected_key = key_label

def reset_session():
    global current_session_selections, prev_selected_key, recording, recording_start_ts
    if recording:
        recording = False
    if current_session_selections:
        all_sessions.append(current_session_selections.copy())
    current_session_selections = []
    prev_selected_key = None
    print("[SESSION] Sesión guardada y reiniciada.")

def compute_and_save_graph(out_prefix="graph"):
    if len(G.nodes) == 0:
        print("El grafo está vacío — no hay selecciones aún.")
        return
    print("[GRAPH] Guardando y calculando métricas...")
    # métricas
    deg_in = dict(G.in_degree(weight='weight'))
    deg_out = dict(G.out_degree(weight='weight'))
    betw = nx.betweenness_centrality(G, weight='weight', normalized=True)
    pager = nx.pagerank(G, weight='weight')
    print("Nodos:", G.number_of_nodes(), "Aristas:", G.number_of_edges())
    print("Top grados (in):", sorted(deg_in.items(), key=lambda x: x[1], reverse=True)[:5])
    print("Top grados (out):", sorted(deg_out.items(), key=lambda x: x[1], reverse=True)[:5])
    print("Top betweenness:", sorted(betw.items(), key=lambda x: x[1], reverse=True)[:5])
    print("Top pagerank:", sorted(pager.items(), key=lambda x: x[1], reverse=True)[:5])

    # comunidades - greedy modularity
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G.to_undirected(), weight='weight'))
        comm_list = [list(c) for c in communities]
        print("Comunidades detectadas:", comm_list)
    except Exception as e:
        print("No se pudo ejecutar detección de comunidades:", e)
        comm_list = []

    # Dibujar grafo
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v].get('weight',1) for u,v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=[300 + 800*pager.get(n,0) for n in G.nodes()])
    nx.draw_networkx_edges(G, pos, width=[1+2*w for w in weights], arrowstyle='->', arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Grafo de selecciones")
    plt.axis('off')
    png_path = f"{out_prefix}.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close()
    # Guardar graphml
    graphml_path = f"{out_prefix}.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"[GRAPH] Guardado PNG: {png_path} y GraphML: {graphml_path}")

# ----------------- Bucle principal -----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(img_rgb)
        hands_results = hands.process(img_rgb)

        # Dibujar landmarks en frame (opcional)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hands_results.multi_hand_landmarks:
            for hand_lm in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # ESCALADO Y CENTRADO EN CANVAS
        h, w = frame.shape[:2]
        scale = min(SCREEN_W / w, SCREEN_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        x_off = (SCREEN_W - new_w) // 2
        y_off = (SCREEN_H - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = frame_resized

        # Definir teclado en coordenadas de canvas (parte superior dentro del frame_resized)
        kb_x = x_off
        kb_y = y_off
        kb_w = new_w
        kb_h = int(KEYBOARD_HEIGHT_FRAC * SCREEN_H)  # fijo respecto a pantalla para consistencia
        key_w = kb_w // len(KEY_LABELS)
        key_h = kb_h
        key_rects = []
        for i in range(len(KEY_LABELS)):
            rx = kb_x + i * key_w
            ry = kb_y
            key_rects.append((rx, ry, key_w, key_h))

        # Dibujar teclado
        for i, rect in enumerate(key_rects):
            rx, ry, rw, rh = rect
            color = KEY_COLORS[i % len(KEY_COLORS)]
            cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), color, thickness=-1)
            # overlay label
            cv2.putText(canvas, KEY_LABELS[i], (rx + 10, ry + rh//2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # Procesar manos: calcular posición de tip del índice (landmark 8)
        left_index_pos = None
        right_index_pos = None
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            # multi_hand_landmarks order matches multi_handedness
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right' (del punto de vista del modelo)
                lm = hand_landmarks.landmark[8]  # index finger tip
                # lm.x,y are normalized wrt original frame width/height
                px = int(lm.x * w * scale) + x_off
                py = int(lm.y * h * scale) + y_off
                if label == 'Left':
                    left_index_pos = (px, py)
                else:
                    right_index_pos = (px, py)
                # dibujar un pequeño círculo
                cv2.circle(canvas, (px, py), 8, (255,255,255), -1)

        # actualizar dwell para ambos dedos
        for finger_name, pos in [('left_index', left_index_pos), ('right_index', right_index_pos)]:
            state = dwell_state[finger_name]
            if pos is None:
                # dedo no detectado: reset temporally
                state['key'] = None
                state['start_ts'] = None
                state['confirmed'] = False
                continue
            px, py = pos
            # identificar tecla actual
            key_hit = None
            for idx, rect in enumerate(key_rects):
                if point_in_rect(px, py, rect):
                    key_hit = KEY_LABELS[idx]
                    # visualizar hover: dibujar borde
                    rx, ry, rw, rh = rect
                    cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), (255,255,255), 3)
                    # progreso si ya started
                    break
            # estado transitions
            if key_hit == state.get('key') and key_hit is not None:
                # sigue en la misma tecla
                if state.get('start_ts') is None:
                    state['start_ts'] = ts
                elapsed = ts - state['start_ts']
                # dibujar barra de progreso circular encima de tecla
                rx, ry, rw, rh = key_rects[KEY_LABELS.index(key_hit)]
                center = (rx + rw - 20, ry + 20)
                radius = 14
                angle = int(360 * min(1.0, elapsed / DWELL_SECONDS))
                cv2.ellipse(canvas, center, (radius,radius), -90, 0, angle, (255,255,255), 3)
                if elapsed >= DWELL_SECONDS and not state.get('confirmed'):
                    # CONFIRMACIÓN
                    state['confirmed'] = True
                    # registrar selección
                    timestamp_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                    print(f"[SELECT] {finger_name} -> {key_hit} @ {timestamp_iso}")
                    # Only add selection if recording mode is on
                    if recording:
                        current_session_selections.append((timestamp_iso, key_hit))
                    # agregar a grafo
                    add_selection_to_graph(key_hit, timestamp_iso)
            else:
                # nuevo key o fuera
                state['key'] = key_hit
                state['start_ts'] = ts if key_hit is not None else None
                state['confirmed'] = False

        # Detección de postura L (por brazo izquierdo y derecho)
        # Landmarks pose: left shoulder 11, left elbow 13, left wrist 15; right: 12,14,16
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            def safe_land(idx):
                try:
                    p = lm[idx]
                    return (p.x * w * scale + x_off, p.y * h * scale + y_off, p.z)
                except:
                    return None
            left_sh = safe_land(11)
            left_el = safe_land(13)
            left_wr = safe_land(15)
            right_sh = safe_land(12)
            right_el = safe_land(14)
            right_wr = safe_land(16)

            # left arm angle
            if left_sh and left_el and left_wr:
                v1 = (left_el[0]-left_sh[0], left_el[1]-left_sh[1])
                v2 = (left_wr[0]-left_el[0], left_wr[1]-left_el[1])
                a_left = angle_between(v1, v2)
                left_angles.append(a_left)
                avg_left = sum(left_angles)/len(left_angles)
                # chequear L
                if L_ANGLE_MIN <= avg_left <= L_ANGLE_MAX:
                    if left_L_start is None:
                        left_L_start = ts
                    elif (ts - left_L_start) >= L_PERSIST_SECONDS:
                        if not recording:
                            recording = True
                            recording_start_ts = ts
                            print("[POSE] L brazo izquierdo detectada → INICIAR grabación")
                else:
                    left_L_start = None
            # right arm angle
            if right_sh and right_el and right_wr:
                v1 = (right_el[0]-right_sh[0], right_el[1]-right_sh[1])
                v2 = (right_wr[0]-right_el[0], right_wr[1]-right_el[1])
                a_right = angle_between(v1, v2)
                right_angles.append(a_right)
                avg_right = sum(right_angles)/len(right_angles)
                if L_ANGLE_MIN <= avg_right <= L_ANGLE_MAX:
                    if right_L_start is None:
                        right_L_start = ts
                    elif (ts - right_L_start) >= L_PERSIST_SECONDS:
                        if recording:
                            # finalizar grabación
                            recording = False
                            # guardar la sesion si tiene algo
                            if current_session_selections:
                                all_sessions.append(current_session_selections.copy())
                            print("[POSE] L brazo derecho detectada → FINALIZAR grabación")
                            # mostrar mensaje reconstruido (concatenar labels)
                            reconstructed = " ".join([s[1] for s in current_session_selections])
                            print("[MESSAGE] Reconstruido:", reconstructed)
                            # limpiar session actual
                            current_session_selections = []
                else:
                    right_L_start = None

        # Mostrar estado en canvas
        status_text = f"Recording: {'ON' if recording else 'OFF'}  Selections (this session): {len(current_session_selections)}  Total nodes: {G.number_of_nodes()}"
        cv2.putText(canvas, status_text, (20, SCREEN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # salir
            print("Saliendo...")
            break
        elif key == ord('c'):
            # limpiar sesión actual
            current_session_selections = []
            prev_selected_key = None
            print("[ACTION] Sesión actual limpiada.")
        elif key == ord('g'):
            compute_and_save_graph()

except KeyboardInterrupt:
    print("Interrumpido por usuario.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    # al salir, si quedaron selecciones guardarlas
    if current_session_selections:
        all_sessions.append(current_session_selections.copy())
    # opcional: guardar grafo final
    if G.number_of_nodes() > 0:
        compute_and_save_graph(out_prefix="graph_on_exit")
    print("Terminado.")
