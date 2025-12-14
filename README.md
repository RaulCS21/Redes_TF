# Sistema de Comunicación Gestual Basado en Grafos

## Objetivo del proyecto

Diseñar e implementar un sistema analítico–computacional en tiempo real que permita la construcción de mensajes mediante interacción gestual sin contacto, integrando detección de postura corporal y manos, un teclado virtual modelado como grafo y técnicas de análisis topológico.

El sistema permite seleccionar palabras a través de gestos de la mano sobre un teclado virtual de 10 teclas, registrar la secuencia de interacción como una red dirigida y analizarla mediante métricas de teoría de grafos, centralidad, comunidades y procesos de difusión conceptual.


## Arquitectura propuesta

La arquitectura del sistema se organiza de forma modular y reproducible:

```
Redes_TF/
├── src/          
│   ├── deteccion_manos.py    
├── outputs
├── README.md
├── requirements.txt
└── .gitignore
```

**Flujo general del sistema:**

1. Captura de video en tiempo real mediante OpenCV.
2. Detección de manos y landmarks con MediaPipe.
3. Detección de postura en L para control de inicio y fin de grabación.
4. Interacción con el teclado virtual mediante permanencia del dedo índice.
5. Registro de transiciones entre palabras como aristas de un grafo dirigido.
6. Análisis estructural del grafo generado.
7. Visualización y exportación de resultados.

---

## Instrucciones de instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/RaulCS21/Redes_TF.git
cd Redes_TF
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

### 3. Activar entorno virtual

**Windows:**

```bash
.venv\Scripts\activate
```


### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Instrucciones de ejecución

Desde la raíz del proyecto, ejecutar:

```bash
python src/deteccion_manos.py  
```

Al iniciar el sistema:

* Una postura en **L con el brazo izquierdo** activa la grabación.
* Una postura en **L con el brazo derecho** finaliza la grabación.
* Mantener el dedo índice sobre una tecla por ≥ 3 segundos selecciona una palabra.
* Al finalizar, el grafo de interacción se guarda automáticamente en `outputs/`.

---

## Requisitos

Los requerimientos del proyecto se encuentran especificados en el archivo `requirements.txt`:

```
opencv-python
mediapipe
networkx
matplotlib
numpy
```
