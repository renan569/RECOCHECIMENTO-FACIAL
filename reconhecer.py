"""Reconhecimento facial em tempo real com DeepFace + OpenCV.

Uso: python reconhecer.py      (Q para sair)
Antes, cadastre pessoas com: python cadastrar.py
"""
import os
import cv2
from deepface import DeepFace

PASTA_BASE = "base_faces"
MODELO = "Facenet"            # outros: VGG-Face, ArcFace, SFace...
DETECTOR = "opencv"           # outros: retinaface, mtcnn (mais precisos, mais lentos)
METRICA = "cosine"
LIMITE_PADRAO = 0.40          # usado se o DeepFace não informar o limite
ANALISAR_A_CADA = 10          # analisa 1 frame a cada N (deixa o vídeo fluido)

if not os.path.isdir(PASTA_BASE) or not os.listdir(PASTA_BASE):
    raise SystemExit("Nenhum rosto cadastrado. Rode primeiro: python cadastrar.py")


def reconhecer(frame):
    """Retorna lista de (x, y, w, h, nome, distancia) para os rostos do frame."""
    resultados = []
    try:
        encontrados = DeepFace.find(
            img_path=frame,
            db_path=PASTA_BASE,
            model_name=MODELO,
            detector_backend=DETECTOR,
            distance_metric=METRICA,
            enforce_detection=False,
            silent=True,
        )
    except Exception as erro:
        print("Erro na análise:", erro)
        return resultados

    for tabela in encontrados:
        if tabela.empty:
            continue
        melhor = tabela.iloc[0]
        x, y = int(melhor["source_x"]), int(melhor["source_y"])
        w, h = int(melhor["source_w"]), int(melhor["source_h"])
        if w == 0 or h == 0:       # nenhum rosto detectado
            continue
        limite = float(melhor["threshold"]) if "threshold" in tabela.columns else LIMITE_PADRAO
        distancia = float(melhor["distance"])
        if distancia <= limite:
            nome = os.path.basename(os.path.dirname(melhor["identity"]))
        else:
            nome = "desconhecido"
        resultados.append((x, y, w, h, nome, distancia))
    return resultados


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Não foi possível abrir a webcam.")

contador = 0
ultimos = []

print("Reconhecimento iniciado. Pressione Q para sair.")
while True:
    ok, frame = camera.read()
    if not ok:
        break

    if contador % ANALISAR_A_CADA == 0:
        ultimos = reconhecer(frame)
    contador += 1

    for x, y, w, h, nome, distancia in ultimos:
        cor = (0, 200, 0) if nome != "desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, f"{nome} ({distancia:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

    cv2.imshow("Reconhecimento facial", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
