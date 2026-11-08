"""Cadastra o rosto de uma pessoa usando a webcam.

Uso: python cadastrar.py
  ESPAÇO -> tira uma foto    |    Q -> sai
Dica: tire de 5 a 10 fotos, com ângulos e luzes diferentes.
"""
import os
import cv2

PASTA_BASE = "base_faces"

nome = input("Nome da pessoa: ").strip().lower().replace(" ", "_")
if not nome:
    raise SystemExit("Nome inválido.")

pasta = os.path.join(PASTA_BASE, nome)
os.makedirs(pasta, exist_ok=True)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Não foi possível abrir a webcam.")

fotos = len(os.listdir(pasta))
print("ESPAÇO = tirar foto | Q = sair")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    visual = frame.copy()
    cv2.putText(visual, f"{nome} - fotos: {fotos} (ESPACO = foto, Q = sair)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Cadastro de rosto", visual)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord(" "):
        fotos += 1
        caminho = os.path.join(pasta, f"{nome}_{fotos}.jpg")
        cv2.imwrite(caminho, frame)
        print("Foto salva:", caminho)
    elif tecla == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print(f"Cadastro finalizado: {fotos} foto(s) de {nome}.")
