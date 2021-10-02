import cv2
import numpy as np 

#lower = {'red': (166, 84,141), 'blue':(97, 100, 117), 'green': (36, 0, 0)}
#upper = {'red': (186, 255, 255), 'blue': (117, 255, 255),'green': (75, 255, 255) }


#Função para redimensionar a imagem 
def redim(img, largura):
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation = cv2.cv2.INTER_AREA)
    return img 

camera = cv2.VideoCapture(0)

while True:
    (sucesso, frame) = camera.read()
    if not sucesso:
        break
    frame = redim(frame, 320)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((9,9), np.uint8)
#mascara de vermelho 
    mask1 = cv2.inRange(frame_hsv,(166, 84,141), (186, 255, 255))
#mascara de azul
    mask2 = cv2.inRange(frame_hsv, (99, 100, 117), (117, 255, 255))
#mascara de verde
    mask3 = cv2.inRange(frame_hsv,(36, 0, 0), (75, 255,255))
    mask0 = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask0, mask3)
#tirar ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        epsilon = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            cv2.putText(frame, 'Triangulo', (x, y), 1, 1.5, (0, 255, 255), 2)
    
        if len(approx) == 4:
            aspect_ratio = float(w)/h
            if aspect_ratio == 1:
                cv2.putText(frame, 'Quadrado', (x, y), 1, 1.5, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'Retangulo', (x, y), 1, 1.5, (0, 255, 255), 2)
    
        if len(approx) == 5:
            cv2.putText(frame, 'Pentagono', (x, y), 1, 1.5, (0, 255, 255), 2)

        if len(approx) == 6:
            cv2.putText(frame, 'Hexagono', (x, y), 1, 1.5, (0, 255, 255), 2)
    
        if len(approx) > 10:
            cv2.putText(frame, 'Circulo', (x, y), 1, 1.5, (0, 255, 255), 2)
    
        cv2.drawContours(frame, [approx], 0, (0,255,0),2)

    cv2.imshow('Tracking', redim(frame, 640))
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

camera.release()
cv2.destroyAllWindows()
