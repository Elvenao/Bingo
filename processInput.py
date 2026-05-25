import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import json

#This model is for predicting the position of 
#the bounding boxes for the circles 
circlesModel = YOLO("detectCircles.pt")
#This model is for predicting the position of
#the bounding box for the card
cardModel = YOLO("detectCard.pt")  
CONF_CARD = 0.70
CONF_CIRCLE = 0.50

#This method is for ordering the 
def order_corners(pts):
    # Ordenar: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def orderPoints(result):

    if len(result[0].boxes.xyxy) != 24:
        return None
    filas = 5
    columnas = 5
    r2 = result[0].boxes.xyxy.tolist()
    #Order by Cx
    for i in range(len(r2)):
        for j in range(len(r2)-1-i):
            if(r2[j][0] > r2[j+1][0]):
                aux = r2[j]
                r2[j] = r2[j+1]
                r2[j+1] = aux

    newR2 = [[0] * columnas for _ in range(filas)]
    x = 0
    y = 0

    #Put all those values into a new list adding the 0 values for the center
    for i in range(24):
        if(x != 2 or y != 2):
            newR2[x][y] = r2[i]
        else:
            newR2[x][y] = [0, 0, 0, 0]
            y+=1
            newR2[x][y] = r2[i]
            
        y += 1
        if y >= columnas:
            y = 0
            x += 1

    #Rearrange all these values by Y
    for i in range(5):
        for k in range(5):
            for j in range(4-k):
                if newR2[i][j][1] > newR2[i][j+1][1]:
                    aux = newR2[i][j]
                    newR2[i][j] = newR2[i][j+1] 
                    newR2[i][j+1] = aux

    for i in range(2):
        aux  = newR2[2][i] 
        newR2[2][i] = newR2[2][i+1] 
        newR2[2][i+1] = aux

    newList = []
    for i in range(5):
        for j in range(5):
            val = newR2[j][i]
            if val != 0 and val != [0, 0, 0, 0]:
                newList.append(val)
    
        
    return newList

def process_input(entrada):

    img = entrada
    
    # Step 1: Detect the card
    res_card = cardModel.predict(img, verbose=False)

    if len(res_card) == 0 or res_card[0].obb is None:
        return 0

    r = res_card[0].obb

    if len(r) == 0:
        return 0
    conf_card = float(r.conf[0])

    # Low confidence
    if conf_card < CONF_CARD:
        return 0

    try:

        # Sort 4 corners
        esquinas = order_corners(r.xyxyxyxy[0].cpu().numpy())
        
        w = int(float(r.xywhr[0][2]))  # actual width
        h = int(float(r.xywhr[0][3]))  # actual height
        if w <= 0 or h <= 0:
            return 0

        dst = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        # Perspective transformation
        M = cv2.getPerspectiveTransform(esquinas.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (w, h))
    except:
        return 0

    warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    # Step 2: Detect circles from the warped card
    res_circles = circlesModel.predict(warped_pil, verbose=False)

    if len(res_circles) == 0:
        return 0

    boxes = res_circles[0].boxes

    if boxes is None:
        return 0
    # If circles detected are different from 24
    # return error
    if len(boxes) != 24:
        return 0
    confs = boxes.conf.cpu().numpy()

    av = np.mean(confs)

    if av < CONF_CIRCLE:
        return 0
    for c in confs:
        if c < 0.30:
            return 0
    # Paso 2: ordenar
    sorted_boxes = orderPoints(res_circles)

    if sorted_boxes is None:
        return 0

    if len(sorted_boxes) != 24:
        return 0

    
    crops = []
    # Paso 6: guardar crops
    try:

        for box in sorted_boxes:

            x1, y1, x2, y2 = map(int, box)

            crop = warped_pil.crop((x1, y1, x2, y2))

            crops.append(crop)

    except:
        return 0
    return crops