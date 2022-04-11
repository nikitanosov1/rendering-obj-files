import math

import numpy as np
from PIL import Image
import random
np.seterr(divide='ignore', invalid='ignore')

SCALE_ALL = 1

# FOR CAR
H = W = int(1000 * SCALE_ALL)
SCALE = int(200 * SCALE_ALL)

# FOR RABBIT
#H = W = 1000
#SCALE = 10000

SHIFT_Y = int(-500 * SCALE_ALL)

LIGHT_VECTOR = [0.0, 0.0, -1.0]
INIT_Z_BUFFER = 100000

def line1(x1, y1, x2, y2, matrixOfImage):
    count = 1000
    step = 1.0/count
    for k in range(count):
        try:
            t = step * k
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            matrixOfImage[y, x] = 255
        except Exception:
            pass


def drawStar1(x, y, r, matrixOfImage):
    a = 0
    step = math.pi/20
    while a < 2*math.pi:
        line1(x, y, x + r * math.cos(a), y + r * math.sin(a), matrixOfImage)
        a += step
    image = Image.fromarray(matrixOfImage, 'L')
    image.save("image1.jpg")

def line2(x1, y1, x2, y2, matrixOfImage):
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    for x in range(int(x1), int(x2)):
        try:
            t = (x - x1) / (x2 - x1)
            y = int(y1 + t * (y2 - y1))
            matrixOfImage[y, x] = 255
        except Exception:
            pass

def drawStar2(x, y, r, matrixOfImage):
    a = 0
    step = math.pi/20
    while a < 2*math.pi:
        line2(x, y, x + r * math.cos(a), y + r * math.sin(a), matrixOfImage)
        a += step
    image = Image.fromarray(matrixOfImage, 'L')
    image.save("image2.jpg")

def line3(x1, y1, x2, y2, matrixOfImage):
    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    for y in range(int(y1), int(y2)):
        try:
            t = (y - y1) / (y2 - y1)
            x = int(x1 + t * (x2 - x1))
            matrixOfImage[y, x] = 255
        except Exception:
            pass

def drawStar3(x, y, r, matrixOfImage):
    a = 0
    step = math.pi/20
    while a < 2*math.pi:
        line3(x, y, x + r * math.cos(a), y + r * math.sin(a), matrixOfImage)
        a += step
    image = Image.fromarray(matrixOfImage, 'L')
    image.save("image3.jpg")

def drawStar4(x, y, r, matrixOfImage):
    a = 0
    step = math.pi/20
    while a < 2*math.pi:
        x1 = x + r * math.cos(a)
        y1 = y + r * math.sin(a)
        if abs(y1 - y) > abs(x1 - x):
            line3(x, y, x1, y1, matrixOfImage)
        else:
            line2(x, y, x1, y1, matrixOfImage)
        a += step
    image = Image.fromarray(matrixOfImage, 'L')
    image.save("image4.jpg")

def line5(x1, y1, x2, y2, matrixOfImage):
    try:
        if abs(y2 - y1) > abs(x2 - x1):
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            error = 0
            dx = x2 - x1
            dy = y2 - y1
            derror = dx / dy
            y = y1
            x = x1
            while y < y2:
                matrixOfImage[int(y + 0.5), math.floor(x)] = 255 * (1 - abs(error))
                matrixOfImage[int(y + 0.5), math.floor(x) + 1] = 255 * abs(error)

                error += derror
                if abs(error) > 0.5:
                    if error > 0.5:
                        x += 1
                        error -= 1
                    if error < -0.5:
                        x -= 1
                        error += 1
                y += 1

        else:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            error = 0
            dx = x2 - x1
            dy = y2 - y1
            derror = dy/dx
            y = y1
            x = x1
            while x < x2:
                matrixOfImage[math.floor(y), int(x + 0.5)] = 255 * (1 - abs(error))
                matrixOfImage[math.floor(y) + 1, int(x + 0.5)] = 255 * abs(error)

                error += derror
                if abs(error) > 0.5:
                    if error > 0.5:
                        y += 1
                        error -= 1
                    if error < -0.5:
                        y -= 1
                        error += 1
                x += 1

    except Exception:
        pass

def line6(x1, y1, x2, y2, matrixOfImage):
    try:
        if abs(y2 - y1) > abs(x2 - x1):
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            error = 0
            dx = x2 - x1
            dy = y2 - y1
            derror = dx / dy
            y = y1
            x = x1
            while y < y2:
                matrixOfImage[int(y), int(x)] = 255

                error += derror
                if abs(error) > 0.5:
                    if error > 0.5:
                        x += 1
                        error -= 1
                    if error < -0.5:
                        x -= 1
                        error += 1
                y += 1

        else:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            error = 0
            dx = x2 - x1
            dy = y2 - y1
            derror = dy/dx
            y = y1
            x = x1
            while x < x2:
                matrixOfImage[int(y), int(x)] = 255

                error += derror
                if abs(error) > 0.5:
                    if error > 0.5:
                        y += 1
                        error -= 1
                    if error < -0.5:
                        y -= 1
                        error += 1
                x += 1

    except Exception:
        pass

def drawStar5(x, y, r, matrixOfImage):
    a = 0
    step = math.pi/30
    while a < 2*math.pi:
        line5(x, y, x + r * math.cos(a), y - r * math.sin(a), matrixOfImage)
        a += step
    image = Image.fromarray(matrixOfImage, 'L')
    image.save("image5.jpg")

def calcBarCoords(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = -1
    lambda1 = -1
    lambda2 = -1
    try:
        lambda0 = float(((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2)))
        lambda1 = float(((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)))
        lambda2 = float(((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1)))
        #print(lambda0 + lambda1 + lambda2)
    except Exception:
        pass
    return [lambda0, lambda1, lambda2]

def checkAllElemOfArrayArePositive(x):
    for i in range(len(x)):
        if x[i] < 0:
            return False
    return True

def drawTriangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, zBuffer, matrixOfImage):
    xmin = min(x0, x1, x2)
    if xmin < 0:
        xmin = 0
    ymin = min(y0, y1, y2)
    if ymin < 0:
        ymin = 0
    xmax = max(x0, x1, x2)
    if xmax > W:
        xmax = W - 1
    ymax = max(y0, y1, y2)
    if ymax > H:
        ymax = H - 1

    normVectorOfPolygon = calcNormVector(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cos = cosBetweenVectors(LIGHT_VECTOR, normVectorOfPolygon)
    if cos > 0:
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        for x in range(int(xmin), int(xmax) + 1):
            for y in range(int(ymin), int(ymax) + 1):
                barCoords = calcBarCoords(x, y, x0, y0, x1, y1, x2, y2)
                if checkAllElemOfArrayArePositive(barCoords):
                    z = barCoords[0] * z0 + barCoords[1] * z1 + barCoords[2] * z2
                    if z < zBuffer[y, x]:
                        #matrixOfImage[y, x] = [r * abs(cos), g * abs(cos), b * abs(cos)]
                        matrixOfImage[y, x] = [255 * abs(cos), 0, 0]
                        zBuffer[y, x] = z

def drawPolygonWith4tops(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, zBuffer, matrixOfImage):
    drawTriangle(x0, y0, z0,
                 x1, y1, z1,
                 x3, y3, z3,
                 zBuffer, matrixOfImage)
    drawTriangle(x1, y1, z1,
                 x2, y2, z2,
                 x3, y3, z3,
                 zBuffer, matrixOfImage)

def calcNormVector(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    a = [x1-x0, y1-y0, z1-z0]
    b = [x1-x2, y1-y2, z1-z2]
    c = np.cross(a, b)
    return c / np.linalg.norm(c)


def cosBetweenVectors(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.clip(np.dot(v1, v2), -1.0, 1.0)

def run():
    zBuffer = np.zeros((H, W), dtype=np.uint8) + INIT_Z_BUFFER
    matrixOfImage = np.zeros((H, W, 3), dtype=np.uint8)
    #drawStar1(100, 100, 90, matrixOfImage.copy())
    #drawStar2(100, 100, 90, matrixOfImage.copy())
    #drawStar3(100, 100, 90, matrixOfImage.copy())
    #drawStar4(100, 100, 90, matrixOfImage.copy())
    #drawStar5(100, 100, 70, matrixOfImage.copy())


    tops = []  # Массив вершин
    polygons = []
    temp = []
    #file = open("fox.obj", "r")
    #file = open("another_model.obj", "r")
    file = open("Audi RS7 Sport Perfomance.obj", "r")
    for line in file:
        line = line.split()
        if len(line) > 0 and line[0] == 'v':
            temp = list(map(float, [line[1], line[2], line[3]]))
            for i in range(len(temp)):
                temp[i] = SCALE * temp[i] + W/2 # Сохраняю сразу отмасштабированные вершины
            temp[1] += SHIFT_Y
            tops.append(temp)
        if len(line) > 0 and line[0] == 'f':
            if len(line) == 4:
                polygons.append([int(line[1].split("/")[0]), int(line[2].split("/")[0]), int(line[3].split("/")[0])])
            if len(line) == 5:
                polygons.append([int(line[1].split("/")[0]), int(line[2].split("/")[0]), int(line[3].split("/")[0]), int(line[4].split("/")[0])])
    file.close()


    # ОТРИСОВКА ВЕРШИН
    #for top in tops:
    #    x = top[0]
    #    y = top[1]
    #    z = top[2]
    #    matrixOfImage[int(H - y), int(x)] = 255

    for polygon in polygons:
        top1 = tops[polygon[0] - 1]
        top2 = tops[polygon[1] - 1]
        top3 = tops[polygon[2] - 1]
        #line5(top1[0], H - top1[1], top2[0], H - top2[1], matrixOfImage)
        #line5(top2[0], H - top2[1], top3[0], H - top3[1], matrixOfImage)
        #line5(top3[0], H - top3[1], top1[0], H - top1[1], matrixOfImage)
        if len(polygon) == 3:
            drawTriangle(top1[0], H - top1[1], top1[2],
                         top2[0], H - top2[1], top2[2],
                         top3[0], H - top3[1], top3[2],
                         zBuffer, matrixOfImage)
        elif len(polygon) == 4:
            top4 = tops[polygon[3] - 1]
            drawPolygonWith4tops(top1[0], H - top1[1], top1[2],
                                 top2[0], H - top2[1], top2[2],
                                 top3[0], H - top3[1], top3[2],
                                 top4[0], H - top4[1], top4[2],
                                 zBuffer, matrixOfImage)
        else:
            drawTriangle(top1[0], H - top1[1], top1[2],
                         top2[0], H - top2[1], top2[2],
                         top3[0], H - top3[1], top3[2],
                         zBuffer, matrixOfImage)

    image = Image.fromarray(matrixOfImage, 'RGB')
    image.save("image6.jpg")

if __name__ == '__main__':
    run()
