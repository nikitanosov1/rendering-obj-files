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

LIGHT_VECTOR = [0.0, 0.0, 1.0]
INIT_Z_BUFFER = 100000

norm_vector_of_tops = []

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

def drawTriangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, zBuffer, matrixOfImage, index_top0, index_top1, index_top2):

    v0 = norm_vector_of_tops[index_top0]
    v1 = norm_vector_of_tops[index_top1]
    v2 = norm_vector_of_tops[index_top2]

    x0, y0, z0 = rotate(x0, y0, z0)
    x1, y1, z1 = rotate(x1, y1, z1)
    x2, y2, z2 = rotate(x2, y2, z2)

    # print (v0)


    v0[0], v0[1], v0[2] = rotate(v0[0], v0[1], v0[2])
    v1[0], v1[1], v1[2] = rotate(v1[0], v1[1], v1[2])
    v2[0], v2[1], v2[2] = rotate(v2[0], v2[1], v2[2])


    # print (v0)
    # exit()

    l0 = cosBetweenVectors(LIGHT_VECTOR, v0)
    l1 = cosBetweenVectors(LIGHT_VECTOR, v1)
    l2 = cosBetweenVectors(LIGHT_VECTOR, v2)

    #l0, l1, l2 = abs(l0), abs(l1), abs(l2)

    x0_screen, y0_screen, z0_screen = pixelToScreenPixel(x0, y0, z0)
    #print(x0, y0, z0, x0_screen, y0_screen, z0_screen)
    x1_screen, y1_screen, z1_screen = pixelToScreenPixel(x1, y1, z1)
    x2_screen, y2_screen, z2_screen = pixelToScreenPixel(x2, y2, z2)

    xmin = min(x0_screen, x1_screen, x2_screen)
    if xmin < 0:
        xmin = 1
    ymin = min(y0_screen, y1_screen, y2_screen)
    if ymin < 0:
        ymin = 1
    xmax = max(x0_screen, x1_screen, x2_screen)
    if xmax > W:
        xmax = W - 1
    ymax = max(y0_screen, y1_screen, y2_screen)
    if ymax > H:
        ymax = H - 1

    normVectorOfPolygon = calcNormVector(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cos = cosBetweenVectors(LIGHT_VECTOR, normVectorOfPolygon)
    if cos < 0:
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        for x in range(int(xmin), int(xmax) + 1):
            for y in range(int(ymin), int(ymax) + 1):
                barCoords = calcBarCoords(x, y, x0_screen, y0_screen, x1_screen, y1_screen, x2_screen, y2_screen)
                if checkAllElemOfArrayArePositive(barCoords):
                    z = barCoords[0] * z0 + barCoords[1] * z1 + barCoords[2] * z2
                    if z < zBuffer[y, x]:
                        #matrixOfImage[y, x] = [r * abs(cos), g * abs(cos), b * abs(cos)]
                        #matrixOfImage[H - y, x] = [255 * abs(cos), 0, 0]

                        matrixOfImage[H - y, x] = [-255 * (l0 * barCoords[0] + l1 * barCoords[1] + l2 * barCoords[2]), 0, 0]

                        #print(y, x)
                        zBuffer[y, x] = z

def drawPolygonWith4tops(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, zBuffer, matrixOfImage, index_top0, index_top1, index_top2, index_top3):
    drawTriangle(x0, y0, z0,
                 x1, y1, z1,
                 x3, y3, z3,
                 zBuffer, matrixOfImage,
                 index_top0, index_top1, index_top3)
    drawTriangle(x1, y1, z1,
                 x2, y2, z2,
                 x3, y3, z3,
                 zBuffer, matrixOfImage,
                 index_top1, index_top2, index_top3)

def calcNormVector(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    a = [x1-x0, y1-y0, z1-z0]
    b = [x1-x2, y1-y2, z1-z2]
    c = -np.cross(a, b)
    return c / np.linalg.norm(c)

def cosBetweenVectors(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.clip(np.dot(v1, v2), -1.0, 1.0)

def pixelToScreenPixel(x, y, z):
    pixel = np.array([[x], [y], [z]], float)
    #scale = 1
    scale = 0.5

    t = np.array([[0], [0], [0.15*scale]], float)
    intrinsic = np.array([[400*scale, 0, 500], [0, 400*scale, 500], [0, 0, 1]], float)
    result = np.dot(intrinsic, pixel + t)
    result /= result[2]
    result = list(map(float, result))
    return result[0], result[1], result[2]

def rotate(x, y, z):
    a = 0
    b = math.pi/15#math.pi/4
    c = 0#math.pi/4
    pixel = np.array([[x], [y], [z]], float)
    X_rotate = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]], float)
    Y_rotate = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]], float)
    Z_rotate = np.array([[math.cos(c), math.sin(c), 0], [-math.sin(c), math.cos(c), 0], [0, 0, 1]], float)
    result = np.dot(Z_rotate, pixel)
    result = np.dot(Y_rotate, result)
    result = np.dot(X_rotate, result)
    result = list(map(float, result))
    return result[0], result[1], result[2]

def run():
    zBuffer = np.full((H, W), np.inf, dtype=np.float64)
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
    file = open("another_model.obj", "r")
    #file = open("Audi RS7 Sport Perfomance.obj", "r")
    for line in file:
        line = line.split()
        if len(line) > 0 and line[0] == 'v':
            temp = list(map(float, [line[1], line[2], line[3]]))
            #for i in range(len(temp)):
                #temp[i] = SCALE * temp[i] + W/2 # Сохраняю сразу отмасштабированные вершины
            #temp[1] += SHIFT_Y
            tops.append(temp)
        if len(line) > 0 and line[0] == 'f':
            if len(line) == 4:
                polygons.append([int(line[1].split("/")[0]), int(line[2].split("/")[0]), int(line[3].split("/")[0]),
                                 int(line[1].split("/")[2]), int(line[2].split("/")[2]), int(line[3].split("/")[2])])

            if len(line) == 5:
                polygons.append([int(line[1].split("/")[0]), int(line[2].split("/")[0]), int(line[3].split("/")[0]), int(line[4].split("/")[0]),
                                 int(line[1].split("/")[2]), int(line[2].split("/")[2]), int(line[3].split("/")[2]), int(line[4].split("/")[2])])

        if len(line) > 0 and line[0] == 'vn':
            temp = list(map(float, [line[1], line[2], line[3]]))
            norm_vector_of_tops.append(temp)

    print (norm_vector_of_tops[0])
    print(polygons[0])
    print(len(tops))
    print(len(norm_vector_of_tops))
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
        if len(polygon) == 6:
            drawTriangle(top1[0], top1[1], top1[2],
                         top2[0], top2[1], top2[2],
                         top3[0], top3[1], top3[2],
                         zBuffer, matrixOfImage,
                         polygon[3] - 1, polygon[4] - 1, polygon[5] - 1)
        elif len(polygon) == 8:
            top4 = tops[polygon[3] - 1]

            drawPolygonWith4tops(top1[0], top1[1], top1[2],
                                 top2[0], top2[1], top2[2],
                                 top3[0], top3[1], top3[2],
                                 top4[0], top4[1], top4[2],
                                 zBuffer, matrixOfImage,
                                 polygon[4] - 1, polygon[5] - 1, polygon[6] - 1, polygon[7] - 1)
        else:
            drawTriangle(top1[0], top1[1], top1[2],
                         top2[0], top2[1], top2[2],
                         top3[0], top3[1], top3[2],
                         zBuffer, matrixOfImage,
                         polygon[3] - 1, polygon[4] - 1, polygon[5] - 1)

    image = Image.fromarray(matrixOfImage, 'RGB')
    image.save("image6.jpg")

if __name__ == '__main__':
    run()
