# импорт необходимых пакетов
import numpy as np
import argparse
import time
import cv2
import os

# построить аргумент синтаксический анализ и разбор аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# загрузить метки класса COCO, на которые была обучена наша модель YOLO
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# инициализировать список цветов для представления каждой возможной метки класса
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# производные пути к весам YOLO и конфигурации модели
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# загрузить наш детектор объектов YOLO, обученный на наборе данных COCO (80 классов)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# загрузите наше входное изображение и захватите его пространственные размеры
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# определяют только имена *выходных* слоев, которые нам нужны из YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# создайте большой двоичный объект из входного изображения, а затем выполните прямое
# проход детектора объектов YOLO, дающий нам наши ограничители и
# связанные вероятности
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# показать информацию о времени на YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# инициализируйте наши списки обнаруженных ограничителений, доверительных связей и
# идентификаторы классов соответственно
boxes = []
confidences = []
classIDs = []

# цикл над каждым из выходов слоя
for output in layerOutputs:
    # цикл над каждым из обнаружений
    for detection in output:
        # извлеките идентификатор класса и достоверность (т.е. вероятность)
        # обнаружение текущего объекта
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # отфильтровывайте слабые прогнозы, обеспечивая обнаружение
        # вероятность больше минимальной вероятности
        if confidence > args["confidence"]:
            # масштабируйте координаты ограничительной рамки относительно
            # размер изображения, имея в виду, что YOLO на самом деле
            # возвращает центральную (x, y)-координаты границы
            # box, за которым следуют ширина и высота коробок
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # использовать центральные (x, y)-координаты для получения верхней и
            # и левый угол ограничителя
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # обновить наш список координат ограничителю, доверительных ложей,
            # и идентификаторы классов
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# применять подавление без максимумов для подавления слабых, перекрывающихся границ
# коробки
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])

# убедитесь, что существует хотя бы одно обнаружение
if len(idxs) > 0:
    # цикл над индексами, которые мы сохраняем
    for i in idxs.flatten():
        # извлечь координаты ограничителя
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # нарисуйте прямоугольник ограничителя и метку на изображении
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)
# показать выходное изображение
cv2.imshow("Image", image)
cv2.waitKey(0)