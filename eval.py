from pathlib import Path
import torch
import os
import math
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from ultralytics import YOLO
import asyncio
import cv2
from shapely.geometry import Polygon
from utils import config
import pandas as pd
from scipy.spatial.distance import cdist

class_mapping = {}

class YoloSensitiveDetector():
    def __init__(self, detection_model, classification_model, img_size, img_size_pos, batch_size, iou):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.img_size = img_size
        self.img_size_pos = img_size_pos
        self.batch_size = batch_size
        self.post_processor = YoloPosProcessing()
        self.output_dir = f'{config.getPath()}/runs/classification/predict'
        self.iou = iou
        os.makedirs(self.output_dir, exist_ok=True)

        if torch.cuda.is_available():
            self.detection_model.cuda()
            self.classification_model.cuda()
        else:
            self.detection_model.cpu()
            self.classification_model.cpu()

    async def inference(self, input_img):

        # Verifica se a imagem está no formato BGR (OpenCV)
        if input_img.shape[2] == 3:  # Verifica se a imagem tem 3 canais
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = Image.fromarray(input_img)
        detection_results = self.detection_model.predict(source=input_img, save=True, save_conf=False, verbose=False,
                                                         imgsz=self.img_size, conf=0.1, iou=self.iou)
        if len(detection_results) == 0 or len(detection_results[0].boxes) == 0:
            return {}, {}

        dict_results = {"names": detection_results[0].names, "bboxes": [], "classes": [], "confs": []}
        dict_results["bboxes"] = detection_results[0].boxes.xywh.int().tolist()
        dict_results["classes"] = detection_results[0].boxes.cls.tolist()
        dict_results["confs"] = detection_results[0].boxes.conf.tolist()

        post_processed_images = self.post_processor.run_post_process(input_img, dict_results)
        dict_results_pos = {'names': {}, 'softs': [], 'pred': []}

        num_batches = math.ceil(len(post_processed_images) / self.batch_size)
        indices_to_remove = []

        for i in range(num_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(post_processed_images))
            batch_images = post_processed_images[batch_start:batch_end]

            results = self.classification_model.predict(
                source=batch_images,
                save=False,
                save_conf=False,
                verbose=True,
                imgsz=self.img_size_pos,
                show_conf=False,
                project=self.output_dir,
                name=f'batch_{i}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )

            for j, result in enumerate(results):
                dict_results_pos['names'] = result.names
                dict_results_pos['softs'].append(result.probs.data.tolist())
                dict_results_pos['pred'].append([result.probs.top1])

        for index in sorted(indices_to_remove, reverse=True):
            dict_results['bboxes'].pop(index)
            dict_results['classes'].pop(index)
            dict_results['confs'].pop(index)

        return dict_results, dict_results_pos

    async def async_run(self, input_img):
        return await self.inference(input_img)

class YoloPosProcessing():
    def __init__(self):
        self.class_colors = {
            'HSIL': 'red',
            'LSIL': 'orange',
        }

    def run_post_process(self, image_np, detection_results):
        bboxes = detection_results["bboxes"]
        cropped_images = self.extract_cropped_images(image_np, bboxes)
        return cropped_images

    def extract_cropped_images(self, image_np, bboxes):
        cropped_images = []
        for bbox in bboxes:
            center_x, center_y, width, height = bbox

            width = float(width)
            height = float(height)
            max_value = max(width, height)
            width = height = max_value

            left = center_x - (width / 2)
            top = center_y - (height / 2)
            right = center_x + (width / 2)
            bottom = center_y + (width / 2)

            crop_left = max(0, left)
            crop_top = max(0, top)
            crop_right = min(image_np.size[0], right)
            crop_bottom = min(image_np.size[1], bottom)

            cropped_img = image_np.crop((crop_left, crop_top, crop_right, crop_bottom))
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - image_np.size[0])
            pad_bottom = max(0, bottom - image_np.size[1])

            padded_img = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
            padded_img.paste(cropped_img, (int(pad_left), int(pad_top)))
            cropped_images.append(padded_img)

        return cropped_images

class ObjectDetectionEvaluator:
    def __init__(self, detection_model, classification_model, img_size, img_size_pos, batch_size, iou):
        self.detector = YoloSensitiveDetector(detection_model, classification_model, img_size, img_size_pos, batch_size, iou)

        self.class_colors = {
            'HSIL': 'red',
            'LSIL': 'orange',
            'NORMAL': 'green'
        }
        self.class_mapping = {0: 'HSIL', 1: 'LSIL', 2: 'NORMAL', 3: 'BACKGROUND'}

    def load_yolo_annotation(self, txt_file, img_width, img_height):
        bboxes = []
        classes = []

        with open(txt_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                class_id = int(values[0])
                if len(values) == 5:
                    # Bounding box annotation
                    x_center = values[1] * img_width
                    y_center = values[2] * img_height
                    width = values[3] * img_width
                    height = values[4] * img_height

                    # Append bounding box coordinates in the format (left, top, right, bottom)
                    bboxes.append([x_center, y_center, width, height])

                    classes.append(class_id)
                else:
                    # Polygon annotation (assuming format: class_id x1 y1 x2 y2 ... xn yn)
                    x_coords = [values[i] * img_width for i in range(1, len(values), 2)]
                    y_coords = [values[i + 1] * img_height for i in range(1, len(values), 2)]

                    # Calculate the bounding box coordinates
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)

                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2

                    # Append bounding box coordinates in the format (left, top, right, bottom)
                    bboxes.append([x_center, y_center, width, height])

                    classes.append(class_id)

        return bboxes, classes

    def draw_bounding_boxes(self, image, bboxes, classes, confidences):
            draw = ImageDraw.Draw(image)
            class_mapping_count = [('HSIL', 0), ('LSIL', 0), ('NORMAL', 0)]
            global class_mapping   

            for bbox, cls, confidence in zip(bboxes, classes, confidences):
                center_x, center_y, width, height = bbox
                left = center_x - (width / 2)
                top = center_y - (height / 2)
                right = center_x + (width / 2)
                bottom = center_y + (width / 2)
                class_name = self.class_mapping[cls]
                
                # Incrementa a contagem da classe
                for index, tuple_classes in enumerate(class_mapping_count):
                    if class_name in tuple_classes[0]:
                        class_mapping_count[index] = (class_name, tuple_classes[1] + 1)

                # Não desenha bounding boxes da classe 2
                if cls == 2:
                    continue

                color = self.class_colors.get(class_name, 'white')
                draw.rectangle([left, top, right, bottom], outline=color, width=4)
                text = f"{class_name}: {confidence:.2f}"
                draw.text((left, top), text, fill=color, font_size=20)

            return image, class_mapping_count


    def calculate_bbox_area(self, width, height):
        return width * height

    def calculate_union_area(self, box1, box2):
        # Extrair coordenadas e dimensões das caixas delimitadoras
        x1_min, y1_min, w1, h1, _ = box1
        x2_min, y2_min, w2, h2, _ = box2

        # Calcular x_max e y_max para cada caixa
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        # Calcular a área das caixas delimitadoras
        area1 = self.calculate_bbox_area(w1, h1)
        area2 = self.calculate_bbox_area(w2, h2)

        # Calcular as coordenadas da interseção
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calcular a área de interseção
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calcular a área de união
        union_area = area1 + area2 - inter_area

        return union_area

    def is_inside(self, box1, box2):
        x1_min, y1_min, w1, h1, _ = box1
        x2_min, y2_min, w2, h2, _ = box2

        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        return x2_min >= x1_min and y2_min >= y1_min and x2_max <= x1_max and y2_max <= y1_max

    def calculate_iou(self, box1, box2):
        # Extrair coordenadas e dimensões das caixas delimitadoras
        x1_min, y1_min, w1, h1, _ = box1
        x2_min, y2_min, w2, h2, _ = box2

        # Calcular x_max e y_max para cada caixa
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        # Calcular as coordenadas da interseção
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        # Calcular a área de interseção
        inter_area = max(inter_xmax - inter_xmin, 0) * max(inter_ymax - inter_ymin, 0)

        # Calcular a área das caixas delimitadoras
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Calcular IoU
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou


    def min_distance_between_boxes(self, box1, box2):
        # Extrair coordenadas e dimensões das caixas delimitadoras
        x1_min, y1_min, w1, h1, _ = box1
        x2_min, y2_min, w2, h2, _ = box2

        # Calcular x_max e y_max para cada caixa
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        # Defina os pontos das bordas dos bounding boxes
        points1 = np.array([
            [x1_min, y1_min],  # Top-left
            [x1_max, y1_min],  # Top-right
            [x1_min, y1_max],  # Bottom-left
            [x1_max, y1_max]  # Bottom-right
        ])

        points2 = np.array([
            [x2_min, y2_min],  # Top-left
            [x2_max, y2_min],  # Top-right
            [x2_min, y2_max],  # Bottom-left
            [x2_max, y2_max]  # Bottom-right
        ])

        # Calcule todas as distâncias entre os pontos dos dois bounding boxes
        distances = cdist(points1, points2, 'euclidean')

        # Encontre a menor distância
        min_distance = np.min(distances)
        return min_distance

    def partial_inscrit_bbox(self, box1, box2):
        # Extrair coordenadas e dimensões das caixas delimitadoras
        x1_min, y1_min, w1, h1, _ = box1
        x2_min, y2_min, w2, h2, _ = box2

        # Calcular x_max e y_max para cada caixa
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        # Calcular as áreas das caixas delimitadoras
        area_bbox = w1 * h1
        area_other_bbox = w2 * h2

        return area_bbox / float(area_other_bbox)

    def check_inside_bbox(self, small_objects):
        clusters = []
        while small_objects:
            base_obj = small_objects.pop(0)
            cluster = [base_obj]
            converged = False

            while not converged:
                to_remove = []

                for obj in small_objects:
                    bbox1 = base_obj[1:]
                    bbox2 = obj[1:]

                    inscrict_obj = self.is_inside(bbox1, bbox2)

                    if base_obj[0] == obj[0] and inscrict_obj:
                        cluster.append(obj)
                        to_remove.append(obj)
                if to_remove:
                    for obj in to_remove:
                        small_objects.remove(obj)
                else:
                    converged = True
            clusters.append(cluster)
        return clusters

    def cluster_small_objects(self, small_objects, iou_threshold, distance_threshold, area_threshold,
                              overlap_threshold, img_width, img_height, use_distance=True):
        clusters = []

        while small_objects:
            base_obj = small_objects.pop(0)
            cluster = [base_obj]
            converged = False

            while not converged:
                to_remove = []

                for obj in small_objects:
                    bbox1 = base_obj[1:]
                    bbox2 = obj[1:]

                    iou = self.calculate_iou(bbox1, bbox2)
                    union_area = self.calculate_union_area(bbox1, bbox2)
                    inscrict_obj = self.is_inside(bbox1, bbox2)
                    dist = self.min_distance_between_boxes(bbox1, bbox2)
                    partial_inscrict = self.partial_inscrit_bbox(bbox1, bbox2)

                    if (base_obj[0] == obj[0] and iou > iou_threshold and union_area < area_threshold) or \
                            (base_obj[0] == obj[0] and inscrict_obj and union_area < area_threshold) or \
                            (base_obj[0] == obj[0] and partial_inscrict > overlap_threshold and iou > 0.):

                        # print('iou', iou,'union_area', union_area, 'inscrict_obj', inscrict_obj, 'partial_inscrict', partial_inscrict, 'dist', dist)
                        cluster.append(obj)
                        to_remove.append(obj)

                    # elif use_distance and base_obj[0] == obj[0]:
                    elif use_distance and base_obj[0] == obj[0] and dist < distance_threshold:
                        # print('iou', iou, 'union_area', union_area, 'inscrict_obj', inscrict_obj, 'partial_inscrict',
                        #       partial_inscrict, 'dist', dist)
                        # print('iou', iou,'union_area', union_area, 'inscrict_obj', inscrict_obj, 'partial_inscrict', partial_inscrict, 'dist', dist)

                        if obj not in cluster:
                            cluster.append(obj)
                        if obj not in to_remove:
                            to_remove.append(obj)

                if to_remove:
                    for obj in to_remove:
                        small_objects.remove(obj)
                else:
                    converged = True

            clusters.append(cluster)

        return clusters

    def filter_small_objects(self, annotations, area_threshold):
        small_objects = []
        large_objects = []

        for ann in annotations:
            _, _, _, bw, bh, _ = ann
            area = bw * bh

            if area < area_threshold * 1.3:
                small_objects.append(ann)
            else:
                large_objects.append(ann)

        return small_objects, large_objects

    def sum_cluster_bboxes(self, clusters):
        clustered_annotations = []
        for cluster in clusters:
            if len(cluster) > 1:
                class_id = cluster[0][0]

                x_min = min(obj[1] for obj in cluster)
                y_min = min(obj[2] for obj in cluster)
                x_max = max(obj[1] + obj[3] for obj in cluster)
                y_max = max(obj[2] + obj[4] for obj in cluster)

                combined_width = x_max - x_min
                combined_height = y_max - y_min

                new_x = x_min
                new_y = y_min

                combined_confidence = max(obj[5] for obj in cluster)

                clustered_annotations.append((class_id, new_x, new_y, combined_width, combined_height, combined_confidence))
            else:
                clustered_annotations.append(cluster[0])
        return clustered_annotations

    def sort_bboxes_by_area_and_proximity(self, bboxes, reference_point=(0, 0)):
        def sort_key(bbox):
            x, y, xw, yh = bbox[1], bbox[2], bbox[3], bbox[4]
            area = xw * yh
            distance = math.sqrt((x - reference_point[0]) ** 2 + (y - reference_point[1]) ** 2)
            return (distance, -area)

        # Ordena a lista de bounding boxes pelo tamanho da área e pela proximidade
        sorted_bboxes = sorted(bboxes, key=sort_key)
        return sorted_bboxes

    def sort_bboxes_by_area(self, bboxes, ):
        # Ordena a lista de bounding boxes pelo tamanho da área (xw * yh)
        sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[3] * bbox[4], reverse=False)
        return sorted_bboxes

    def agg_clusters_objects(self, annotations, area_threshold, distance_threshold, iou_threshold,
                             overlap_threshold, img_height, img_width, use_distance=True):
        print(annotations)
        small_objects, large_objects = self.filter_small_objects(annotations, area_threshold)
        # print('small_objects', small_objects)
        # print('large_objects', large_objects)
        # small_objects = self.sort_bboxes_by_area(small_objects)
        small_objects = self.sort_bboxes_by_area_and_proximity(small_objects)
        clusters = self.cluster_small_objects(small_objects, iou_threshold, distance_threshold,
                                              area_threshold, overlap_threshold, img_width,
                                              img_height, use_distance)
        clustered_annotations = self.sum_cluster_bboxes(clusters)
        annotations_final = large_objects + clustered_annotations

        annotations_final = self.sort_bboxes_by_area_and_proximity(annotations_final)

        annotations_clusters = self.check_inside_bbox(annotations_final)
        annotations_final = self.sum_cluster_bboxes(annotations_clusters)
        # print('annotations_final', annotations_final)
        # converted_annotations = self.convert_to_yolov8_format(annotations=annotations_final,
        #                                                       img_width=img_width,
        #                                                       img_height=img_height)
        return annotations_final

    def convert_to_yolov8_format(self, annotations, img_width, img_height):
        yolov8_annotations = []
        for ann in annotations:
            class_id, x_min, y_min, x_max, y_max = ann

            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            yolov8_annotations.append((class_id, x_center, y_center, width, height))

        return yolov8_annotations

    async def evaluate(self, dir_path, classes_of_interest, confidence_threshold, merged_objects):
        for file in os.listdir(dir_path):
            if file.endswith(('.jpg', '.png')):
                img_path = os.path.join(dir_path, file)

                input_img = cv2.imread(img_path)
                img_height, img_width = input_img.shape[:2]

                input_img_pil = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

                detection_results, classification_results = await self.detector.async_run(input_img)

                pred_bboxes = [bbox[:4] for bbox in detection_results['bboxes']]
                pred_classes = [item for sublist in classification_results['pred'] for item in sublist]
                pred_confidences = [round(max(sublist), 2) for sublist in classification_results['softs']]
                # Filtra predições relevantes
                filtered_pred_bboxes = []
                filtered_pred_classes = []
                filtered_pred_confidences = []
                for bbox, cls, confidence in zip(pred_bboxes, pred_classes, pred_confidences):
                    if cls in classes_of_interest and confidence >= confidence_threshold:  # Considera apenas as classes relevantes
                        filtered_pred_bboxes.append(bbox)
                        filtered_pred_classes.append(cls)
                        filtered_pred_confidences.append(confidence)

                if merged_objects == True:
                    # Filtra predições relevantes
                    merged_pred_bboxes = []
                    merged_pred_classes = []
                    merged_pred_confidences = []

                    merger_objs = []
                    for bbox, cls, confidence in zip(filtered_pred_bboxes, filtered_pred_classes, filtered_pred_confidences):
                        if cls in [0, 1, 2]:  # Considera apenas as classes relevantes
                            center_x, center_y, width, height = bbox
                            x = center_x - (width / 2)
                            y = center_y - (height / 2)
                            xw = width
                            yh = height

                            merger_objs.append([cls, x, y, xw, yh, confidence])

                    if img_width > 1600:
                        area_threshold = 30000
                        distance_threshold = 50
                    else:
                        area_threshold = 20000
                        distance_threshold = 30

                    # Combina caixas com IoU > 0
                    # filtered_pred_bboxes = self.merge_boxes_same_class(filtered_pred_bboxes, filtered_pred_classes)

                    merger_objs = self.agg_clusters_objects(annotations=merger_objs, area_threshold=area_threshold,
                                                            distance_threshold=distance_threshold, iou_threshold=0.,
                                                            overlap_threshold=0.1, img_height=img_height,
                                                            img_width=img_width, use_distance=True)
                    for obj_bbox in merger_objs:
                        # print('obj_bbox', obj_bbox)
                        cls, x, y, xw, yh, confidence = obj_bbox
                        center_x = x + (xw / 2)
                        center_y = y + (yh / 2)
                        width = xw
                        height = yh

                        merged_pred_bboxes.append([int(center_x), int(center_y), int(width), int(height)])
                        merged_pred_classes.append(cls)
                        merged_pred_confidences.append(confidence)

                    img_with_pred_boxes, report = self.draw_bounding_boxes(input_img_pil.copy(), merged_pred_bboxes,
                                                                       merged_pred_classes, merged_pred_confidences)
                else:
                    img_with_pred_boxes, report = self.draw_bounding_boxes(input_img_pil.copy(), filtered_pred_bboxes,
                                                                       filtered_pred_classes, filtered_pred_confidences)
                # Capturar o shape da imagem (largura, altura)
                ww, hh = input_img_pil.size
                # Imprimir o shape da imagem
                # print(f"{hh}, {ww}")
                # print(f"{img_height},{ img_width }")

                img_with_pred_boxes.save(os.path.join(self.detector.output_dir, f"{file}_pred.jpg"))

                # print("Removing file {}".format(file))

                os.remove(dir_path + '/' + file)

                return report

def sum_categories(data):
    category_sums = {'HSIL': 0, 'LSIL': 0, 'NORMAL': 0}
    for entry in data:
        for category, value in entry:
            category_sums[category] += value
    
    return category_sums

async def predict_image_local(path: str):
    detection_model = YOLO('model/anomaly.pt')
    classification_model = YOLO('model/classification.pt')
    img_size = 640 # Tamanho do input do modelo de detecção 640x640
    img_size_pos = 320 # Tamanho do input do modelo de classificação 320x320
    classes_of_interest = [0, 1, 2] # Classes de interesse a serem contabilizadas (0: HSIL, 1:LSIL, 2:NORMAL)
    confidence_threshold = 0.8 # Threshold de confiança da saída do modelo de classificação para reduzir falsos positivos
    batch_size = 16 # Ajustar conforme recurso computacional: Inferência em batch dos crops que chegam no modelo de classificação
    merged_objects = True # Utiliza a função 'agg_clusters_objects' para fazer o merge nas bboxes
    iou = 0.4 # Threshold de iou usando no Non max supression do localizador. Quanto menor o threshold, menos bounding boxes localizadas
    evaluator = ObjectDetectionEvaluator(detection_model, classification_model, img_size, img_size_pos, batch_size, iou)
    
    return await evaluator.evaluate(path, classes_of_interest, confidence_threshold, merged_objects)

def input_data(hsil,lsil,normal,hsil_rate,lsil_rate,normal_rate,real_result,predicted_result,time):
    try:
        experiments = pd.read_csv('experiments.csv')
        new_row = pd.DataFrame({
                "hsil": [hsil],
                "lsil": [lsil],
                "normal": [normal],
                "hsil_rate": [hsil_rate],
                "lsil_rate": [lsil_rate],
                "normal_rate": [normal_rate],
                "real_result": [real_result],
                "predicted_result": [predicted_result],
                "time": [time]
            })
        experiments = pd.concat([experiments, new_row], ignore_index=False)
        experiments.to_csv('experiments.csv', index=False)
    except:
        new_row = pd.DataFrame({
                "hsil": [hsil],
                "lsil": [lsil],
                "normal": [normal],
                "hsil_rate": [hsil_rate],
                "lsil_rate": [lsil_rate],
                "normal_rate": [normal_rate],
                "real_result": [real_result],
                "predicted_result": [predicted_result],
                "time": [time]
            })
        new_row.to_csv('experiments.csv', index=False)

def calculate_rates(hsil, lsil, normal, real_result, time):
    hsil_rate = 0
    lsil_rate = 0
    if hsil + lsil != 0:
        hsil_rate = hsil / (hsil+lsil)
        lsil_rate = lsil / (hsil+lsil)
    normal_rate = normal / (hsil + lsil + normal)
        
    print(f'HSIL:{hsil} \n LSIL: {lsil} \n NORMAL: {normal}\n')
    print(f'HSIL_RATE:{hsil_rate} \n LSIL_RATE: {lsil_rate} \n NORMAL_RATE: {normal_rate}')

    if hsil_rate >= 0.40:
        input_data(hsil, lsil, normal, hsil_rate, lsil_rate, normal_rate, real_result, 'HSIL', time)
    elif lsil_rate >= 0.35:
        input_data(hsil, lsil, normal, hsil_rate, lsil_rate, normal_rate, real_result, 'LSIL', time)
    else:
        input_data(hsil, lsil, normal, hsil_rate, lsil_rate, normal_rate, real_result, 'NEGATIVO', time)


def analyze_classes_from_prediction(report, initial_time: datetime | None = None, real_result:str | None = None, final_report: bool | None = False):
    print('\n\n\n========================REPORT========================\n', report, '\n\n\n')
    if final_report and initial_time and real_result:
        end_time_analysis = datetime.now()
        classes_amount = sum_categories(report)
        overall_time = (end_time_analysis-initial_time).total_seconds()
        hsil = classes_amount.get('HSIL')
        lsil = classes_amount.get('LSIL')
        normal = classes_amount.get('NORMAL')
        calculate_rates(hsil, lsil, normal, real_result=real_result, time=overall_time)


def manage_experiment(uploaded_files):
    real_result = ''
    final_report = []
    initial_time = datetime.now()
    for index, uploaded_file in enumerate(uploaded_files):
        report = asyncio.run(predict_image_local(uploaded_file))
        
        if 'HSIL' in uploaded_file: real_result = 'HSIL'
        elif 'LSIL' in uploaded_file: real_result = 'LSIL'
        elif 'NEGATIVO' in uploaded_file: real_result = 'NEGATIVO'
        else: real_result = 'NOT_IDENTIFIED'        
        analyze_classes_from_prediction(report)
        final_report.append(report)

    analyze_classes_from_prediction(final_report, initial_time=initial_time, real_result=real_result, final_report=True)


if __name__ == "__main__":
    import os
    uploaded_files = []
    for dirpath, dirnames, filenames in os.walk(config.getPathLocal()):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            uploaded_files.append(dirpath)
        if len(uploaded_files) != 0:
            manage_experiment(uploaded_files=uploaded_files)
            uploaded_files = []

    for i in range(0,5):
        for dirpath, dirnames, filenames in os.walk(config.getPathLocal()):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                uploaded_files.append(dirpath)
            if len(uploaded_files) != 0:
                manage_experiment(uploaded_files=uploaded_files)
                uploaded_files = []


async def predict_image():
    detection_model = YOLO('model/anomaly.pt')
    classification_model = YOLO('model/classification.pt')
    img_size = 640 # Tamanho do input do modelo de detecção 640x640
    img_size_pos = 320 # Tamanho do input do modelo de classificação 320x320
    classes_of_interest = [0, 1, 2] # Classes de interesse a serem contabilizadas (0: HSIL, 1:LSIL, 2:NORMAL)
    confidence_threshold = 0.9 # Threshold de confiança da saída do modelo de classificação para reduzir falsos positivos
    batch_size = 8 # Ajustar conforme recurso computacional: Inferência em batch dos crops que chegam no modelo de classificação
    merged_objects = True # Utiliza a função 'agg_clusters_objects' para fazer o merge nas bboxes
    iou = 0.4 # Threshold de iou usando no Non max supression do localizador. Quanto menor o threshold, menos bounding boxes localizadas
    evaluator = ObjectDetectionEvaluator(detection_model, classification_model, img_size, img_size_pos, batch_size, iou)
    dir_path = config.getPath()
    return await evaluator.evaluate(dir_path, classes_of_interest, confidence_threshold, merged_objects)
