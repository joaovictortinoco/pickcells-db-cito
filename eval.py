import torch
import os
import math
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import asyncio
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

class_mapping = {}
class YoloSensitiveDetector():
    def __init__(self, detection_model, classification_model, img_size, img_size_pos, batch_size=8):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.img_size = img_size
        self.img_size_pos = img_size_pos
        self.batch_size = batch_size
        self.post_processor = YoloPosProcessing()
        self.output_dir = './runs/classification/predict'
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
        detection_results = self.detection_model.predict(source=input_img, save=True, save_conf=False, verbose=False, imgsz=self.img_size, conf=0.2)
        if len(detection_results) == 0 or len(detection_results[0].boxes) == 0:
            return {}, {}

        dict_results = {"names": detection_results[0].names, "bboxes": [], "classes": [], "confs": []}
        dict_results["bboxes"] = detection_results[0].boxes.xywh.int().tolist()
        dict_results["classes"] = detection_results[0].boxes.cls.tolist()
        dict_results["confs"] = detection_results[0].boxes.conf.tolist()

        normal_class_index = 2  # Ajuste conforme necessário

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
                if result.probs.top1 != normal_class_index:
                    dict_results_pos['names'] = result.names
                    dict_results_pos['softs'].append(result.probs.data.tolist())
                    dict_results_pos['pred'].append([result.probs.top1])
                else:
                    indices_to_remove.append(batch_start + j)

        for index in sorted(indices_to_remove, reverse=True):
            dict_results['bboxes'].pop(index)
            dict_results['classes'].pop(index)
            dict_results['confs'].pop(index)

        annotated_img = self.post_processor.draw_bounding_boxes(input_img, dict_results, dict_results_pos)
        annotated_img.save(os.path.join(self.output_dir, "annotated_image.jpg"))

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

    def draw_bounding_boxes(self, image, detection_results, classification_results):
        global class_mapping
        class_mapping = classification_results['names']
        draw = ImageDraw.Draw(image)
        for bbox, class_id in zip(detection_results["bboxes"], classification_results['pred']):
            center_x, center_y, width, height = bbox
            left = center_x - (width / 2)
            top = center_y - (height / 2)
            right = center_x + (width / 2)
            bottom = center_y + (width / 2)
            class_name = classification_results['names'][class_id[0]]
            color = self.class_colors.get(class_name, 'white')
            draw.rectangle([left, top, right, bottom], outline=color, width=2)
            draw.text((left, top), class_name, fill=color)
            
        return image

class ObjectDetectionEvaluator:
    def __init__(self, detection_model, classification_model, img_size, img_size_pos, batch_size=1):
        self.detector = YoloSensitiveDetector(detection_model, classification_model, img_size, img_size_pos, batch_size)
        self.class_colors = {
            'HSIL': 'red',
            'LSIL': 'orange',
            'NORMAL': 'green'
        }
        self.class_mapping = {0: 'HSIL', 1: 'LSIL', 2: 'NORMAL', 3: 'BACKGROUND'}

    def load_yolo_annotation(self, txt_file, img_width, img_height):
        bboxes = []
        classes = []
        normal_class_index = 2
        
        with open(txt_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                class_id = int(values[0])
                if class_id == normal_class_index:
                    continue  # Ignore class 2
                
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

    def draw_bounding_boxes(self, image, bboxes, classes):
        draw = ImageDraw.Draw(image)    
        global class_mapping   

        for bbox, cls in zip(bboxes, classes):
            center_x, center_y, width, height = bbox
            left = center_x - (width / 2)
            top = center_y - (height / 2)
            right = center_x + (width / 2)
            bottom = center_y + (width / 2)
            class_name = self.class_mapping[cls]
            color = self.class_colors.get(class_name, 'white')
            draw.rectangle([left, top, right, bottom], outline=color, width=2)
            draw.text((left, top), class_name, fill=color)
        return image

    def conv_to_polygon(self, box):
        x, y, w, h = box
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def intersect_box(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        pol_box1 = self.conv_to_polygon([x1, y1, w1, h1])
        pol_box2 = self.conv_to_polygon([x2, y2, w2, h2])

        pol_area1 = Polygon(pol_box1)
        pol_area2 = Polygon(pol_box2)

        intersection_area = pol_area1.intersection(pol_area2).area
        union_area = pol_area1.union(pol_area2).area

        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def merge_bounding_boxes(self, bboxes, iou_threshold):
        if len(bboxes) == 0:
            return []

        merged_bboxes = []
        bboxes = np.array(bboxes)

        while len(bboxes) > 0:
            current_bbox = bboxes[0]
            del_indices = [0]
            
            i = 1
            while i < len(bboxes):
                bbox = bboxes[i]
                x_min = min(current_bbox[0], bbox[0])
                y_min = min(current_bbox[1], bbox[1])
                x_max = max(current_bbox[0] + current_bbox[2], bbox[0] + bbox[2])
                y_max = max(current_bbox[1] + current_bbox[3], bbox[1] + bbox[3])
                
                # Calcular a interseção e IoU
                intersect_x1 = max(current_bbox[0], bbox[0])
                intersect_y1 = max(current_bbox[1], bbox[1])
                intersect_x2 = min(current_bbox[0] + current_bbox[2], bbox[0] + bbox[2])
                intersect_y2 = min(current_bbox[1] + current_bbox[3], bbox[1] + bbox[3])
                
                intersect_width = max(0, intersect_x2 - intersect_x1)
                intersect_height = max(0, intersect_y2 - intersect_y1)
                
                intersection_area = intersect_width * intersect_height
                area1 = current_bbox[2] * current_bbox[3]
                area2 = bbox[2] * bbox[3]
                iou = intersection_area / float(area1 + area2 - intersection_area)
                
                # Se a IoU for maior que o limiar, mesclar as bounding boxes
                if iou > iou_threshold:
                    x_min = min(current_bbox[0], bbox[0])
                    y_min = min(current_bbox[1], bbox[1])
                    x_max = max(current_bbox[0] + current_bbox[2], bbox[0] + bbox[2])
                    y_max = max(current_bbox[1] + current_bbox[3], bbox[1] + bbox[3])
                    current_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    del_indices.append(i)
                else:
                    i += 1
            
            merged_bboxes.append(current_bbox)
            bboxes = np.delete(bboxes, del_indices, axis=0)

        return merged_bboxes

    def non_max_suppression(self, bboxes, threshold):
        if len(bboxes) == 0:
            return []

        bboxes = np.array(bboxes)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 0] + bboxes[:, 2]
        y2 = bboxes[:, 1] + bboxes[:, 3]
        areas = bboxes[:, 2] * bboxes[:, 3]

        keep = []
        indices = np.argsort(y2)

        while len(indices) > 0:
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[indices[:last]])
            yy1 = np.maximum(y1[i], y1[indices[:last]])
            xx2 = np.minimum(x2[i], x2[indices[:last]])
            yy2 = np.minimum(y2[i], y2[indices[:last]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[indices[:last]] - intersection)

            indices = np.delete(indices, np.concatenate(([last], np.where(iou > threshold)[0])))

        return bboxes[keep]
    
    def intersection_over_union(self, box1, box2):
        # Extrair coordenadas dos centros e dimensões das bounding boxes
        x1_box1, y1_box1, w1_box1, h1_box1 = box1
        x1_box2, y1_box2, w2_box2, h2_box2 = box2
        
        # Calcular coordenadas dos cantos das bounding boxes
        x1_box1_left = x1_box1 - w1_box1 / 2
        y1_box1_top = y1_box1 - h1_box1 / 2
        x1_box1_right = x1_box1 + w1_box1 / 2
        y1_box1_bottom = y1_box1 + h1_box1 / 2
        
        x1_box2_left = x1_box2 - w2_box2 / 2
        y1_box2_top = y1_box2 - h2_box2 / 2
        x1_box2_right = x1_box2 + w2_box2 / 2
        y1_box2_bottom = y1_box2 + h2_box2 / 2
        
        # Encontrar intersecção dos limites das bounding boxes
        x_left = max(x1_box1_left, x1_box2_left)
        y_top = max(y1_box1_top, y1_box2_top)
        x_right = min(x1_box1_right, x1_box2_right)
        y_bottom = min(y1_box1_bottom, y1_box2_bottom)
        
        # Calcular área da intersecção
        if x_right <= x_left or y_bottom <= y_top:
            intersection_area = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calcular áreas das bounding boxes individuais
        area_box1 = w1_box1 * h1_box1
        area_box2 = w2_box2 * h2_box2
        
        # Calcular área da união
        union_area = area_box1 + area_box2 - intersection_area
        
        # Calcular IoU (Intersection over Union)
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou


    def merge_boxes_same_class(self, bboxes, classes):
        merged_bboxes = []
        used = [False] * len(bboxes)
        
        for i in range(len(bboxes)):
            if used[i]:
                continue
            
            current_bbox = bboxes[i]
            current_class = classes[i]
            x_center, y_center, width, height = current_bbox
            
            # Calcular limites da bounding box atual a partir do centro
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            for j in range(i + 1, len(bboxes)):
                if used[j]:
                    continue
                
                bbox_to_compare = bboxes[j]
                class_to_compare = classes[j]
                if current_class != class_to_compare:
                    continue
                
                x1_comp, y1_comp, w_comp, h_comp = bbox_to_compare
                x2_comp = x1_comp + (w_comp / 2)
                y2_comp = y1_comp + (h_comp / 2)
                
                # Verificar a interseção usando a função intersection_over_union modificada
                if self.intersection_over_union(current_bbox, bbox_to_compare) > 0:
                    # Atualizar a bounding box atual para incluir a outra
                    x1 = min(x1, x1_comp)
                    y1 = min(y1, y1_comp)
                    x2 = max(x2, x2_comp)
                    y2 = max(y2, y2_comp)
                    
                    # Marcar bbox_to_compare como usada
                    used[j] = True
            
            # Calcular novas coordenadas do centro e dimensões da bounding box agrupada
            new_x_center = (x1 + x2) / 2
            new_y_center = (y1 + y2) / 2
            new_width = x2 - x1
            new_height = y2 - y1
            
            # Adicionar a bounding box agrupada à lista final
            merged_bboxes.append([new_x_center, new_y_center, new_width, new_height])
        
        return merged_bboxes
    
    async def evaluate(self, dir_path):
        true_labels = []
        predicted_labels = []

        for file in os.listdir(dir_path):
            if file.endswith('.jpg'):
                img_path = os.path.join(dir_path, file)
                annotation_path = os.path.splitext(img_path)[0] + '.txt'

                input_img = cv2.imread(img_path)
                img_height, img_width, _ = input_img.shape

                input_img_pil = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
                # true_bboxes, true_classes = self.load_yolo_annotation(annotation_path, img_width, img_height)
                detection_results, classification_results = await self.detector.async_run(input_img)
                if 'bboxes' not in detection_results or len(detection_results['bboxes']) == 0:
                    # print(true_classes)
                    # true_labels.extend(true_classes)
                    # predicted_labels.extend([3] * len(true_classes))  # Adiciona BACKGROUND para predições
                    continue

                pred_bboxes = [bbox[:4] for bbox in detection_results['bboxes']]
                pred_classes = [item for sublist in classification_results['pred'] for item in sublist]
                
                # Filtra predições relevantes
                filtered_pred_bboxes = []
                filtered_pred_classes = []
                for bbox, cls in zip(pred_bboxes, pred_classes):
                    if cls in [0, 1, 3]:  # Considera apenas as classes relevantes
                        filtered_pred_bboxes.append(bbox)
                        filtered_pred_classes.append(cls)

                # Combina caixas com IoU > 0
                filtered_pred_bboxes = self.merge_boxes_same_class(filtered_pred_bboxes, filtered_pred_classes)

                # Associa cada caixa predita com a classe correta, usando IoU > 0.5
                used_true_indices = set()
                
                # for pred_bbox, pred_cls in zip(filtered_pred_bboxes, filtered_pred_classes):
                #     max_iou = -1
                #     max_true_idx = -1
                    
                    # for true_idx, true_bbox in enumerate(true_bboxes):
                    #     iou = self.intersection_over_union(pred_bbox, true_bbox)
                        
                    #     if iou > max_iou:
                    #         max_iou = iou
                    #         max_true_idx = true_idx
                    
                    # if max_iou >= 0.01 and max_true_idx not in used_true_indices:
                    #     true_labels.append(true_classes[max_true_idx])
                    #     predicted_labels.append(pred_cls)
                    #     used_true_indices.add(max_true_idx)
                    # elif max_iou < 0.5:
                    #     predicted_labels.append(pred_cls)
                    #     true_labels.append(3)  # Considera como BACKGROUND
                
                # Adiciona classes verdadeiras restantes como falsos negativos
                # for true_idx, true_class in enumerate(true_classes):
                #     if true_idx not in used_true_indices:
                #         true_labels.append(true_class)
                #         predicted_labels.append(3)  # Considera como BACKGROUND

                # img_with_true_boxes = self.draw_bounding_boxes(input_img_pil.copy(), true_bboxes, true_classes)
                img_with_pred_boxes = self.draw_bounding_boxes(input_img_pil.copy(), filtered_pred_bboxes, filtered_pred_classes)

                # img_with_true_boxes.save(os.path.join(self.detector.output_dir, f"{os.path.splitext(file)[0]}_true.jpg"))
                img_with_pred_boxes.save(os.path.join(self.detector.output_dir, f"{os.path.splitext(file)[0]}_pred.jpg"))

        # Converte para numpy arrays para usar classification_report e confusion_matrix
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        # Gera o classification report
        report = classification_report(true_labels, predicted_labels, labels=[0, 1], target_names=['HSIL', 'LSIL'])
        print("Classification Report:")
        print(report)
        
        # Gera a matriz de confusão
        cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 3])
        
        # # Plota a matriz de confusão
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HSIL', 'LSIL', 'BACKGROUND'], yticklabels=['HSIL', 'LSIL', 'BACKGROUND'])
        # plt.xlabel('Predicted labels')
        # plt.ylabel('True labels')
        # plt.title('Confusion Matrix')
        # plt.show()
        
        return report, cm

if __name__ == "__main__":
    async def main():
        detection_model = YOLO('model/anomaly_y8.pt')
        classification_model = YOLO('model/classification_od_new.pt')
        img_size = 640
        img_size_pos = 320
        evaluator = ObjectDetectionEvaluator(detection_model, classification_model, img_size, img_size_pos)
        dir_path = 'model/test/test'
        report, cm = await evaluator.evaluate(dir_path)

    asyncio.run(main())

async def predict_image():
    detection_model = YOLO('model/anomaly_y8.pt')
    classification_model = YOLO('model/classification_od_new.pt')
    img_size = 640
    img_size_pos = 320
    evaluator = ObjectDetectionEvaluator(detection_model, classification_model, img_size, img_size_pos)
    dir_path = 'model/test/test'
    report, cm = await evaluator.evaluate(dir_path)

