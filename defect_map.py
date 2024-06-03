
import cv2
import numpy as np
import os


def yolo_to_bbox(yolo_annotation, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, yolo_annotation)
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2


def create_defect_map(image_shape, yolo_annotations):
    defect_map = np.zeros(image_shape[:2], dtype=np.uint8)
    img_height, img_width = image_shape[:2]

    for annotation in yolo_annotations:
        x1, y1, x2, y2 = yolo_to_bbox(annotation, img_width, img_height)
        cv2.rectangle(defect_map, (x1, y1), (x2, y2), 255, -1)

    return defect_map


def load_yolo_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            annotations.append(line.strip().split())
    return annotations


def process_dataset(image_dir, annotation_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        annotation_path = os.path.join(annotation_dir, filename.replace('.png', '.txt'))

        if os.path.exists(annotation_path):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error reading image {image_path}")
                    continue

                annotations = load_yolo_annotations(annotation_path)
                defect_map = create_defect_map(image.shape, annotations)

                output_path = os.path.join(output_dir, filename.replace('.png', '_defect.png'))
                cv2.imwrite(output_path, defect_map)
                print(f"Processed {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# 示例用法
base_dir = 'datasets'
image_dir = os.path.join(base_dir, 'images')
annotation_dir = os.path.join(base_dir, 'annotation')
output_dir = os.path.join(base_dir, 'defect_map')

process_dataset(image_dir, annotation_dir, output_dir)