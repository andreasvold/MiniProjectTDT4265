import os
import numpy as np
from PIL import Image


def calculate_box_statistics(image_dir, label_dir,is_ttpla):
    box_areas = []
    objects_per_image = []
    overlap_count = 0
    total_images = 0

    for label_name in os.listdir(label_dir):
        if label_name.endswith('.txt'):
            if is_ttpla:
                img_name = label_name.replace('.txt', '.jpg') 
            else:
                img_name = label_name.replace('.txt', '.png')
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, label_name)

            if not os.path.isfile(img_path):
                print(f"Warning: Image file {img_path} does not exist!")
                continue

            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue

            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        boxes.append([x_center * img_width, y_center * img_height, width * img_width, height * img_height])

            if len(boxes) == 0:
                print(f"No bounding boxes in {label_name}")
                continue

            # Calculate average bounding box area
            areas = [w * h for _, _, w, h in boxes]
            box_areas.extend(areas)
            objects_per_image.append(len(boxes))

            # Calculate overlap 
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    box1 = boxes[i]
                    box2 = boxes[j]
                    # Simple IoU check
                    if iou(box1, box2) > 0.5:
                        overlap_count += 1

            total_images += 1

    if total_images == 0:
        print("No valid images or labels processed.")
        return

    # Compute and print statistics
    avg_box_area = np.mean(box_areas) if box_areas else float('nan')
    avg_objects_per_image = np.mean(objects_per_image) if objects_per_image else float('nan')
    avg_overlap = overlap_count / total_images if total_images else float('nan')

    print(f"Average Bounding Box Area: {avg_box_area}")
    print(f"Average Objects per Image: {avg_objects_per_image}")
    print(f"Average Overlap (IoU > 0.5): {avg_overlap}")

def iou(box1, box2):
    # Compute IoU for two boxes in format [x_center, y_center, width, height]
    x1_min, y1_min, x1_max, y1_max = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min, x2_max, y2_max = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        inter_area = 0.0

    # Compute union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


