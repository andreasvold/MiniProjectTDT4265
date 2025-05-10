import json
import os


input_folder = "datasets/TTPLA_json/val/ann/"    
output_folder = "datasets/TTPLA_YOLO/labels/valid"   


if os.listdir(output_folder):
    print(f"Output folder '{output_folder}' is not empty.")
else:
    print(f"Output folder '{output_folder}' is empty and ready.")
    os.makedirs(output_folder, exist_ok=True)

    
    def polygon_to_bbox(points):
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return x_min, y_min, x_max, y_max

    
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(input_folder, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)

        image_width = data["size"]["width"]
        image_height = data["size"]["height"]

        lines = []
        for obj in data.get("objects", []):
            if obj.get("classTitle") != "cable":
                continue  

            points = obj["points"]["exterior"]
            x_min, y_min, x_max, y_max = polygon_to_bbox(points)

            
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(yolo_line)

        
        output_txt_path = os.path.join(output_folder, file_name.replace(".json", ".txt"))
        with open(output_txt_path, "w") as f:
            f.write("\n".join(lines))
