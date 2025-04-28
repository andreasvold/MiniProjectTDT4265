import os
from PIL import Image

def check_image_sizes(image_dir):
    image_sizes = []

    for image in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image)
        img = Image.open(img_path)
        image_sizes.append(img.size)  
    
    min_size = min(image_sizes, key=lambda x: (x[0], x[1]))
    max_size = max(image_sizes, key=lambda x: (x[0], x[1]))
    avg_size = (sum(size[0] for size in image_sizes) / len(image_sizes), 
                sum(size[1] for size in image_sizes) / len(image_sizes))

    print(f"Min size: {min_size}")
    print(f"Max size: {max_size}")
    print(f"Average size: {avg_size}")



print("\n Training data \n")
check_image_sizes("/Users/andre/projects/MiniProjectTDT4265/datasets/rgb/images/train")
print("\n Validation data \n")
check_image_sizes("/Users/andre/projects/MiniProjectTDT4265/datasets/rgb/images/valid")