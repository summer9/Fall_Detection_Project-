import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
from pathlib import Path

# Class mapping for your fall detection dataset
class_names = ['fall', 'standing', 'sitting']


def create_voc_xml(image_path, width, height, objects, output_xml_path):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"  # Assuming RGB images

    for obj in objects:
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = class_names[obj['class_id']]
        ET.SubElement(object_elem, "pose").text = "Unspecified"
        ET.SubElement(object_elem, "truncated").text = "0"
        ET.SubElement(object_elem, "difficult").text = "0"
        bndbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(obj['xmin']))
        ET.SubElement(bndbox, "ymin").text = str(int(obj['ymin']))
        ET.SubElement(bndbox, "xmax").text = str(int(obj['xmax']))
        ET.SubElement(bndbox, "ymax").text = str(int(obj['ymax']))

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree = ET.ElementTree(annotation)
    with open(output_xml_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)


def convert_yolo_to_voc(images_dir, labels_dir, output_dir, split_name='train'):
    output_images_dir = os.path.join(output_dir, split_name, "img")  # ORBS naming
    output_annotations_dir = os.path.join(output_dir, split_name, "xml")  # ORBS naming
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    try:
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort()
    except FileNotFoundError as e:
        print(f"Error: Directory not found: {images_dir}")
        raise

    for img_file in image_files:
        src_img_path = os.path.join(images_dir, img_file)
        dst_img_path = os.path.join(output_images_dir, img_file)
        try:
            shutil.copyfile(src_img_path, dst_img_path)  # Avoid permission issues
        except PermissionError as e:
            print(f"Error copying {src_img_path} to {dst_img_path}: {e}")
            continue

        try:
            with Image.open(src_img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error loading image {src_img_path}: {e}")
            continue

        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        objects = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Skipping invalid label in {label_path}: {line.strip()}")
                        continue
                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x_min = (cx - w / 2) * width
                        y_min = (cy - h / 2) * height
                        x_max = (cx + w / 2) * width
                        y_max = (cy + h / 2) * height
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(width, x_max)
                        y_max = min(height, y_max)
                        objects.append({
                            'class_id': class_id,
                            'xmin': x_min,
                            'ymin': y_min,
                            'xmax': x_max,
                            'ymax': y_max
                        })
                    except ValueError as e:
                        print(f"Error parsing label in {label_path}: {line.strip()}, {e}")
                        continue

        output_xml_path = os.path.join(output_annotations_dir, os.path.splitext(img_file)[0] + '.xml')
        create_voc_xml(src_img_path, width, height, objects, output_xml_path)

    print(f"Converted {split_name}: {len(image_files)} images, saved to {output_dir}/{split_name}")


if __name__ == "__main__":
    base_dir = "/media/public_data/temp/Phuong/fall_dataset/split"
    output_base = "/media/public_data/temp/Phuong/fall_dataset/voc"

    splits = {
        "train": {
            "images_dir": os.path.join(base_dir, "train/images"),
            "labels_dir": os.path.join(base_dir, "train/labels")
        },
        "valid": {
            "images_dir": os.path.join(base_dir, "valid/images"),
            "labels_dir": os.path.join(base_dir, "valid/labels")
        },
        "test": {
            "images_dir": os.path.join(base_dir, "test/images"),
            "labels_dir": os.path.join(base_dir, "test/labels")
        }
    }

    for split_name, paths in splits.items():
        convert_yolo_to_voc(
            images_dir=paths["images_dir"],
            labels_dir=paths["labels_dir"],
            output_dir=output_base,
            split_name=split_name
        )

    print(f"VOC dataset created at {output_base}")