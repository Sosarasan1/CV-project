# preprocess_data_v2.py (with train/valid/test split)

import os
import xml.etree.ElementTree as ET
import random
import shutil

# ==========================================================================================
# --- 1. CONFIGURATION: YOU ONLY NEED TO EDIT THIS SECTION ---
# ==========================================================================================

XML_DIR = 'Annotations' 
IMG_DIR = 'JPEGImages' 
OUTPUT_DIR = 'helmet_dataset_final'

# Define the split ratios. These should sum to 1.0
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1 # This is calculated automatically, but good to state

if TRAIN_RATIO + VALID_RATIO + TEST_RATIO > 1.0:
    print("ERROR: Split ratios sum to more than 1. Please adjust.")
    exit()

CLASS_MAPPING = {
    'person': 0,
    'hat': 1,
    'dog': 2
}


# ==========================================================================================
# --- 2. THE LOGIC: YOU DON'T NEED TO EDIT BELOW THIS LINE ---
# ==========================================================================================

def convert_voc_to_yolo(xml_file_path, class_mapping):
    """Converts a single Pascal VOC XML file to a list of YOLO formatted strings."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip().lower()

            print(f"Found class: '{class_name}' in {xml_file_path}")

            if class_name not in class_mapping:
                continue
            class_id = class_mapping[class_name]
            bndbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = [float(bndbox.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            x_center_norm, y_center_norm, width_norm, height_norm = [
                x_center / img_width, y_center / img_height, width / img_width, height / img_height
            ]
            
            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        return yolo_lines
    except Exception as e:
        print(f"Error processing {xml_file_path}: {e}")
        return None

def process_files(file_list, dataset_name, output_dir, img_dir, xml_dir, class_mapping):
    """Processes a list of files for a given dataset split (train/valid/test)."""
    print(f"\n--- Processing {dataset_name.upper()} set ({len(file_list)} files) ---")
    
    # Create subdirectories for the current set
    os.makedirs(os.path.join(output_dir, dataset_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, dataset_name, 'labels'), exist_ok=True)

    for xml_filename in file_list:
        base_filename = xml_filename.replace('.xml', '')
        xml_path = os.path.join(xml_dir, xml_filename)
        
        source_img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(img_dir, base_filename + ext)
            if os.path.exists(potential_path):
                source_img_path = potential_path
                break
        
        if source_img_path is None:
            print(f"Warning: Image for {xml_filename} not found. Skipping.")
            continue

        yolo_data = convert_voc_to_yolo(xml_path, class_mapping)

        if yolo_data is not None and yolo_data:
            dest_label_path = os.path.join(output_dir, dataset_name, 'labels', base_filename + '.txt')
            dest_img_path = os.path.join(output_dir, dataset_name, 'images', os.path.basename(source_img_path))
            with open(dest_label_path, 'w') as f:
                f.write('\n'.join(yolo_data))
            shutil.copy(source_img_path, dest_img_path)

def main():
    print("--- Starting Dataset Preprocessing with Train/Valid/Test Split ---")

    # 1. Get list of all XML files
    if not os.path.isdir(XML_DIR):
        print(f"ERROR: XML directory not found at '{XML_DIR}'")
        return
        
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} XML files to process.")
    random.shuffle(xml_files)

    # 2. Calculate split indices
    total_files = len(xml_files)
    train_end = int(total_files * TRAIN_RATIO)
    valid_end = train_end + int(total_files * VALID_RATIO)

    # 3. Split the file list
    train_files = xml_files[:train_end]
    valid_files = xml_files[train_end:valid_end]
    test_files = xml_files[valid_end:]

    print(f"Splitting dataset: {len(train_files)} train, {len(valid_files)} validation, {len(test_files)} test.")

    # 4. Process each set
    process_files(train_files, 'train', OUTPUT_DIR, IMG_DIR, XML_DIR, CLASS_MAPPING)
    process_files(valid_files, 'valid', OUTPUT_DIR, IMG_DIR, XML_DIR, CLASS_MAPPING)
    process_files(test_files, 'test', OUTPUT_DIR, IMG_DIR, XML_DIR, CLASS_MAPPING)

    print("\n--- Preprocessing Complete! ---")
    print(f"Your dataset is ready in the '{OUTPUT_DIR}' folder with train, valid, and test splits.")


if __name__ == '__main__':
    main()