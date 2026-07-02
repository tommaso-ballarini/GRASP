import os
import random
import argparse
import json
import sys
sys.path.append("src/utils")
from defined import yollava_category_dict, myvlm_category_dict

def assign_yollava_images(subdir_path, image_files, concept, seed):
    """
    For YoLLaVA:
      - Select one random image from the concept folder for training.
      - Use the remaining images from the folder for validation.
      - For testing, load images from the corresponding test directory.
    """
    random.seed(seed)
    # Train: pick one random image.
    random_image = random.choice(image_files)
    train = [os.path.join(subdir_path, random_image)]
    
    # Validation: the rest of the images in the subdirectory.
    val = [os.path.join(subdir_path, f) for f in image_files if f != random_image]
    
    # Test: images from the test directory.
    test_dir = os.path.join("data/yollava-data/test", concept)
    if os.path.isdir(test_dir):
        test = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    else:
        test = []
    
    return {"train": train, "val": val, "test": test}

def assign_myvlm_images(subdir_path, image_files, seed):
    """
    For MyVLM:
      - Randomly sample up to 5 images.
      - Use one random image from the sample for training.
      - Use the rest of the sample for validation.
      - Treat images not in the sample as test images.
    """
    random.seed(seed)
    # sampled_images = random.sample(image_files, min(5, len(image_files)))
    random_image = random.choice(image_files)
    test_images = list(set(image_files) - set(random_image))
    train = [os.path.join(subdir_path, random_image)]
    # val = [os.path.join(subdir_path, f) for f in sampled_images if f != random_image]
    test = [os.path.join(subdir_path, f) for f in test_images]
    
    return {"train": train, "val": [], "test": test}

def create_data_split(dataset_name, seed):
    category = 'all'
    image_paths_dict = {category: {}}
    
    if dataset_name == 'YoLLaVA':
        image_path_json = f"data/yollava-data/train_test_val_seed_{seed}.json"
        root = "data/yollava-data/train"
        for _, concept_list in yollava_category_dict.items():
            for concept in concept_list:
                image_paths_dict[category][concept] = {}
                subdir_path = os.path.join(root, concept)
                if os.path.isdir(subdir_path):
                    image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png'))]
                    if image_files:
                        image_paths_dict[category][concept] = assign_yollava_images(subdir_path, image_files, concept, seed)
        
        print(f'Saving data @ {image_path_json}')                
        with open(image_path_json, 'w') as f:
            json.dump(image_paths_dict, f, indent=2)
            
    elif dataset_name == 'MyVLM':
        image_path_json = f"data/myvlm-data/train_test_val_seed_{seed}.json"
        root = 'data/myvlm-data/'
        for _, concept_list in myvlm_category_dict.items():
            for concept in concept_list:
                image_paths_dict[category][concept] = {}
                subdir_path = os.path.join(root, concept)
                if os.path.isdir(subdir_path):
                    image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png'))]
                    if image_files:
                        image_paths_dict[category][concept] = assign_myvlm_images(subdir_path, image_files, seed)

        print(f'Saving data @ {image_path_json}')
        with open(image_path_json, 'w') as f:
            json.dump(image_paths_dict, f, indent=2)
            
    else:
        raise ValueError("Unsupported dataset name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test split for dataset.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--dataset', type=str, required=True, choices=['YoLLaVA', 'MyVLM'], help='Dataset name')
    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset
    create_data_split(dataset_name, seed)
