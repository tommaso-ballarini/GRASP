import os
import re
import shutil
import glob
import json
import random
import sys
import argparse

sys.path.append("src/utils")
from mini_cpm_info import MiniCPMDescription
from defined import yollava_reverse_category_dict, myvlm_reverse_category_dict
from create_train_test_split import create_data_split

client = MiniCPMDescription()

def extract_answer_term(text, term):
    patterns = {
        'info': r'"info":\s*(?:"([^"]+)"|([\w-]+))',
        'general': r'"general":\s*(?:"([^"]+)"|([\w-]+))',
        'category': r'"category":\s*(?:"([^"]+)"|([\w.]+))',
        'distinct features': r'"distinct features":\s*(?:"([^"]+)"|([\w.]+))',
    }
    
    pattern = patterns.get(term)
    if not pattern:
        return None
    
    match = re.search(pattern, text)
    if match:
        return match.group(1) or match.group(2)
    else:
        parts = text.split(term)
        if parts:
            return re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
        return None

def rename_image_with_concept(image_path, concept_identifier):
    """
    Rename an image to a new name: concept_identifier.ext.
    """
    _, ext = os.path.splitext(os.path.basename(image_path))
    return f"{concept_identifier}{ext}"

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def sample_and_copy_images(concept_identifier, image_paths, target_path, num_train, seed):
    """
    Randomly sample num_train images from image_paths, copy and rename them.
    """
    random.seed(seed)
    sampled_paths = random.sample(image_paths, num_train)
    try:
        for image_path in sampled_paths:
            new_image_name = rename_image_with_concept(image_path, concept_identifier)
            target_image_path = os.path.join(target_path, new_image_name)
            shutil.copy2(image_path, target_image_path)
            print(f"Copied and renamed '{image_path}' to '{target_image_path}'")
    except Exception as e:
        print(e)
            

def create_db_entry(args, image_path, concept_identifier, category):
    """
    Create a database entry for an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' does not exist.")
    
    name = os.path.splitext(os.path.basename(image_path))[0]
    if not args.user_defined:
        category = None
    
    info_str = client.generate_caption(image_path, category, concept_identifier, args)
    info_str = info_str.replace('```json\n', '').replace('\n```', '')
    
    try:
        info = json.loads(info_str)
        desc = info
        if 'category' in info:
            category = info['category']
        
        entry = {
            "name": name,
            "image": [image_path],
            "info": desc,
            "category": category,
        }
        return entry
    except Exception:
        print("Resolving error")
        entry = {
            "name": name,
            "image": [image_path],
            "info": {
                "general": extract_answer_term(info_str, "general"),
                "category": extract_answer_term(info_str, "category"),
                "distinct features": extract_answer_term(info_str, 'distinct features')
            },
            "category": extract_answer_term(info_str, "category"),
        }
        return entry

def create_db(args, path_to_cat, category_name):
    """
    Create a JSON database file for all images in the specified directory.
    """
    image_paths = glob.glob(os.path.join(path_to_cat, '*.jpg')) + glob.glob(os.path.join(path_to_cat, '*.png'))
    if not image_paths:
        raise FileNotFoundError(f"No images found in the directory '{path_to_cat}'.")
    
    db_entries = {"concept_dict": {}, "path_to_concept": {}}
    for image_path in image_paths:
        cid = os.path.basename(image_path).split('.')[0]
        if cid in myvlm_reverse_category_dict:
            category_name = myvlm_reverse_category_dict[cid]
        elif cid in yollava_reverse_category_dict:
            category_name = yollava_reverse_category_dict[cid]
        
        entry = create_db_entry(args, image_path, cid, category_name)
        key = f'<{cid}>'
        if key not in db_entries["concept_dict"]:
            db_entries["concept_dict"][key] = entry
        else:
            db_entries["concept_dict"][key]['image'].append(image_path)
        db_entries["path_to_concept"][image_path] = key
       
    if args.user_defined:
        if args.template_based:
            database_name = 'database_user_defined_cat_template.json'
        else:
             database_name = 'database_user_defined_cat_no_template.json'
    else:
        if args.template_based:
            database_name = 'database_clip_based_cat_template.json'
        else:    
            database_name = 'database_clip_based_cat_no_template.json'
    # if detailed else 'database.json'
    db_path = os.path.join(path_to_cat, database_name)
    with open(db_path, 'w') as db_file:
        json.dump(db_entries, db_file, indent=4)
    print(f"Database created at '{db_path}' with {len(db_entries['concept_dict'])} entries.")
    return len(db_entries['concept_dict'])

def get_filename_for_dataname(dataname, seed):
    dataname_2_filename = {
        "PerVA": f"data/perva-data/train_test_val_seed_{seed}.json",
        "YoLLaVA": f"data/yollava-data/train_test_val_seed_{seed}.json",
        "MyVLM": f"data/myvlm-data/train_test_val_seed_{seed}.json"
    }
    return dataname_2_filename.get(dataname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a database for a given dataset.")
    parser.add_argument('--dataset', type=str, required=True, choices=["PerVA", "YoLLaVA", "MyVLM"], help="Dataset name")
    parser.add_argument('--seed', type=int, required=True, help="Random seed value")
    parser.add_argument('--user_defined', action='store_true', help="User defined categories")
    parser.add_argument('--template_based', action='store_true', help="template")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    random.seed(seed)
    num_train_imgs = 1
    if dataset == 'PerVA':
        try:
            json_filepath = get_filename_for_dataname(dataset, seed)
        except FileNotFoundError:
            print(f"{json_filepath} Not Found")
        json_data = load_json(json_filepath)
        # Process each category separately
        total = 0
        for category, cat_data in json_data.items():
            target_dir = f'example_database/{dataset}_seed_{seed}/{category}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                for concept, image_path_dict in cat_data.items():
                    train_images = image_path_dict.get('train', [])
                    if train_images:
                        sample_and_copy_images(concept, train_images, target_dir, num_train_imgs, seed)
            else:
                print("Images exist, creating db")
            total+=create_db(args, target_dir, category)
        print(f"number of cat:{total}")
    elif dataset in ["YoLLaVA", "MyVLM"]:
        try:
            json_filepath = get_filename_for_dataname(dataset, seed)
        except FileNotFoundError:
            print(f"{json_filepath} Not Found")
        json_data = load_json(json_filepath)
        target_dir = f'example_database/{dataset}_seed_{seed}/all/'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            for concept, image_path_dict in json_data['all'].items():
                train_images = image_path_dict.get('train', [])
                if train_images:
                    sample_and_copy_images(concept, train_images, target_dir, num_train_imgs, seed)
        else:
            print("Images exist, creating db")
        create_db(args, target_dir, 'all')
    else:
        raise ValueError("Unsupported dataset name")