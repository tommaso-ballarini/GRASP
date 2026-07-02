import os
import json
import glob
import argparse


def get_subdirs_and_images(filename='train_test_val_split.json', category=None):
    subdirs = []
    val_images, test_images = [], []
    with open(filename, 'r') as f:
        json_data = json.load(f)
    for key in json_data[category].keys():
        subdirs.append(key)
        test_images.extend(json_data[category][key]['test'])
        val_images.extend(json_data[category][key]['val'])
    return subdirs, test_images, val_images

def create_image_question_list(test_images, val_images, subdir_name):
    result = {}
    split_dict = {"test":test_images, "val":val_images}
    for split, images in split_dict.items():
        image_question_list = []
        for image_path in images:
            question = f"Is <{subdir_name}> in the image? Answer with a single word, either yes or no."
            gt = "yes" if subdir_name in image_path else "no"
            image_question_list.append({
                "image": image_path,
                "question": question,
                "gt": gt
            })
        result[split] = image_question_list
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create eval files for a given dataset.")
    parser.add_argument('--dataset', type=str, required=True, choices=["PerVA", "YoLLaVA", "MyVLM"], help="Dataset name")
    parser.add_argument('--seed', type=int, required=True, help="Random seed value")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    data = {
        "PerVA":{
            'db_path':f'example_database/PerVA_seed_{seed}',
            'json_path':f'data/perva-data/train_test_val_seed_{seed}.json'
            },
        "YoLLaVA":{
            'db_path':f'example_database/YoLLaVA_seed_{seed}',
            'json_path':f'data/yollava-data/train_test_val_seed_{seed}.json'
            },
        "MyVLM":{
            'db_path':f'example_database/MyVLM_seed_{seed}',
            'json_path':f'data/myvlm-data/train_test_val_seed_{seed}.json'
            }
        }
    categories = os.listdir(data[dataset]['db_path'])
    for category in categories:
        subdirs, test_images, val_images = get_subdirs_and_images(filename=data[dataset]['json_path'], category=category)
        print("Subdirectories:", len(subdirs))
        print("Test Images:", len(test_images))
        print("Val Images:", len(val_images))
        for subdir in subdirs:
            result = create_image_question_list(test_images, val_images, subdir)
            os.makedirs(f"eval_files/{dataset}_seed_{args.seed}/{category}", exist_ok=True)
            with open(f"eval_files/{dataset}_seed_{args.seed}/{category}/{subdir}.json", "w") as f:
                json.dump(result, f, indent=4)
