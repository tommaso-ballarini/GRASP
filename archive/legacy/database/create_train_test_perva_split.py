import os
import glob
import json
import time
import clip
import torch
import numpy as np
import argparse
from PIL import Image
from sklearn.cluster import KMeans
import fcntl  # For file locking
from tqdm import *

class CLIPImageProcessor:
    def __init__(self, clip_model="ViT-B/32", device="cuda"):
        self.device = device
        self.clip_model, self.clip_transform = clip.load(clip_model, device=device)
        self.clip_model.eval()
    
    def extract_clip_features(self, image_files):
        imgs = [Image.open(image_file) for image_file in image_files]
        clip_input = torch.stack([self.clip_transform(img).unsqueeze(0).to(self.device) 
                                  for img in imgs]).squeeze()
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(clip_input).cpu().numpy()
        return clip_features
    
    def get_closest_to_mean_features(self, image_features, top_n=5):
        mean_feature = np.mean(image_features, axis=0)
        distances = np.linalg.norm(image_features - mean_feature, axis=1)
        closest_indices = np.argsort(distances)[:top_n]
        return mean_feature, closest_indices
    
    def get_distant_to_mean_features(self, mean_feature, image_features, top_n=5):
        distances = np.linalg.norm(image_features - mean_feature, axis=1)
        top_n = min(top_n, len(distances))
        farthest_indices = np.argsort(distances)[-top_n:]
        return farthest_indices
    
    def cluster_features(self, image_features, num_clusters, seed):
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
        cluster_labels = kmeans.fit_predict(image_features)
        cluster_centroids = kmeans.cluster_centers_
        distances = np.linalg.norm(image_features - cluster_centroids[cluster_labels], axis=1)

        representative_indices = np.zeros(num_clusters, dtype=int)
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            cluster_distances = distances[cluster_mask]
            closest_index_in_cluster = np.argmin(cluster_distances)
            representative_indices[cluster_id] = cluster_indices[closest_index_in_cluster]
        return representative_indices, cluster_centroids, cluster_labels
    
    def get_train_val_split(self, train_image_files, test_image_files, num_samples, seed):
        train_clip_features = self.extract_clip_features(train_image_files)
        test_clip_features = self.extract_clip_features(test_image_files)
        # indices, _, _ = self.cluster_features(clip_features, num_samples, seed)
        mean_feature, tr_indices = self.get_closest_to_mean_features(train_clip_features, top_n=num_samples)
        te_indices = self.get_distant_to_mean_features(mean_feature, test_clip_features, top_n=num_samples)
        tr_image_files = [train_image_files[index] for index in tr_indices]
        val_image_files = [image_file for image_file in train_image_files if image_file not in tr_image_files]
        test_image_files = [test_image_files[index] for index in te_indices]
        return tr_image_files, val_image_files, test_image_files

def update_json(file_path, key, value):
    """Safely update a JSON file in a concurrent environment."""
    try:
        # import pdb;pdb.set_trace()
        with open(file_path, 'w') as f:
            # fcntl.flock(f, fcntl.LOCK_EX)  # Acquire exclusive lock
            try:
                data = json.load(f) if os.stat(file_path).st_size > 0 else {}
            except json.JSONDecodeError:
                data = {}
            
            # Update the data
            if key not in data:
                data[key] = {}
            data[key].update(value)
            
            # Truncate and write new data
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

            # Release lock before closing the file
            # fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error updating JSON file: {e}")


def count_json_keys(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Count top-level keys
        top_level_key_count = len(data)
        num_concepts = 0
        for concept in data.keys():
            num_concepts+=len(data[concept])
        # Count keys within each dictionary at the top level
        # nested_key_counts = {key: len(value) if isinstance(value, dict) else 0 for key, value in data.items()}

        print(num_concepts)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processor = CLIPImageProcessor(clip_model="ViT-B/32", device="cuda")
    with open('data/perva-data/PerVA_concept_list.txt', 'r') as f:
        lines = f.readlines()
    data = {}
    start = time.time()
    for line in tqdm(lines):
        category_name, concept_name = line.strip().split(',')
        path = f'data/perva-data/train/{category_name}/{concept_name}'
        te_path = f'data/perva-data/test/{category_name}/{concept_name}'
        tr_img_paths = glob.glob(f"{path}/*.jpg", recursive=True)
        te_img_paths = glob.glob(f"{te_path}/*.jpg", recursive=True)
        
        num_samples = min(5, len(tr_img_paths)-1)
        train_images, val_images, test_images = processor.get_train_val_split(tr_img_paths, te_img_paths, num_samples=num_samples, seed=args.seed)
        if category_name not in data:
            data[category_name] = {}
        data[category_name][concept_name] = {
                "train": train_images, 
                "val": val_images,
                "test": test_images
            }
    save_file = f'data/perva-data/train_test_val_perva_seed_{args.seed}.json'
    with open(save_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Time taken: {time.time() - start}")
    print(f"File saved @ {save_file}")