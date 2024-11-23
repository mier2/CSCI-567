import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import random
from tqdm import tqdm
import argparse
import wandb

from model import CAML, get_elmes

class PaintingsDataset(Dataset):
    def __init__(self, csv_file, subset, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data['Subset'] = self.data['Subset'].str.strip("'")
        self.data = self.data[self.data['Subset'] == subset]
        self.data = self.data[self.data['Image URL'].notna() & (self.data['Image URL'] != '')]
        self.transform = transform
        self.classes = sorted(list(set([label.strip() for labels in self.data['Labels'] for label in labels.split(',')])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_url = self.data.iloc[idx]['Image URL']
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading item {idx} (URL: {img_url}): {str(e)}")
            img = torch.zeros((3, 224, 224))

        labels = self.data.iloc[idx]['Labels'].split(',')
        label_idx = self.class_to_idx[labels[0].strip()]

        return img, label_idx

def evaluate_nw1s(model, test_dataset, device, n_episodes=1000, max_way=5):
    model.eval()
    correct = 0
    total = 0

    class_indices = {}
    for idx, (_, label) in enumerate(test_dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    valid_classes = [cls for cls, indices in class_indices.items() if len(indices) >= 2]
    n_way = min(len(valid_classes), max_way)

    if n_way < 2:
        print(f"Warning: Only {len(valid_classes)} classes have sufficient samples for evaluation.")
        print("At least 2 classes are required. Skipping evaluation.")
        return {
            "nw1s_accuracy": 0.0,
            "nw1s_correct": 0,
            "nw1s_total": 0,
            "nw1s_episodes": 0,
            "n_way": 0
        }

    print(f"Performing {n_way}-way 1-shot evaluation")

    for _ in tqdm(range(n_episodes), desc=f"{n_way}w-1s Testing"):
        episode_classes = random.sample(valid_classes, n_way)
        
        support_images, support_labels, query_images, query_labels = [], [], [], []
        
        for class_idx, class_label in enumerate(episode_classes):
            selected_indices = random.sample(class_indices[class_label], 2)
            
            support_img, _ = test_dataset[selected_indices[0]]
            query_img, _ = test_dataset[selected_indices[1]]
            
            support_images.append(support_img)
            support_labels.append(class_idx)
            query_images.append(query_img)
            query_labels.append(class_idx)

        support_images = torch.stack(support_images).to(device)
        query_images = torch.stack(query_images).to(device)
        
        support_labels = torch.tensor(support_labels).to(device)
        query_labels = torch.tensor(query_labels).to(device)

        support_images = support_images.unsqueeze(1)  # [n_way, 1, 3, 224, 224]
        support_labels = support_labels.unsqueeze(1)  # [n_way, 1]
        query_images = query_images.unsqueeze(0)  # [1, n_way, 3, 224, 224]

        with torch.no_grad():
            logits = model.module.meta_test(support_images, support_labels, query_images, way=n_way, shot=1)
            _, predicted = logits.max(1)
            correct += predicted.eq(query_labels).sum().item()
            total += query_labels.size(0)

    accuracy = 100. * correct / total if total > 0 else 0.0
    return {
        "nw1s_accuracy": accuracy,
        "nw1s_correct": correct,
        "nw1s_total": total,
        "nw1s_episodes": n_episodes,
        "n_way": n_way
    }

def main():
    parser = argparse.ArgumentParser(description="Test CAML model on Paintings Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_5.pth", help="Path to model checkpoint")
    parser.add_argument("--csv_file", type=str, default="data/painting_dataset_2021.csv", help="Path to Paintings dataset CSV file")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--n_episodes", type=int, default=1000, help="Number of episodes for evaluation")
    parser.add_argument("--max_way", type=int, default=5, help="Maximum number of ways for n-way 1-shot evaluation")
    args = parser.parse_args()

    wandb.init(project="caml_test", entity="xiaoyazhou", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nn.DataParallel(CAML(num_classes=200, device=device))
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    paintings_test_dataset = PaintingsDataset(args.csv_file, 'test', transform=transform)

    print("Starting n-way 1-shot evaluation...")
    results = evaluate_nw1s(model, paintings_test_dataset, device, n_episodes=args.n_episodes, max_way=args.max_way)

    print(f"{results['n_way']}-way 1-shot accuracy: {results['nw1s_accuracy']:.2f}%")
    print(f"Correct predictions: {results['nw1s_correct']}")
    print(f"Total predictions: {results['nw1s_total']}")
    print(f"Number of episodes: {results['nw1s_episodes']}")

    wandb.log(results)
    wandb.finish()

if __name__ == "__main__":
    main()