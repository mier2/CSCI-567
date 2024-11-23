import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import wandb  # Commented out
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from model import CAML
import numpy as np

class TinyImageNetDataset(Dataset):
    def __init__(self, huggingface_dataset, transform=None):
        self.dataset = huggingface_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
       
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class MinimalPaintingsDataset(Dataset):
    def __init__(self, csv_file, transform=None, num_classes=5, images_per_class=6):
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.data['Subset'] = self.data['Subset'].str.strip("'")
        self.data = self.data[self.data['Subset'] == 'test']
        self.data = self.data[self.data['Image URL'].notna() & (self.data['Image URL'] != '')]
        
       
        self.classes = self.data['Labels'].unique()[:num_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            class_data = self.data[self.data['Labels'] == cls].iloc[:images_per_class]
            for _, row in class_data.iterrows():
                img_url = row['Image URL']
                try:
                    response = requests.get(img_url, timeout=10)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    self.images.append(img)
                    self.labels.append(self.class_to_idx[cls])
                    print(f"Loaded image for class {cls}")
                except Exception as e:
                    print(f"Error loading image for class {cls}: {str(e)}")
                    # If loading fails, create a dummy image
                    self.images.append(torch.randn(3, 224, 224))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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
        
        self.preloaded_images = []
        print("Preloading images...")
        for idx in tqdm(range(len(self.data))):
            img_url = self.data.iloc[idx]['Image URL']
            try:
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                self.preloaded_images.append(img)
            except Exception as e:
                print(f"Error loading item {idx} (URL: {img_url}): {str(e)}")
                self.preloaded_images.append(torch.zeros((3, 224, 224)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.preloaded_images[idx]
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

def train_epoch(model, train_loader, optimizer, device, epoch, args):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, labels)
        main_loss = F.cross_entropy(outputs, labels)
        
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        reg_loss = actual_model.get_regularization_loss()
        
        loss = main_loss + args.reg_coef * reg_loss

        loss.backward()
        
       
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_accuracy = 100. * train_correct / train_total
    return train_loss / len(train_loader), train_accuracy

def train(args, model, device):
    wandb.init(project="hybrid-caml", entity="xiaoyazhou", config=args)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tiny_imagenet = load_dataset("zh-plus/tiny-imagenet")
    train_dataset = TinyImageNetDataset(tiny_imagenet['train'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    paintings_test_dataset = PaintingsDataset(args.csv_file, 'test', transform=transform)

    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    def train_phase(phase, num_epochs, lr):
        nonlocal model, actual_model
        print(f"Starting Phase {phase}")
        
        optimizer = optim.Adam([
            {'params': actual_model.image_encoder.parameters(), 'lr': lr * 0.1},
            {'params': actual_model.hybrid_encoder.parameters()},
            {'params': actual_model.transformer_encoder.parameters()},
            {'params': actual_model.classifier.parameters()},
        ], lr=lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        best_test_accuracy = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, epoch, args)

            wandb.log({
                f"epoch_phase{phase}": epoch + 1,
                f"train_loss_phase{phase}": train_loss,
                f"train_accuracy_phase{phase}": train_accuracy,
            })
            
            print(f"Phase {phase} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

            if (epoch + 1) % args.eval_interval == 0:
                test_results = evaluate_nw1s(model, paintings_test_dataset, device)
                test_accuracy = test_results['nw1s_accuracy']
                print(f"Phase {phase} - {test_results['n_way']}w-1s Test Accuracy on Paintings: {test_accuracy:.2f}%")

                wandb.log({
                    f"epoch_phase{phase}": epoch + 1,
                    f"test_accuracy_phase{phase}": test_accuracy,
                })

                scheduler.step(test_accuracy)

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    torch.save(model.state_dict(), f'best_model_phase{phase}.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"Early stopping triggered after {epoch+1} epochs in Phase {phase}")
                        break

        model.load_state_dict(torch.load(f'best_model_phase{phase}.pth'))
        return best_test_accuracy

    
    phase1_accuracy = train_phase(1, args.num_epochs, args.learning_rate)

   
    actual_model.hybrid_encoder.hdc_vectors.requires_grad = True
    phase2_accuracy = train_phase(2, args.num_epochs_phase2, args.learning_rate * 0.1)

    final_results = evaluate_nw1s(model, paintings_test_dataset, device, n_episodes=10000)
    print(f"Final {final_results['n_way']}w-1s Test Accuracy on Paintings (10,000 episodes): {final_results['nw1s_accuracy']:.2f}%")

    wandb.log({
        "final_test_accuracy": final_results['nw1s_accuracy'],
        "phase1_best_accuracy": phase1_accuracy,
        "phase2_best_accuracy": phase2_accuracy
    })
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid CAML on TinyImageNet and test on Paintings Dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=100, help="Maximum number of epochs for Phase 1")
    parser.add_argument("--num_epochs_phase2", type=int, default=50, help="Maximum number of epochs for Phase 2")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--csv_file", type=str, default="data/painting_dataset_2021.csv", help="Path to Paintings dataset CSV file")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--eval_interval", type=int, default=5, help="Interval (in epochs) for evaluation and potential early stopping")
    parser.add_argument("--reg_coef", type=float, default=0.01, help="Coefficient for regularization loss")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.num_gpus > 1:
        print(f"Using {args.num_gpus} GPUs!")
        device = torch.device("cuda")
        args.batch_size = args.batch_size * args.num_gpus
        model = nn.DataParallel(CAML(num_classes=200, device=device))
        model = model.to(device)
    else:
        print("Using single GPU or CPU.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CAML(num_classes=200, device=device)
        model = model.to(device)

    print(model)
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # csv_file = "data/painting_dataset_2021.csv"  # Replace with your actual CSV file path
    # minimal_test_dataset = MinimalPaintingsDataset(csv_file, transform=transform, num_classes=5, images_per_class=6)

    # # Run 5w1s test case before training
    # print("Initial 5w1s test:")
    # evaluate_nw1s(model, minimal_test_dataset, device, n_episodes=2)

    
    train(args, model, device)

    #  nohup python train.py --num_gpus 2 --batch_size 64 --num_epochs 4 --patience 5 --eval_interval 5 > caml_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    # nohup python train.py --batch_size 64 --num_epochs 100 --num_epochs_phase2 50 --learning_rate 0.001 --reg_coef 0.01 > caml_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    