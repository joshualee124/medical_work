#!/usr/bin/env python3
"""
Standalone Checkpoint Analyzer
Just loads and tests your trained models - no training involved!
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.parallel import DataParallel
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
import numpy as np


class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):   
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        print(f"Looking for datasets in: {os.path.abspath(root_dir)}")

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"ERROR: Directory {root_dir} does not exist!")

        folders_found = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            folders_found.append(folder)

            if os.path.isdir(folder_path):
                # Assign label
                label = 1 if folder.lower().startswith(("brain", "head", "spine", "orbit")) else 0
    
                images_in_folder = []
                for root, _, files in os.walk(folder_path):
                    for img_file in files:
                        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                            images_in_folder.append(os.path.join(root, img_file))

                print(f"  Found {len(images_in_folder)} images in {folder} (Label={label})")

                self.image_paths.extend(images_in_folder)
                self.labels.extend([label] * len(images_in_folder))

        print(f"All folders found: {folders_found}")
        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Total Label 1 images: {sum(1 for l in self.labels if l == 1)}")
        print(f"Total Label 0 images: {sum(1 for l in self.labels if l == 0)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def analyze_checkpoint(checkpoint_path, val_dataset, device, save_wrong_pics=True, max_wrong_pics=200):
    """Analyze a specific checkpoint model on validation dataset"""
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")
    
    # Create model architecture
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = DataParallel(model)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create validation dataloader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Evaluation
    val_correct = 0
    val_total = 0
    wrong_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Collect wrong predictions for visualization
            if save_wrong_pics and len(wrong_predictions) < max_wrong_pics:
                for i in range(len(images)):
                    if predicted[i] != labels[i]:
                        wrong_predictions.append({
                            'image': images[i].cpu(),
                            'true_label': labels[i].item(),
                            'predicted_label': predicted[i].item(),
                            'confidence': torch.softmax(outputs[i], dim=0).max().item(),
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
                        if len(wrong_predictions) >= max_wrong_pics:
                            break
    
    val_acc = 100 * val_correct / val_total
    print(f"📊 Checkpoint Performance:")
    print(f"  Validation Accuracy: {val_acc:.2f}%")
    print(f"  Wrong Predictions: {val_total - val_correct}")
    print(f"  Total Samples: {val_total}")
    
    # Save wrong prediction images
    if save_wrong_pics and wrong_predictions:
        # Create clean directory name
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        wrong_pics_dir = f"wrong_predictions_{checkpoint_name}"
        os.makedirs(wrong_pics_dir, exist_ok=True)
        
        print(f"📸 Saving {len(wrong_predictions)} wrong prediction images to {wrong_pics_dir}/")
        
        for idx, wrong_pred in enumerate(wrong_predictions):
            # Convert tensor to PIL image
            img_tensor = wrong_pred['image']
            img_pil = transforms.ToPILImage()(img_tensor)
            
            # Create filename with prediction info
            filename = f"wrong_{idx:03d}_true_{wrong_pred['true_label']}_pred_{wrong_pred['predicted_label']}_conf_{wrong_pred['confidence']:.3f}.png"
            filepath = os.path.join(wrong_pics_dir, filename)
            
            # Save image
            img_pil.save(filepath)
        
        print(f"✅ Saved {len(wrong_predictions)} wrong prediction images")
    
    return val_acc


def main():
    print("🚀 Checkpoint Analyzer - No Training, Just Analysis!")
    print("=" * 60)
    
    # Configuration
    data_dir = "datasets"  # Your dataset directory
    val_split = 0.2
    
    # Update these paths to your actual checkpoint locations
    checkpoints = [
        "checkpoint_epoch_550.pth",  # Update this path
        "checkpoint_epoch_10.pth"   # Update this path
    ]
    
    # Check if checkpoints exist
    existing_checkpoints = []
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            existing_checkpoints.append(checkpoint)
            print(f"✅ Found checkpoint: {checkpoint}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint}")
    
    if not existing_checkpoints:
        print("\n❌ No checkpoints found! Please update the paths in the script.")
        return
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")
    
    # Load dataset
    print(f"\n📁 Loading dataset from: {data_dir}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = BrainDataset(root_dir=data_dir, transform=transform)
    
    # Stratified split
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels, random_state=42)
    
    val_dataset = Subset(full_dataset, val_idx)
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Analyze each checkpoint
    results = {}
    for checkpoint_path in existing_checkpoints:
        print(f"\n{'='*60}")
        accuracy = analyze_checkpoint(checkpoint_path, val_dataset, device)
        if accuracy is not None:
            results[checkpoint_path] = accuracy
    
    # Simple comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📈 ACCURACY COMPARISON")
        print("="*60)
        
        for checkpoint_path, accuracy in results.items():
            checkpoint_name = os.path.basename(checkpoint_path)
            print(f"{checkpoint_name}: {accuracy:.2f}%")
    
    print(f"\n🎉 Analysis complete! Check the wrong_predictions_* folders for misclassified images.")


if __name__ == "__main__":
    main()
