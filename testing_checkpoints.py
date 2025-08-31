import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.parallel import DataParallel
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
import wandb 
import numpy as np
from torchvision.transforms import ToPILImage
from collections import OrderedDict
import torchvision.transforms as transforms


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

        # ✅ Show count of label distribution
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
    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    full_dataset = BrainDataset(root_dir='datasets', transform=transform)

    # Stratified split

    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,num_workers = 32)
    checkpoint_path = 'checkpoint_epoch_10.pth'
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # load checkpoint safely
    ckpt = torch.load("checkpoint_epoch_10.pth", map_location="cpu")

    # some trainings save under 'state_dict'
    state = ckpt.get("state_dict", ckpt)

    # strip 'module.' if present
    new_state = OrderedDict()
    for k, v in state.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[new_k] = v

    # if classifier shape differs, drop it so your new fc stays
    msd = model.state_dict()
    for head_k in ["fc.weight", "fc.bias"]:
        if head_k in new_state and new_state[head_k].shape != msd[head_k].shape:
            new_state.pop(head_k)

    # load (allow dropped head if needed)
    model.load_state_dict(new_state, strict=False)

    # now move to GPU(s)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)
    model.eval()
    true1pred1_counter = 0
    false1pred1_counter = 0
    false1pred0_counter = 0
    true1pred0_counter = 0
    to_pil = ToPILImage()

    # make sure output dirs exist
    os.makedirs("true1pred1", exist_ok=True)
    os.makedirs("true1pred0", exist_ok=True)
    os.makedirs("true0pred0", exist_ok=True)
    os.makedirs("true0pred1", exist_ok=True)

    true1pred1_counter = 0
    true1pred0_counter = 0
    true0pred0_counter = 0
    true0pred1_counter = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)   # [batch]

            for i in range(len(labels)):
                img = to_pil(images[i].cpu())  # convert one image to PIL
                y, p = int(labels[i].item()), int(preds[i].item())

                if y == 1 and p == 1 and true1pred1_counter < 100:
                    img.save(f"true1pred1/pred_{true1pred1_counter}.png")
                    true1pred1_counter += 1

                elif y == 1 and p == 0 and true1pred0_counter < 100:
                    img.save(f"true1pred0/pred_{true1pred0_counter}.png")
                    false1pred1_counter += 1

                elif y == 0 and p == 0 and true0pred0_counter < 100:
                    img.save(f"true0pred0/pred_{true0pred0_counter}.png")
                    false1pred0_counter += 1

                elif y == 0 and p == 1 and true0pred1_counter < 100:
                    img.save(f"true0pred1/pred_{true0pred1_counter}.png")
                    true1pred0_counter += 1

            # early exit once you hit 100 in each bucket
            if (true1pred1_counter >= 100 and false1pred1_counter >= 100 and
                false1pred0_counter >= 100 and true1pred0_counter >= 100):
                break





        #save 400 images, 100 images where predicted is 1 and 100 images where predicted is 0
        