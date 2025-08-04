import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split

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

                # Collect images
                images_in_folder = [
                    os.path.join(folder_path, img_file)
                    for img_file in os.listdir(folder_path)
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg"))
                ]

                print(f"  Found {len(images_in_folder)} images in {folder} (Label={label})")

                self.image_paths.extend(images_in_folder)
                self.labels.extend([label] * len(images_in_folder))

        print(f"All folders found: {folders_found}")
        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(data_dir="datasets", num_epochs=10, batch_size=32, lr=0.0001, save_path="resnet50_ct_classifier.pth", val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations for ResNet50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = BrainDataset(root_dir=data_dir, transform=transform)

    # Stratified split
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Load ResNet50
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_model()
