import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            # Only include folders ending with _png
            if os.path.isdir(folder_path) and folder.endswith("_png"):
                label = 1 if folder.lower().startswith(("brain", "head", "spine", "orbit")) else 0

                # Collect images
                images_in_folder = []
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        images_in_folder.append(os.path.join(folder_path, img_file))
                
                # Limit images per class if specified
                if max_images_per_class and len(images_in_folder) > max_images_per_class:
                    import random
                    images_in_folder = random.sample(images_in_folder, max_images_per_class)
                
                self.image_paths.extend(images_in_folder)
                self.labels.extend([label] * len(images_in_folder))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(data_dir="datasets", num_epochs=5, batch_size=32, lr=0.001, save_path="resnet50_ct_classifier.pth", max_images_per_class=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations for ResNet50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with image limit per class
    dataset = BrainDataset(root_dir=data_dir, transform=transform, max_images_per_class=max_images_per_class)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Total images found: {len(dataset)}")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
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

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
