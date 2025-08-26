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

def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calculate comprehensive metrics for binary classification"""
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic accuracy
    accuracy = (y_true == y_pred).mean()
    
    # Calculate TP, TN, FP, FN manually for sensitivity/specificity
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    f1 = f1_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    auc = 0
    if y_scores is not None:
        auc = roc_auc_score(y_true, y_scores)
    
    # Calculate high-confidence accuracy (above 80% for positive predictions)
    high_conf_accuracy = 0
    high_conf_count = 0
    if y_scores is not None:
        y_scores = np.array(y_scores)
        # Find predictions with confidence > 80% for positive class (label=1)
        # y_scores > 0.8 means >80% confidence it's a brain/neuro image
        high_conf_mask = y_scores > 0.8
        high_conf_indices = np.where(high_conf_mask)[0]
        
        if len(high_conf_indices) > 0:
            high_conf_true = y_true[high_conf_indices]
            high_conf_pred = y_pred[high_conf_indices]
            high_conf_accuracy = (high_conf_true == high_conf_pred).mean()
            high_conf_count = len(high_conf_indices)
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'auc': auc,
        'high_conf_accuracy': high_conf_accuracy,
        'high_conf_count': high_conf_count
    }

#try increasing batch size 
# is it using all 4 gpus?

def train_model(data_dir="datasets", num_epochs=1000, batch_size=512, lr=0.0005, save_path="resnet50_ct_classifier.pth", val_split=0.2):
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    wandb.init(project="neurofinder_classifier", config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "architecture": "resnet50",
        "optimizer": "Adam",
        "val_split": val_split,
        "gpus": torch.cuda.device_count()
    })

    # Transformations for ResNet50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    # Load dataset
    full_dataset = BrainDataset(root_dir=data_dir, transform=transform)

    # Stratified split
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers = 32)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Load ResNet50
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    model = DataParallel(model)

    criterion = nn.CrossEntropyLoss()#maybe try binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Collect predictions for metrics
        train_preds = []
        train_labels = []
        train_scores = []

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
            
            # Collect for metrics
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().detach().numpy())
            train_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(train_labels, train_preds, train_scores)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        # Collect validation predictions
        val_preds = []
        val_labels = []
        val_scores = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Collect for metrics
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().detach().numpy())
                val_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_acc = 100 * val_correct / val_total
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(val_labels, val_preds, val_scores)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Val F1: {val_metrics['f1_score']:.3f} | "
              f"Val AUC: {val_metrics['auc']:.3f} | "
              f"Val High-Conf Acc: {val_metrics['high_conf_accuracy']:.3f} ({val_metrics['high_conf_count']} samples)")

        # Log all metrics to wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            # Training metrics
            "train_sensitivity": train_metrics['sensitivity'],
            "train_specificity": train_metrics['specificity'],
            "train_f1_score": train_metrics['f1_score'],
            "train_balanced_accuracy": train_metrics['balanced_accuracy'],
            "train_auc": train_metrics['auc'],
            "train_high_conf_accuracy": train_metrics['high_conf_accuracy'],
            "train_high_conf_count": train_metrics['high_conf_count'],
            # Validation metrics
            "val_sensitivity": val_metrics['sensitivity'],
            "val_specificity": val_metrics['specificity'],
            "val_f1_score": val_metrics['f1_score'],
            "val_balanced_accuracy": val_metrics['balanced_accuracy'],
            "val_auc": val_metrics['auc'],
            "val_high_conf_accuracy": val_metrics['high_conf_accuracy'],
            "val_high_conf_count": val_metrics['high_conf_count']
        })

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Final model saved to {save_path}")
    wandb.save(save_path)

    

    # =========================
# Checkpoint evaluation (accuracy only + 200 wrong images)
# =========================
import csv, shutil
from glob import glob

def _build_eval_dataset(data_dir, val_split):
    # Deterministic (no augmentation) eval transforms
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_eval = BrainDataset(root_dir=data_dir, transform=eval_transform)

    # Recreate the SAME split used in training
    indices = list(range(len(full_eval)))
    labels = full_eval.labels
    _, val_idx_local = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=42
    )

    val_subset = Subset(full_eval, val_idx_local)
    return full_eval, val_subset, val_idx_local

def _build_resnet50_head(num_classes=2):
    m = models.resnet50(weights=None)  # weights don't matter; checkpoint will load real weights
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m

def _load_state_flex(model, state):
    # Accept raw state_dict, {"state_dict": ...}, and DataParallel ("module.") keys
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        fixed = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        model.load_state_dict(fixed, strict=True)
    return model

@torch.no_grad()
def _eval_one_checkpoint(
    ckpt_path, device, val_loader, val_image_paths, val_labels,
    out_dir="val_misclassified_200", max_wrong=200
):
    os.makedirs(out_dir, exist_ok=True)

    model = _build_resnet50_head(num_classes=2).to(device)
    state = torch.load(ckpt_path, map_location=device)
    _load_state_flex(model, state)
    model.eval()

    all_prob1, all_pred = [], []
    for images, _ in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        preds = (prob1 >= 0.5).long()
        all_prob1.append(prob1.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    import numpy as np
    y_scores = np.concatenate(all_prob1)
    y_pred = np.concatenate(all_pred)
    val_labels_np = np.array(val_labels)

    # Accuracy only
    acc = (y_pred == val_labels_np).mean()
    print(f"[VAL] {os.path.basename(ckpt_path)}  Accuracy: {acc:.4f}")

    # Save exactly 200 misclassified images (val order)
    mis_idx = np.where(y_pred != val_labels_np)[0]
    take = mis_idx[:max_wrong]
    manifest_path = os.path.join(out_dir, f"misclassified_{os.path.basename(ckpt_path)}.csv")

    saved = 0
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig_path", "saved_path", "true_label", "pred_label", "prob_class1"])
        for k in take:
            src = val_image_paths[k]
            true_l = int(val_labels_np[k]); pred_l = int(y_pred[k]); p1 = float(y_scores[k])
            new_name = f"WRONG_true{true_l}_pred{pred_l}_p1-{p1:.3f}__{os.path.basename(src)}"
            dst = os.path.join(out_dir, new_name)
            try:
                shutil.copy2(src, dst)
                w.writerow([src, dst, true_l, pred_l, p1])
                saved += 1
            except Exception:
                pass

    print(f"Saved {saved} misclassified images to: {out_dir}")
    print(f"Manifest CSV: {manifest_path}")
    return acc, saved, manifest_path

def evaluate_checkpoints(
    data_dir="datasets",
    val_split=0.2,
    checkpoints_glob="checkpoint_epoch_*.pth",
    out_dir="val_misclassified_200",
    batch_size=256,
    num_workers=8,
    max_wrong=200
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    full_eval, val_subset, val_idx_local = _build_eval_dataset(data_dir, val_split)

    # DataLoader
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # Paths + labels in val order for saving copies
    val_image_paths = [full_eval.image_paths[i] for i in val_idx_local]
    val_labels = [full_eval.labels[i] for i in val_idx_local]

    # Resolve checkpoints
    paths = sorted(glob(checkpoints_glob))
    if not paths and os.path.isfile(checkpoints_glob):
        paths = [checkpoints_glob]

    if not paths:
        print(f"[Eval] No checkpoints matched: {checkpoints_glob}")
        return

    print(f"[Eval] Found {len(paths)} checkpoint(s):")
    for p in paths:
        print(" -", p)

    # Evaluate each
    for p in paths:
        _eval_one_checkpoint(
            p, device, val_loader, val_image_paths, val_labels,
            out_dir=out_dir, max_wrong=max_wrong
        )

# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="eval",
                        help="train = run training loop, eval = only evaluate checkpoints (default).")
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--checkpoints", default="checkpoint_epoch_*.pth",
                        help="Glob or single path to a .pth checkpoint.")
    parser.add_argument("--out_dir", default="val_misclassified_200")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_wrong", type=int, default=200,
                        help="Exactly this many misclassified images will be attempted; if fewer exist, all will be saved.")

    # (Optional) training hyperparams if you ever run --mode train
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train_batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--save_path", default="resnet50_ct_classifier.pth")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.train_batch,
            lr=args.lr,
            save_path=args.save_path,
            val_split=args.val_split
        )
    else:
        evaluate_checkpoints(
            data_dir=args.data_dir,
            val_split=args.val_split,
            checkpoints_glob=args.checkpoints,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_wrong=args.max_wrong
        )
