"""
Training script for Pokémon Classifier

This script trains a CNN model to classify Pokémon images.
The trained model can then be used with the explainability analysis tool.
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np

from utils_vis import MEAN, STD

# #region agent log
LOG_PATH = "/Users/tksiankop/Downloads/pokemon_gradcam_starter/.cursor/debug.log"
# #endregion

def log_debug(session_id: str, run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    """Log debug information to NDJSON file"""
    # #region agent log
    try:
        import time as time_module
        log_entry = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time_module.time() * 1000)
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion


class PokemonDataset(Dataset):
    """Dataset for Pokémon images"""
    
    def __init__(self, image_dir: str, class_names: list, transform=None, is_train=True):
        self.image_dir = Path(image_dir)
        self.class_names = sorted(class_names)
        self.transform = transform
        self.is_train = is_train
        
        # Build dataset: (image_path, class_idx)
        self.samples = []
        for img_path in self.image_dir.glob("*.png"):
            pokemon_name = img_path.stem
            try:
                class_idx = next(
                    i for i, name in enumerate(self.class_names)
                    if name.lower() == pokemon_name.lower()
                )
                self.samples.append((str(img_path), class_idx))
            except StopIteration:
                continue
        
        # #region agent log
        log_debug("debug-session", "train", "A", "train_pokemon_classifier.py:PokemonDataset.__init__",
                 "Dataset initialized", {"num_samples": len(self.samples), "num_classes": len(self.class_names),
                                       "is_train": is_train})
        # #endregion
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # #region agent log
            log_debug("debug-session", "train", "B", "train_pokemon_classifier.py:PokemonDataset.__getitem__",
                     "Error loading image", {"img_path": img_path, "error": str(e)})
            # #endregion
            # Return a black image if loading fails
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_transforms(is_train=True):
    """Get data transforms for training/validation"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


def create_model(model_arch: str, num_classes: int, pretrained: bool = True):
    """Create model architecture"""
    # #region agent log
    log_debug("debug-session", "train", "C", "train_pokemon_classifier.py:create_model",
             "Creating model", {"model_arch": model_arch, "num_classes": num_classes, "pretrained": pretrained})
    # #endregion
    
    if model_arch.lower().startswith('resnet'):
        arch_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        if model_arch.lower() not in arch_map:
            raise ValueError(f"Unsupported ResNet architecture: {model_arch}")
        
        # Handle pretrained parameter (newer torchvision uses weights parameter)
        try:
            if pretrained:
                model = arch_map[model_arch.lower()](weights='DEFAULT')
            else:
                model = arch_map[model_arch.lower()](weights=None)
        except TypeError:
            # Fallback for older torchvision versions
            model = arch_map[model_arch.lower()](pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_arch.lower().startswith('efficientnet'):
        try:
            arch_map = {
                'efficientnet_b0': models.efficientnet_b0,
                'efficientnet_b1': models.efficientnet_b1,
                'efficientnet_b2': models.efficientnet_b2,
                'efficientnet_b3': models.efficientnet_b3,
                'efficientnet_b4': models.efficientnet_b4,
            }
            if model_arch.lower() not in arch_map:
                raise ValueError(f"Unsupported EfficientNet architecture: {model_arch}")
            
            # Handle pretrained parameter (newer torchvision uses weights parameter)
            try:
                if pretrained:
                    model = arch_map[model_arch.lower()](weights='DEFAULT')
                else:
                    model = arch_map[model_arch.lower()](weights=None)
            except TypeError:
                # Fallback for older torchvision versions
                model = arch_map[model_arch.lower()](pretrained=pretrained)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        except AttributeError:
            raise ValueError(f"EfficientNet not available in this torchvision version")
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")
    
    # #region agent log
    log_debug("debug-session", "train", "C", "train_pokemon_classifier.py:create_model",
             "Model created", {"model_arch": model_arch})
    # #endregion
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # #region agent log
    log_debug("debug-session", "train", "D", "train_pokemon_classifier.py:train_epoch",
             "Starting training epoch", {"epoch": epoch, "num_batches": len(dataloader)})
    # #endregion
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # #region agent log
    log_debug("debug-session", "train", "D", "train_pokemon_classifier.py:train_epoch",
             "Epoch completed", {"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_acc})
    # #endregion
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # #region agent log
    log_debug("debug-session", "train", "E", "train_pokemon_classifier.py:validate",
             "Starting validation", {"num_batches": len(dataloader)})
    # #endregion
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    # #region agent log
    log_debug("debug-session", "train", "E", "train_pokemon_classifier.py:validate",
             "Validation completed", {"loss": val_loss, "accuracy": val_acc})
    # #endregion
    
    return val_loss, val_acc


def get_class_names(image_dir: str) -> list:
    """Extract class names from image filenames"""
    image_dir = Path(image_dir)
    class_names = sorted(set(img.stem for img in image_dir.glob("*.png")))
    return class_names


def main():
    parser = argparse.ArgumentParser(description="Train Pokémon Classifier")
    parser.add_argument("--data_dir", type=str, default="data/pokemon_pics",
                       help="Directory containing Pokémon images")
    parser.add_argument("--model_arch", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                               "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
                       help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                       help="Don't use pretrained weights")
    
    args = parser.parse_args()
    
    # #region agent log
    log_debug("debug-session", "train", "F", "train_pokemon_classifier.py:main",
             "Training started", {"args": vars(args)})
    # #endregion
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes")
    print(f"Classes: {', '.join(class_names[:10])}{'...' if len(class_names) > 10 else ''}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    full_dataset = PokemonDataset(
        args.data_dir,
        class_names,
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update transforms for validation
    val_dataset.dataset.transform = get_transforms(is_train=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print(f"Creating {args.model_arch} model...")
    model = create_model(args.model_arch, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print()
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'num_classes': num_classes,
            'class_names': class_names,
            'model_arch': args.model_arch,
        }
        
        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            print()
    
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"  - best.pth: Best model (Val Acc: {best_val_acc:.2f}%)")
    print(f"  - latest.pth: Latest model")
    print()
    print("To use with explainability analysis:")
    print(f"  python run_explainability.py --model_path {checkpoint_dir}/best.pth --model_arch {args.model_arch}")


if __name__ == "__main__":
    main()

