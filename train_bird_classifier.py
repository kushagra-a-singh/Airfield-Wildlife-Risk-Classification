"""
Training Script for Bird Species Classification
Fine-tunes ResNet18 and MobileNet models for airport bird species classification
Supports ensemble training for 7 airport bird species
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.classification.species_classifier import (
    MobileNetBirdClassifier,
    ResNetBirdClassifier,
)

DATA_DIR = "data/airport7/"
CLASS_NAMES = [
    "black_kite",
    "brahminy_kite",
    "cormorant",
    "stork",
    "egret",
    "pigeon",
    "crow",
]
NUM_CLASSES = 7


class AirportBirdDataset(Dataset):
    """Dataset for airport bird species classification (7 classes)"""

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        #STRONG DATA AUGMENTATION for training
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(split)
        self.split = split

        #Load class mappings
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(CLASS_NAMES)
        }
        self.idx_to_class = {
            idx: class_name for idx, class_name in enumerate(CLASS_NAMES)
        }

        #Load image paths and labels
        self.images = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        """Load images from train/ or val/ directory"""
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            print(f"Error: {self.split} directory not found at {split_dir}")
            return

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.jpg"):
                    if img_file.exists():
                        self.images.append(str(img_file))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        #Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_default_transform(self, split: str):
        """Default transforms for training/validation"""
        if split == "train":
            #STRONG DATA AUGMENTATION
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(25),
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2
                    ),
                    transforms.GaussianBlur(3),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )


class BirdClassifierTrainer:
    """Trainer for airport bird species classification models"""

    def __init__(
        self, model_type: str = "resnet", num_classes: int = 7, config: Dict = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.num_classes = num_classes
        self.config = config or self._get_default_config()

        #Initialize model
        if model_type == "resnet":
            self.model = ResNetBirdClassifier(num_classes)
        elif model_type == "mobilenet":
            self.model = MobileNetBirdClassifier(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)

        #Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get("scheduler_step", 7),
            gamma=self.config.get("scheduler_gamma", 0.1),
        )

    def _get_default_config(self) -> Dict:
        """Default training configuration"""
        return {
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "scheduler_step": 7,
            "scheduler_gamma": 0.1,
            "batch_size": 32,
            "num_epochs": 15,
            "early_stopping_patience": 5,
            "save_best_only": True,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        save_path: str = "models/",
        model_name: str = None,
    ):
        """Train the model with improved monitoring and early stopping"""
        num_epochs = num_epochs or self.config.get("num_epochs", 15)
        model_name = model_name or f"{self.model_type}_airport7"

        print(f"Training {self.model_type} model for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Number of classes: {self.num_classes}")

        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = self.config.get("early_stopping_patience", 5)

        training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(num_epochs):
            #Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )

            #Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

            #Calculate accuracies
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total

            #Store history
            training_history["train_loss"].append(train_loss / len(train_loader))
            training_history["train_acc"].append(train_acc)
            training_history["val_loss"].append(val_loss / len(val_loader))
            training_history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(
                f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
            )
            print(
                f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%"
            )

            #Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                os.makedirs(save_path, exist_ok=True)

                #Save model state
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_path, f"{model_name}.pth"),
                )

                #Save training config
                config_save_path = os.path.join(save_path, f"{model_name}_config.json")
                with open(config_save_path, "w") as f:
                    json.dump(
                        {
                            "model_type": self.model_type,
                            "num_classes": self.num_classes,
                            "best_val_acc": best_val_acc,
                            "training_config": self.config,
                            "training_history": training_history,
                            "class_names": CLASS_NAMES,
                            "dataset": "airport7",
                        },
                        f,
                        indent=2,
                    )

                print(f"  ✅ Saved best model with validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement for {patience_counter} epochs")

            #Early stopping
            if patience_counter >= early_stopping_patience:
                print(
                    f"  🛑 Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

            self.scheduler.step()

        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        return {"best_val_acc": best_val_acc, "training_history": training_history}


def create_data_loaders(
    data_dir: str, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders for airport7 dataset"""
    train_dataset = AirportBirdDataset(data_dir, split="train")
    val_dataset = AirportBirdDataset(data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_ensemble_models(
    data_dir: str = "data/airport7/", save_path: str = "models/", config: Dict = None
) -> Dict[str, float]:
    """Train ensemble of ResNet18 and MobileNetV2 models"""
    print(f"Training ensemble models on {data_dir}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")

    train_loader, val_loader = create_data_loaders(
        data_dir, config.get("batch_size", 32)
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    #Train ResNet18
    print("\n" + "=" * 50)
    print("Training ResNet18...")
    resnet_trainer = BirdClassifierTrainer("resnet", NUM_CLASSES, config)
    resnet_results = resnet_trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.get("num_epochs", 15),
        save_path=save_path,
        model_name="resnet18_airport7",
    )

    #Train MobileNetV2
    print("\n" + "=" * 50)
    print("Training MobileNetV2...")
    mobilenet_trainer = BirdClassifierTrainer("mobilenet", NUM_CLASSES, config)
    mobilenet_results = mobilenet_trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.get("num_epochs", 15),
        save_path=save_path,
        model_name="mobilenetv2_airport7",
    )

    #Save ensemble configuration
    ensemble_config = {
        "models": ["resnet18_airport7", "mobilenetv2_airport7"],
        "accuracies": {
            "resnet18": resnet_results["best_val_acc"],
            "mobilenetv2": mobilenet_results["best_val_acc"],
        },
        "ensemble_method": "average",
        "airport_birds": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "dataset": "airport7",
    }

    with open(f"{save_path}/ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print("\n" + "=" * 50)
    print("ENSEMBLE TRAINING COMPLETED")
    print("=" * 50)
    print(f"ResNet18 Best Validation Accuracy: {resnet_results['best_val_acc']:.4f}")
    print(
        f"MobileNetV2 Best Validation Accuracy: {mobilenet_results['best_val_acc']:.4f}"
    )
    print(
        f"Expected Ensemble Accuracy: {(resnet_results['best_val_acc'] + mobilenet_results['best_val_acc']) / 2:.4f}"
    )
    print(f"Models saved in: {save_path}")
    print(f"Ensemble config saved: {save_path}/ensemble_config.json")

    return {
        "resnet18": resnet_results["best_val_acc"],
        "mobilenetv2": mobilenet_results["best_val_acc"],
    }


def main():
    """Main training function for airport7 dataset"""
    print("Airport Bird Species Classification Training")
    print("=" * 50)
    print("Using Airport7 Dataset (7 classes)")

    #Configuration for training
    config = {
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "scheduler_step": 7,
        "scheduler_gamma": 0.1,
        "batch_size": 32,
        "num_epochs": 15,
        "early_stopping_patience": 5,
        "save_best_only": True,
    }

    #Train ensemble models
    results = train_ensemble_models(
        data_dir="data/airport7/", save_path="models/", config=config
    )

    if results:
        print("\n✅ Training completed successfully!")
        print("You can now use the ensemble models in your system.")
        print("\nTo use in your system:")
        print("1. Set use_ensemble=True in SpeciesClassifier")
        print("2. The system will automatically load both models")
        print("3. Use ensemble predictions for best accuracy")
        print(
            f"\nNote: Models are trained on Airport7 dataset with {NUM_CLASSES} classes"
        )
        print(f"Airport birds included: {', '.join(CLASS_NAMES)}")
    else:
        print("\n❌ Training failed. Please check your airport7 dataset setup.")
        print(
            "Make sure data/airport7/ contains train/ and val/ directories with your 7 classes."
        )


if __name__ == "__main__":
    main()

