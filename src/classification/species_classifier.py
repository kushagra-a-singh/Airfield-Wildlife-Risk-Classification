"""
Species Classifier Module
Comprehensive bird species classification for airport bird strike prevention
Optimized for 7 airport bird species
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not available for classification")


@dataclass
class BirdClassification:
    """Data class for bird classification results"""

    species: str
    confidence: float
    size_category: str  #small, medium, large
    behavior: str  #flying, perching, diving, soaring
    altitude_estimate: float  #meters
    speed_estimate: float  #m/s
    wing_span_estimate: float  #meters
    scientific_name: str
    family: str
    risk_category: str  #low, medium, high


class ResNetBirdClassifier(nn.Module):
    """ResNet18-based bird species classifier for airport birds"""

    def __init__(self, num_classes: int = 7):
        super(ResNetBirdClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


class MobileNetBirdClassifier(nn.Module):
    """MobileNetV2-based bird species classifier for airport birds"""

    def __init__(self, num_classes: int = 7):
        super(MobileNetBirdClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.mobilenet(x)


class SpeciesClassifier:
    """Main species classifier for airport bird detection and classification"""

    def __init__(self, model_path: str = None, use_ensemble: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ensemble = use_ensemble

        self.class_names = [
            "black_kite",
            "brahminy_kite",
            "cormorant",
            "stork",
            "egret",
            "pigeon",
            "crow",
        ]

        self.airport_birds = {
            "black_kite": "high",
            "brahminy_kite": "high",
            "cormorant": "medium",
            "stork": "medium",
            "egret": "medium",
            "pigeon": "low",
            "crow": "low",
        }

        if model_path:
            self._load_models(model_path)
        else:
            self._load_default_models()

    def _load_default_models(self):
        """Load default trained models"""
        try:
            resnet_path = "models/resnet18_airport7.pth"
            mobilenet_path = "models/mobilenetv2_airport7.pth"

            if os.path.exists(resnet_path) and os.path.exists(mobilenet_path):
                self.resnet_model = ResNetBirdClassifier(num_classes=7)
                self.mobilenet_model = MobileNetBirdClassifier(num_classes=7)

                self.resnet_model.load_state_dict(
                    torch.load(resnet_path, map_location=self.device)
                )
                self.mobilenet_model.load_state_dict(
                    torch.load(mobilenet_path, map_location=self.device)
                )

                self.resnet_model.to(self.device)
                self.mobilenet_model.to(self.device)
                self.resnet_model.eval()
                self.mobilenet_model.eval()

                print("✅ Loaded ensemble models (ResNet18 + MobileNetV2)")
                print(f"   Dataset: Airport7 (7 classes)")
                print(f"   Classes: {', '.join(self.class_names)}")
            else:
                print("⚠️  Ensemble models not found. Using single model.")
                self.use_ensemble = False
                self._load_single_model()

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self._load_single_model()

    def _load_single_model(self):
        """Load a single model as fallback"""
        try:
            model_path = "models/resnet18_airport7.pth"
            if os.path.exists(model_path):
                self.model = ResNetBirdClassifier(num_classes=7)
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                print("✅ Loaded ResNet18 model")
            else:
                model_path = "models/mobilenetv2_airport7.pth"
                if os.path.exists(model_path):
                    self.model = MobileNetBirdClassifier(num_classes=7)
                    self.model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    print("✅ Loaded MobileNetV2 model")
                else:
                    print("❌ No trained models found. Please train models first.")
                    print("Run: python train_bird_classifier.py")
                    return

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"❌ Error loading single model: {e}")

    def predict(self, image: np.ndarray) -> Dict:
        """Predict bird species from image"""
        if not hasattr(self, "model") and not hasattr(self, "resnet_model"):
            return {"error": "No models loaded"}

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        try:
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            if self.use_ensemble and hasattr(self, "resnet_model"):
                #Ensemble prediction
                with torch.no_grad():
                    resnet_output = self.resnet_model(image_tensor)
                    mobilenet_output = self.mobilenet_model(image_tensor)

                    #Average predictions
                    ensemble_output = (resnet_output + mobilenet_output) / 2
                    probabilities = torch.softmax(ensemble_output, dim=1)

                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                    species = self.class_names[predicted_class]
                    risk_level = self.airport_birds.get(species, "unknown")

                    return {
                        "species": species,
                        "confidence": confidence,
                        "risk_level": risk_level,
                        "method": "ensemble",
                    }
            else:
                #Single model prediction
                with torch.no_grad():
                    output = self.model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)

                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                    species = self.class_names[predicted_class]
                    risk_level = self.airport_birds.get(species, "unknown")

                    return {
                        "species": species,
                        "confidence": confidence,
                        "risk_level": risk_level,
                        "method": "single",
                    }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def get_risk_assessment(self, species: str, confidence: float) -> str:
        """Get risk assessment for airport bird detection"""
        base_risk = self.airport_birds.get(species, "unknown")

        if confidence < 0.5:
            return "low" 
        elif confidence < 0.7:
            return "medium"
        else:
            return base_risk

    def classify_birds_in_detections(self, detections, frame):
        """Classify multiple birds in detections"""
        classifications = []

        for detection in detections:
            #Extract bird region from frame using detection bbox
            x1, y1, x2, y2 = detection.bbox
            bird_region = frame[int(y1) : int(y2), int(x1) : int(x2)]

            if bird_region.size > 0:  #Check if region is valid
                try:
                    #Classify the bird region
                    result = self.predict(bird_region)

                    if "error" not in result:
                        #Create classification object
                        classification = BirdClassification(
                            species=result["species"],
                            confidence=result["confidence"],
                            size_category=self._estimate_size(detection.bbox),
                            behavior=self._estimate_behavior(result["species"]),
                            altitude_estimate=self._estimate_altitude(detection.bbox),
                            speed_estimate=self._estimate_speed(result["species"]),
                            wing_span_estimate=self._estimate_wing_span(
                                result["species"]
                            ),
                            scientific_name=self._get_scientific_name(
                                result["species"]
                            ),
                            family=self._get_family(result["species"]),
                            risk_category=result["risk_level"],
                        )

                        classifications.append(classification)

                except Exception as e:
                    print(f"Error classifying bird detection: {e}")
                    continue

        return classifications

    def _estimate_size(self, bbox):
        """Estimate bird size based on bounding box"""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if area < 1000:
            return "small"
        elif area < 5000:
            return "medium"
        else:
            return "large"

    def _estimate_behavior(self, species):
        """Estimate bird behavior based on species"""
        soaring_birds = ["black_kite", "brahminy_kite"]
        diving_birds = ["cormorant", "egret"]
        perching_birds = ["pigeon", "crow"]

        if species in soaring_birds:
            return "soaring"
        elif species in diving_birds:
            return "diving"
        elif species in perching_birds:
            return "perching"
        else:
            return "flying"

    def _estimate_altitude(self, bbox):
        """Estimate bird altitude based on position in frame"""
        x1, y1, x2, y2 = bbox
        #Higher in frame = lower altitude
        frame_height = 480  
        relative_height = (y1 + y2) / 2 / frame_height

        if relative_height < 0.3:
            return 100.0  #High altitude
        elif relative_height < 0.7:
            return 50.0  #Medium altitude
        else:
            return 20.0  #Low altitude

    def _estimate_speed(self, species):
        """Estimate bird speed based on species"""
        fast_birds = ["black_kite", "brahminy_kite"]
        medium_birds = ["cormorant", "egret", "stork"]
        slow_birds = ["pigeon", "crow"]

        if species in fast_birds:
            return 15.0
        elif species in medium_birds:
            return 10.0  
        elif species in slow_birds:
            return 5.0  
        else:
            return 8.0  

    def _estimate_wing_span(self, species):
        """Estimate bird wing span based on species"""
        large_birds = ["black_kite", "brahminy_kite", "stork"]
        medium_birds = ["cormorant", "egret"]
        small_birds = ["pigeon", "crow"]

        if species in large_birds:
            return 1.5  
        elif species in medium_birds:
            return 1.0  
        elif species in small_birds:
            return 0.5  
        else:
            return 0.8 

    def _get_scientific_name(self, species):
        """Get scientific name for species"""
        scientific_names = {
            "black_kite": "Milvus migrans",
            "brahminy_kite": "Haliastur indus",
            "cormorant": "Phalacrocorax carbo",
            "stork": "Ciconia ciconia",
            "egret": "Ardea alba",
            "pigeon": "Columba livia",
            "crow": "Corvus brachyrhynchos",
        }
        return scientific_names.get(species, "Unknown")

    def _get_family(self, species):
        """Get bird family for species"""
        families = {
            "black_kite": "Accipitridae",
            "brahminy_kite": "Accipitridae",
            "cormorant": "Phalacrocoracidae",
            "stork": "Ciconiidae",
            "egret": "Ardeidae",
            "pigeon": "Columbidae",
            "crow": "Corvidae",
        }
        return families.get(species, "Unknown")

    def get_model_info(self):
        """Get information about available models and their performance"""
        model_info = {
            "available_models": ["ensemble", "resnet18", "mobilenetv2"],
            "current_model": "ensemble" if self.use_ensemble else "single",
            "num_classes": 7,
            "dataset": "Airport7 (7 Airport Birds)",
            "class_names": self.class_names,
        }

        ensemble_config_path = "models/ensemble_config.json"
        if os.path.exists(ensemble_config_path):
            try:
                with open(ensemble_config_path, "r") as f:
                    config = json.load(f)
                    if "accuracies" in config:
                        model_info["model_accuracies"] = config["accuracies"]
            except Exception as e:
                print(f"Warning: Could not load model accuracies: {e}")

        return model_info
