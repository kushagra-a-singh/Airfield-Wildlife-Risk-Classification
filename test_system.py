"""
Comprehensive Testing Script for Bird Detection System
Tests the system with sample videos/images and provides detailed analytics
Optimized for Airport7 dataset(7 airport bird species)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/yolov8x.pt"

AIRPORT7_CLASSES = [
    "black_kite",
    "brahminy_kite",
    "cormorant",
    "stork",
    "egret",
    "pigeon",
    "crow",
]


def ensure_yolo_model():
    if not os.path.exists(MODEL_PATH):
        print(f"YOLOv8 model not found at {MODEL_PATH}. Downloading...")
        try:
            from src.utils.model_downloader import download_yolov8_weights

            download_yolov8_weights(model_name="yolov8x.pt", dest_dir="models")
        except ImportError:
            print("Warning: model_downloader not available")
    else:
        print(f"YOLOv8 model found at {MODEL_PATH}.")


ensure_yolo_model()

from src.classification.species_classifier import SpeciesClassifier
from src.detection.enhanced_bird_detector import EnhancedBirdDetector
from src.risk_assessment.risk_calculator import RiskCalculator


class SystemConfig:
    """Configuration class for system testing"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = config_dict or self._get_default_config()

    def _get_default_config(self):
        """Get default configuration for testing"""
        return {
            "model_type": "ensemble",  #ensemble, resnet18, mobilenetv2
            "detection_mode": "balanced",  #fast, balanced, accurate
            "max_frames": 100,
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "use_integrated_dataset": False, 
            "airport_birds": [
                "black_kite",
                "brahminy_kite",
                "cormorant",
                "stork",
                "egret",
                "pigeon",
                "crow",
            ],
            "num_classes": 7,  
            "dataset": "airport7",
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value

    def save_config(self, path: str = "test_config.json"):
        """Save configuration to file"""
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {path}")

    def load_config(self, path: str = "test_config.json"):
        """Load configuration from file"""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.config.update(json.load(f))
            print(f"Configuration loaded from {path}")
        else:
            print(f"Configuration file {path} not found, using defaults")


class SystemTester:
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        print("Initializing system components for airport7 dataset...")

        self.detector = EnhancedBirdDetector(use_ensemble=False)  
        self.classifier = SpeciesClassifier(
            use_ensemble=True
        )  
        self.risk_calc = RiskCalculator()
        self.results = {}

        self.detector.set_performance_mode(
            self.config.get("detection_mode", "balanced")
        )

        print("✅ System components initialized successfully")
        print(f"   Model type: {self.config.get('model_type')}")
        print(f"   Detection mode: {self.config.get('detection_mode')}")
        print(f"   Dataset: Airport7 (7 classes)")
        print(f"   Airport Birds: {', '.join(self.config.get('airport_birds'))}")

        model_info = self.classifier.get_model_info()
        print(f"   Available models: {model_info['available_models']}")
        if "model_accuracies" in model_info:
            print(f"   Model accuracies: {model_info['model_accuracies']}")

    def test_airport_bird_detection(self):
        """Test airport bird detection with airport7 dataset"""
        print("\n" + "=" * 60)
        print("🔍 AIRPORT BIRD DETECTION TEST")
        print("=" * 60)

        
        test_images = []

        airport7_dir = Path("data/airport7/")
        if airport7_dir.exists():
            for split in ["train", "val"]:
                split_dir = airport7_dir / split
                if split_dir.exists():
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            
                            images = list(class_dir.glob("*.jpg"))[:2]
                            test_images.extend(images)

        if not test_images:
            print("❌ No test images found in airport7 dataset")
            print(
                "Please ensure data/airport7/train/ and data/airport7/val/ contain images"
            )
            return

        print(f"📁 Found {len(test_images)} test images from airport7 dataset")
        print("🔄 Processing images...")

        results = []
        for i, img_path in enumerate(test_images[:10]):  
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                detections = self.detector.detect_birds_in_frame(image, i)

                if detections:
                    for detection in detections:
                        x1, y1, x2, y2 = detection.bbox
                        bird_region = image[int(y1) : int(y2), int(x1) : int(x2)]

                        if bird_region.size > 0:
                            classification = self.classifier.predict(bird_region)

                            if "error" not in classification:
                                results.append(
                                    {
                                        "image": str(img_path),
                                        "detection": detection,
                                        "classification": classification,
                                    }
                                )

                print(f"  ✅ Processed: {img_path.name}")

            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")

        if results:
            self._generate_airport_test_report(results)
            print(f"\n✅ Airport bird detection test completed!")
            print(f"📊 Processed {len(results)} detections")

            print("\n📊 AIRPORT BIRD CLASSIFICATION RESULTS:")
            print("=" * 50)
            species_counts = {}
            for result in results:
                species = result["classification"]["species"]
                confidence = result["classification"]["confidence"]
                risk_level = result["classification"]["risk_level"]

                if species not in species_counts:
                    species_counts[species] = {
                        "count": 0,
                        "confidences": [],
                        "risk_levels": [],
                    }

                species_counts[species]["count"] += 1
                species_counts[species]["confidences"].append(confidence)
                species_counts[species]["risk_levels"].append(risk_level)

            for species, data in species_counts.items():
                avg_confidence = sum(data["confidences"]) / len(data["confidences"])
                most_common_risk = max(
                    set(data["risk_levels"]), key=data["risk_levels"].count
                )

                risk_emoji = (
                    "🔴"
                    if most_common_risk == "high"
                    else "🟡" if most_common_risk == "medium" else "🟢"
                )

                print(f"  {risk_emoji} {species.upper()}:")
                print(f"    📈 Count: {data['count']} detections")
                print(f"    🎯 Avg Confidence: {avg_confidence:.2f}")
                print(f"    ⚠️  Risk Level: {most_common_risk.upper()}")
                print()
        else:
            print("❌ No successful detections/classifications found")

    def test_video_detection(self):
        """Test video detection using the configured video path"""
        video_path = self.config.get("video_path")
        max_frames = self.config.get("max_frames", 100)

        if not video_path:
            print("❌ No video path configured")
            return None

        return self.test_with_video(video_path, max_frames)

    def test_with_video(self, video_path: str, max_frames: int = 100):
        """Test the system with a video file"""
        print(f"Testing with video: {video_path}")

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        all_detections = []
        all_classifications = []
        all_risks = []
        start_time = time.time()

        print(f"Processing up to {max_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            try:
                detections = self.detector.detect_birds_in_frame(frame, frame_count)
                classifications = self.classifier.classify_birds_in_detections(
                    detections, frame
                )
                risks = self.risk_calc.calculate_risks(classifications)

                all_detections.extend(detections)
                all_classifications.extend(classifications)
                all_risks.extend(risks)

                frame_count += 1

                if frame_count % 10 == 0:
                    print(".", end="", flush=True)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                frame_count += 1
                continue

        cap.release()
        print()  
        total_time = time.time() - start_time
        detection_stats = self.detector.get_detection_statistics()

        #Calculate classification stats
        classification_stats = {
            "total_classifications": len(all_classifications),
            "species_distribution": {},
            "size_distribution": {"small": 0, "medium": 0, "large": 0},
            "behavior_distribution": {},
        }

        for classification in all_classifications:
            # Species distribution
            species = classification.species
            if species not in classification_stats["species_distribution"]:
                classification_stats["species_distribution"][species] = 0
            classification_stats["species_distribution"][species] += 1

            #Size distribution
            size = classification.size_category
            if size in classification_stats["size_distribution"]:
                classification_stats["size_distribution"][size] += 1

            #behavior distribution
            behavior = classification.behavior
            if behavior not in classification_stats["behavior_distribution"]:
                classification_stats["behavior_distribution"][behavior] = 0
            classification_stats["behavior_distribution"][behavior] += 1

        #Calculate risk stats
        risk_stats = {
            "total_risks": len(all_risks),
            "risk_levels": {"low": 0, "moderate": 0, "high": 0},
            "average_risk_score": 0.0,
        }

        if all_risks:
            risk_scores = [r.risk_score for r in all_risks]
            risk_stats["average_risk_score"] = sum(risk_scores) / len(risk_scores)

            for risk in all_risks:
                level = risk.risk_level
                if level in risk_stats["risk_levels"]:
                    risk_stats["risk_levels"][level] += 1

        results = {
            "video_path": video_path,
            "frames_processed": frame_count,
            "total_detections": len(all_detections),
            "total_classifications": len(all_classifications),
            "total_risks": len(all_risks),
            "detection_stats": detection_stats,
            "classification_stats": classification_stats,
            "risk_stats": risk_stats,
            "performance": {
                "total_time": total_time,
                "average_fps": frame_count / total_time if total_time > 0 else 0,
                "effective_fps": detection_stats.get("effective_fps", 0),
                "frame_skip": detection_stats.get("frame_skip", 1),
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.results[video_path] = results
        return results

    def test_with_image(self, image_path: str):
        """Test the system with a single image"""
        print(f"Testing with image: {image_path}")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image: {image_path}")
            return None

        # Run pipeline
        try:
            detections = self.detector.detect_birds_in_frame(frame, 0)
            classifications = self.classifier.classify_birds_in_detections(
                detections, frame
            )
            risks = self.risk_calc.calculate_risks(classifications)

            # Draw results
            frame_with_boxes = self.detector.draw_detections(frame, detections)

            # Save annotated image
            output_path = f"test_output_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, frame_with_boxes)
            print(f"Annotated image saved as: {output_path}")

            results = {
                "image_path": image_path,
                "detections": len(detections),
                "classifications": len(classifications),
                "risks": len(risks),
                "detection_details": [
                    {
                        "bbox": d.bbox,
                        "confidence": d.confidence,
                        "class": d.class_name,
                        "id": d.id,
                        "frame_id": d.frame_id,
                        "timestamp": d.timestamp,
                    }
                    for d in detections
                ],
                "classification_details": [
                    {
                        "species": c.species,
                        "size": c.size_category,
                        "behavior": c.behavior,
                    }
                    for c in classifications
                ],
                "risk_details": [
                    {
                        "species": r.species,
                        "risk_level": r.risk_level,
                        "risk_score": r.risk_score,
                    }
                    for r in risks
                ],
                "output_image": output_path,
                "timestamp": datetime.now().isoformat(),
            }

            self.results[image_path] = results
            return results

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def generate_report(self, output_file: str = "test_report.json"):
        """Generate a comprehensive test report"""
        report = {
            "test_summary": {
                "total_tests": len(self.results),
                "test_timestamp": datetime.now().isoformat(),
                "system_version": "1.0.0",
                "dataset": "Airport7 (7 classes)",
            },
            "results": self.results,
            "overall_statistics": self._calculate_overall_stats(),
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Test report saved as: {output_file}")
        return report

    def _calculate_overall_stats(self):
        """Calculate overall statistics across all tests"""
        total_detections = 0
        total_classifications = 0
        total_risks = 0
        risk_levels = {"low": 0, "medium": 0, "high": 0}

        for result in self.results.values():
            if "total_detections" in result:
                total_detections += result["total_detections"]
            if "total_classifications" in result:
                total_classifications += result["total_classifications"]
            if "total_risks" in result:
                total_risks += result["total_risks"]

            if "risk_stats" in result:
                risk_stats = result["risk_stats"]
                for level in ["low", "medium", "high"]:
                    risk_levels[level] += risk_stats.get("risk_levels", {}).get(
                        level, 0
                    )

        return {
            "total_detections": total_detections,
            "total_classifications": total_classifications,
            "total_risks": total_risks,
            "risk_level_distribution": risk_levels,
        }

    def _generate_airport_test_report(self, results):
        """Generate airport-specific test report"""
        serializable_results = []
        for result in results:
            detection_dict = {
                "id": result["detection"].id,
                "bbox": result["detection"].bbox,
                "confidence": result["detection"].confidence,
                "class_id": result["detection"].class_id,
                "class_name": result["detection"].class_name,
                "frame_id": result["detection"].frame_id,
                "timestamp": result["detection"].timestamp,
            }

            serializable_result = {
                "image": result["image"],
                "detection": detection_dict,
                "classification": result["classification"],
            }
            serializable_results.append(serializable_result)

        report = {
            "test_type": "airport_bird_detection",
            "dataset": "airport7",
            "classes": AIRPORT7_CLASSES,
            "results": serializable_results,
            "timestamp": datetime.now().isoformat(),
        }

        report_path = "airport_test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Airport test report saved: {report_path}")

    def _generate_final_report(self, filename: str):
        """Generate final test report"""
        return self.generate_report(filename)


def main():
    """Main testing function with improved user interface"""
    while True:
        try:
            print("\n" + "=" * 60)
            print("🦅 BIRD DETECTION SYSTEM TESTING")
            print("=" * 60)
            print("Select testing mode:")
            print("1. Test with Video")
            print("2. Test with Image")
            print("3. Exit")

            mode_choice = input("\nEnter your choice (1-3): ").strip()

            if mode_choice == "3":
                print("Exiting. Goodbye! 👋")
                break
            elif mode_choice not in ["1", "2"]:
                print("❌ Invalid choice. Please select 1, 2, or 3.")
                continue

            if mode_choice == "1":
                file_type = "video"
                print("\n📹 VIDEO TESTING MODE")
                print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv")
            else:
                file_type = "image"
                print("\n🖼️  IMAGE TESTING MODE")
                print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")

            file_path = input(f"\nEnter the path to your {file_type} file: ").strip()

            if not file_path:
                print("❌ No file path provided.")
                continue

            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                print("Please check the file path and try again.")
                continue

            # Check file type
            file_ext = os.path.splitext(file_path)[1].lower()
            if mode_choice == "1":
                valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
                if file_ext not in valid_extensions:
                    print(f"❌ Unsupported video format: {file_ext}")
                    print(f"Supported formats: {', '.join(valid_extensions)}")
                    continue
            else:
                valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
                if file_ext not in valid_extensions:
                    print(f"❌ Unsupported image format: {file_ext}")
                    print(f"Supported formats: {', '.join(valid_extensions)}")
                    continue

            # Model selection
            print(f"\n✅ File validated: {file_path}")
            print("\n" + "=" * 50)
            print("🤖 SELECT MODEL")
            print("=" * 50)
            print("1. Ensemble (ResNet18 + MobileNetV2) - Best Accuracy")
            print("2. ResNet18 - High Accuracy, Medium Speed")
            print("3. MobileNetV2 - Good Accuracy, Fast Speed")
            print("4. Back to Main Menu")

            model_choice = input("\nEnter your choice (1-4): ").strip()

            if model_choice == "4":
                continue
            elif model_choice == "2":
                model = "resnet18"
            elif model_choice == "3":
                model = "mobilenetv2"
            elif model_choice == "1":
                model = "ensemble"
            else:
                print("❌ Invalid choice. Using Ensemble model.")
                model = "ensemble"

            # Parse command line arguments for compatibility
            try:
                parser = argparse.ArgumentParser(
                    description="Bird Detection System Testing"
                )
                parser.add_argument(
                    "--detection-mode",
                    choices=["fast", "balanced", "accurate"],
                    default="balanced",
                    help="Detection mode (default: balanced)",
                )
                parser.add_argument(
                    "--max-frames",
                    type=int,
                    default=100,
                    help="Maximum frames to process for videos (default: 100)",
                )

                args = parser.parse_args([])  # Empty list to avoid sys.argv conflicts
            except Exception as e:
                print(f"⚠️  Warning: Could not parse arguments, using defaults: {e}")
                args = type(
                    "Args", (), {"detection_mode": "balanced", "max_frames": 100}
                )()

            print(f"\n🚀 INITIALIZING SYSTEM")
            print(f"   Model: {model.upper()}")
            print(f"   File: {file_path}")
            print(f"   Detection Mode: {args.detection_mode}")
            if mode_choice == "1":
                print(f"   Max Frames: {args.max_frames}")

            # Load or create configuration
            try:
                config = SystemConfig()
                config.load_config()
            except Exception as e:
                print(f"⚠️  Warning: Could not load config, using defaults: {e}")
                config = SystemConfig()

            # Override config with selections
            config.set("model_type", model)
            config.set("detection_mode", args.detection_mode)
            config.set("max_frames", args.max_frames)

            # Create tester with configuration
            try:
                tester = SystemTester(config)
            except Exception as e:
                print(f"❌ Failed to initialize system: {e}")
                print(
                    "Please check if all required models and dependencies are available."
                )
                continue

            # Run airport bird detection test
            print("\n🔍 RUNNING AIRPORT BIRD DETECTION TEST")
            print("=" * 50)
            try:
                tester.test_airport_bird_detection()
                print("✅ Airport bird detection test completed")
            except Exception as e:
                print(f"❌ Airport bird detection test failed: {e}")
                print("Continuing with file test...")

            # Test the provided file
            print(f"\n🎯 TESTING YOUR {file_type.upper()}")
            print("=" * 50)

            try:
                if mode_choice == "1":
                    # Video testing
                    result = tester.test_with_video(file_path, args.max_frames)
                    if result:
                        print("✅ Video test completed")
                        # Show clean classification results
                        if "classification_stats" in result:
                            stats = result["classification_stats"]
                            if (
                                "species_distribution" in stats
                                and stats["species_distribution"]
                            ):
                                print("\n📊 DETECTION RESULTS:")
                                print("=" * 40)
                                for species, count in stats[
                                    "species_distribution"
                                ].items():
                                    print(f"  🦅 {species.upper()}: {count} detections")
                                print()
                                # Show performance stats
                                if "performance" in result:
                                    perf = result["performance"]
                                    print("⚡ PERFORMANCE STATS:")
                                    print("=" * 40)
                                    print(
                                        f"  Frames Processed: {result['frames_processed']}"
                                    )
                                    print(f"  Total Time: {perf['total_time']:.2f}s")
                                    print(f"  Average FPS: {perf['average_fps']:.1f}")
                                    print(
                                        f"  Total Detections: {result['total_detections']}"
                                    )
                                    print()
                        # Save annotated video with bounding boxes and species labels
                        try:
                            import cv2

                            from src.detection.enhanced_bird_detector import (
                                BirdDetection,
                            )

                            cap = cv2.VideoCapture(file_path)
                            frame_id = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret or frame_id >= args.max_frames:
                                    break
                                detections = tester.detector.detect_birds_in_frame(
                                    frame, frame_id
                                )
                                species_labels = {}
                                for det in detections:
                                    x1, y1, x2, y2 = det.bbox
                                    bird_region = frame[
                                        int(y1) : int(y2), int(x1) : int(x2)
                                    ]
                                    if bird_region.size > 0:
                                        classification = tester.classifier.predict(
                                            bird_region
                                        )
                                        if "species" in classification:
                                            species_labels[det.id] = classification[
                                                "species"
                                            ].upper()
                                frame_with_boxes = tester.detector.draw_detections(
                                    frame, detections, species_labels
                                )
                                # Save each annotated frame as an image instead of writing to a video
                                out_path = f"annotated_frames/{os.path.splitext(os.path.basename(file_path))[0]}_frame_{frame_id:04d}.jpg"
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                cv2.imwrite(out_path, frame_with_boxes)
                                frame_id += 1
                            cap.release()
                            print(f"🖼️ Annotated frames saved in: annotated_frames/")
                        except Exception as e:
                            print(f"⚠️  Could not save annotated frames: {e}")
                else:
                    # Image testing
                    result = tester.test_with_image(file_path)
                    if result:
                        print("✅ Image test completed")
                        # Draw bounding boxes and species labels
                        try:
                            import cv2

                            from src.detection.enhanced_bird_detector import (
                                BirdDetection,
                            )

                            frame = cv2.imread(file_path)
                            detections = tester.detector.detect_birds_in_frame(frame, 0)
                            species_labels = {}
                            for det in detections:
                                x1, y1, x2, y2 = det.bbox
                                bird_region = frame[
                                    int(y1) : int(y2), int(x1) : int(x2)
                                ]
                                if bird_region.size > 0:
                                    classification = tester.classifier.predict(
                                        bird_region
                                    )
                                    if "species" in classification:
                                        species_labels[det.id] = classification[
                                            "species"
                                        ].upper()
                            frame_with_boxes = tester.detector.draw_detections(
                                frame, detections, species_labels
                            )
                            out_path = f"annotated_{os.path.basename(file_path)}"
                            cv2.imwrite(out_path, frame_with_boxes)
                            print(f"📸 Annotated image saved as: {out_path}")
                        except Exception as e:
                            print(f"⚠️  Could not save annotated image: {e}")
                        if (
                            "classification_details" in result
                            and result["classification_details"]
                        ):
                            print("\n📊 DETECTION RESULTS:")
                            print("=" * 40)
                            for i, classification in enumerate(
                                result["classification_details"]
                            ):
                                print(f"  Detection {i+1}:")
                                print(
                                    f"    🦅 Species: {classification['species'].upper()}"
                                )
                                print(f"    📏 Size: {classification['size']}")
                                print(f"    🦋 Behavior: {classification['behavior']}")
                                print()
                        if "risk_details" in result and result["risk_details"]:
                            print("⚠️  RISK ASSESSMENT:")
                            print("=" * 40)
                            for i, risk in enumerate(result["risk_details"]):
                                risk_emoji = (
                                    "🔴"
                                    if risk["risk_level"] == "high"
                                    else (
                                        "🟡" if risk["risk_level"] == "medium" else "🟢"
                                    )
                                )
                                print(f"  Detection {i+1}:")
                                print(
                                    f"    {risk_emoji} Species: {risk['species'].upper()}"
                                )
                                print(
                                    f"    {risk_emoji} Risk Level: {risk['risk_level'].upper()}"
                                )
                                print(
                                    f"    {risk_emoji} Risk Score: {risk['risk_score']:.2f}"
                                )
                                print()
                    else:
                        print("❌ Image test failed or no detections found")

            except Exception as e:
                print(f"❌ Error during file testing: {e}")
                print("Please check if the file is valid and try again.")

            # Generate report
            print("📄 GENERATING REPORT")
            print("=" * 50)
            try:
                report_name = f"test_report_{model}.json"
                tester.generate_report(report_name)
                print(f"✅ Test report saved: {report_name}")
            except Exception as e:
                print(f"❌ Failed to generate report: {e}")

            print("\n" + "=" * 60)
            print("✅ Test completed successfully!")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Returning to main menu...")
            continue
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Returning to main menu...")
            continue


if __name__ == "__main__":
    main()
