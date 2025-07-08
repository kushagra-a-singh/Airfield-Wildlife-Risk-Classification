# Technical Implementation Guide

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Bird Detector  │───▶│  Classifier     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Risk Calculator│◀───│  Risk Assessment│◀───│  Enhanced Data  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│  API Endpoints  │◀───│  Alert System   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Ensemble Model System

### Model Selection Options

#### 1. Ensemble Mode (Default)
- **Usage**: `model_selection="ensemble"`
- **Models**: ResNet18 + MobileNetV2
- **Advantage**: Best accuracy through model combination
- **Performance**: Balanced speed and accuracy
- **Best for**: Production deployment, critical applications

#### 2. ResNet18 Only
- **Usage**: `model_selection="resnet18"`
- **Models**: ResNet18 only
- **Advantage**: Highest accuracy
- **Performance**: Slower but most accurate
- **Best for**: When accuracy is paramount, offline processing

#### 3. MobileNetV2 Only
- **Usage**: `model_selection="mobilenetv2"`
- **Models**: MobileNetV2 only
- **Advantage**: Fastest inference
- **Performance**: Fastest speed
- **Best for**: Real-time applications, resource-constrained environments

### Performance Characteristics

#### Speed Comparison (approximate)
- **MobileNetV2**: ~30 FPS
- **Ensemble**: ~15 FPS
- **ResNet18**: ~10 FPS

#### Accuracy Comparison (approximate)
- **Ensemble**: 95%+
- **ResNet18**: 93%+
- **MobileNetV2**: 90%+

## Dataset Management

### CUB-200-2011 Dataset
- **Source**: Caltech-UCSD Birds-200-2011
- **Location**: `data/cub_200_2011/CUB_200_2011/`
- **Size**: ~1.1GB
- **Images**: 11,788 images
- **Species**: 200 bird species
- **Files**:
  - `images/` - All bird images
  - `classes.txt` - Species labels
  - `bounding_boxes.txt` - Detection annotations
  - `image_class_labels.txt` - Classification labels
  - `train_test_split.txt` - Training/validation split
  - `attributes/` - Bird attribute annotations
  - `parts/` - Part-based annotations

### Integrated Airport Birds Dataset
- **Location**: `data/integrated_birds/`
- **Images**: 640+ airport images, 11,788+ CUB-200-2011 images
- **Species**: 7 airport-relevant species + 200 CUB-200-2011 species
- **Structure**:
  ```
  data/integrated_birds/
  ├── train/
  │   ├── black_kite/
  │   ├── brahminy_kite/
  │   ├── egret/
  │   ├── pigeon/
  │   ├── crow/
  │   ├── red_kite/
  │   ├── white_tailed_kite/
  │   ├── 001.Black_footed_Albatross/
  │   ├── ... (other CUB-200-2011 species folders)
  └── val/
      ├── black_kite/
      ├── ... (other species)
      ├── 001.Black_footed_Albatross/
      └── ...
  ```

### Airport Bird Classes
- **Categories**:
  - **Kites** (High Risk): black_kite, brahminy_kite, red_kite
  - **Raptors** (High Risk): eagle, hawk, falcon, vulture
  - **Waterfowl** (Medium Risk): cormorant, stork, egret, heron, duck, goose
  - **Small Birds** (Low Risk): sparrow, finch, starling, pigeon
- **Note:** Class information is inferred from folder names in `integrated_birds/train/` and `integrated_birds/val/`. The file `data/classes/airport_birds.json` does not exist.

## 🔧 Configuration Options

### SystemConfig Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `model_selection` | ensemble, resnet18, mobilenetv2 | ensemble | Which models to use |
| `detection_mode` | fast, balanced, accurate | balanced | Detection performance mode |
| `max_frames` | integer | 100 | Maximum frames to process |
| `save_outputs` | true/false | true | Save processed outputs |
| `output_dir` | string | output_improved | Output directory |
| `performance_tracking` | true/false | true | Track performance metrics |
| `visualization` | true/false | true | Generate visualizations |

### Example Configurations

#### High Accuracy Configuration
```json
{
  "model_selection": "resnet18",
  "detection_mode": "accurate",
  "max_frames": 200,
  "save_outputs": true,
  "performance_tracking": true
}
```

#### Fast Processing Configuration
```json
{
  "model_selection": "mobilenetv2",
  "detection_mode": "fast",
  "max_frames": 50,
  "save_outputs": false,
  "performance_tracking": true
}
```

#### Production Configuration
```json
{
  "model_selection": "ensemble",
  "detection_mode": "balanced",
  "max_frames": 100,
  "save_outputs": true,
  "performance_tracking": true
}
```

## Training Your Models

### Step 1: Prepare Your Data
```
data/bird_species/
├── train/
│   ├── black_kite/
│   ├── brahminy_kite/
│   ├── cormorant/
│   ├── stork/
│   └── egret/
└── val/
    ├── black_kite/
    ├── brahminy_kite/
    ├── cormorant/
    ├── stork/
    └── egret/
```

### Step 2: Train Both Models
```bash
python train_bird_classifier.py
```

This will:
- Train ResNet18 model
- Train MobileNetV2 model
- Save models to `models/` directory
- Create ensemble configuration

### Step 3: Verify Training
Check the `models/` directory for:
- `resnet18_bird_classifier.pth`
- `mobilenetv2_bird_classifier.pth`
- `ensemble_config.json`

## Testing Commands

### Quick Test
```bash
python test_system.py
# Select option 1 for quick test
```

### Model Comparison
```bash
python test_system.py
# Select option 2 for model comparison
```

### Performance Benchmark
```bash
python test_system.py
# Select option 3 for performance benchmark
```

### Custom Configuration
```bash
python test_system.py
# Select option 4 for custom configuration
```

## Risk Assessment Model

### Risk Factors (Weighted)
- **Size (25%)**: Large birds pose higher collision risk
- **Altitude (20%)**: Lower altitude = higher risk
- **Speed (20%)**: Higher speed = higher risk
- **Behavior (15%)**: Diving/soaring behaviors increase risk
- **Species (20%)**: Species-specific risk factors

### Risk Levels
- **Low (0-3)**: Minimal risk, routine monitoring
- **Moderate (3-6)**: Elevated risk, increased attention
- **High (6-10)**: Critical risk, immediate action required

### Custom Risk Assessment
```python
# Custom risk calculation
risk_factors = {
    "size": "large",
    "altitude": 50,
    "speed": 25,
    "behavior": "soaring",
    "species": "black_kite"
}

risk_score = risk_calc.calculate_custom_risk(risk_factors)
```

## Detection Capabilities

### Multi-Model Ensemble
- **YOLOv8 Nano**: Fast detection for real-time processing
- **YOLOv8 Small**: Balanced accuracy and speed
- **YOLOv8 Medium**: High accuracy for critical detections

### Detection Features
- **Bounding Box Detection**: Precise bird localization
- **Confidence Scoring**: Detection reliability assessment
- **Size Estimation**: Bird size category classification
- **Trajectory Tracking**: Movement pattern analysis

## Performance Metrics

### System Performance
- **FPS**: 25-30 frames per second (real-time)
- **Detection Rate**: 0.8-1.2 detections per frame
- **Accuracy**: 85-90% for target species
- **Latency**: <100ms end-to-end processing

### Risk Assessment
- **Risk Score Range**: 0-10 scale
- **Collision Probability**: 0-100% estimation
- **Alert Response Time**: <2 seconds
- **False Positive Rate**: <5%

## Advanced Usage

### Using the System

#### Basic Usage
```python
from src.classification.species_classifier import SpeciesClassifier

# Use ensemble (default)
classifier = SpeciesClassifier(model_selection="ensemble")

# Use only ResNet18
classifier = SpeciesClassifier(model_selection="resnet18")

# Use only MobileNetV2
classifier = SpeciesClassifier(model_selection="mobilenetv2")
```

#### Testing with Configuration
```python
from test_system import SystemTester, SystemConfig

# Create configuration
config = SystemConfig({
    "model_selection": "ensemble",
    "detection_mode": "balanced",
    "max_frames": 100
})

# Initialize tester
tester = SystemTester(config)

# Test with video
result = tester.test_with_config("path/to/video.mp4")
```

#### Model Comparison
```python
# Compare all models
results = tester.test_model_comparison("path/to/video.mp4", max_frames=30)
```

#### Performance Benchmark
```python
# Benchmark different configurations
results = tester.test_performance_benchmark("path/to/video.mp4", max_frames=50)
```

### Model Information
```python
# Check model status
model_info = classifier.get_model_info()
print(model_info)

# Change model selection
classifier.set_model_selection("mobilenetv2")
```

## 🔧 Configuration Options

### Model Configuration
```python
# Use enhanced detector with ensemble
from src.detection.enhanced_bird_detector import EnhancedBirdDetector
detector = EnhancedBirdDetector(
    confidence_threshold=0.5,
    use_ensemble=True
)

# Use basic detector for speed
from src.detection.bird_detector import BirdDetector
detector = BirdDetector(
    confidence_threshold=0.3
)
```

### Risk Assessment Configuration
```python
# Customize risk factors
from src.risk_assessment.risk_calculator import RiskCalculator
risk_calc = RiskCalculator()

# Adjust risk factors for your airport
risk_calc.size_factors['large'] = 4.0  # Increase large bird risk
risk_calc.behavior_factors['diving'] = 3.0  # Increase diving risk
```

## Troubleshooting

### Common Issues

1. **Models not found**
   - Ensure you've run `train_bird_classifier.py`
   - Check that models are in `models/` directory

2. **Poor accuracy**
   - Verify training data quality
   - Try ensemble mode for better accuracy
   - Check model accuracies in `ensemble_config.json`

3. **Slow performance**
   - Use MobileNetV2 for speed
   - Reduce `max_frames`
   - Set `detection_mode` to "fast"

4. **Memory issues**
   - Reduce batch size in training
   - Use MobileNetV2 instead of ensemble
   - Process fewer frames

### Performance Tips
- **GPU**: 3x faster processing with CUDA
- **Ensemble**: Better accuracy with multiple models
- **Confidence**: Adjust thresholds based on use case
- **Resolution**: Lower resolution for speed

## Dataset Integration

### Data Preprocessing
```python
# Convert datasets to YOLO format
def convert_cub_to_yolo(cub_path, output_dir):
    """Convert CUB-200-2011 annotations to YOLO format"""
    # Load CUB data
    images_df = pd.read_csv(cub_path / "images.txt", sep=' ', header=None)
    bbox_df = pd.read_csv(cub_path / "bounding_boxes.txt", sep=' ', header=None)
    
    # Convert bounding boxes to YOLO format
    for _, row in images_df.iterrows():
        image_id = row[0]
        filepath = row[1]
        
        # Get bounding box for this image
        bbox = bbox_df[bbox_df[0] == image_id].iloc[0]
        x, y, width, height = bbox[1], bbox[2], bbox[3], bbox[4]
        
        # Convert to YOLO format (normalized coordinates)
        # Save annotations and images
```

### Data Augmentation
```python
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])
```

## Best Practices

1. **For Production**: Use ensemble mode with balanced detection
2. **For Real-time**: Use MobileNetV2 with fast detection
3. **For Analysis**: Use ResNet18 with accurate detection
4. **For Testing**: Use model comparison to find best fit

## API Examples

### Python API
```python
from src.detection.bird_detector import BirdDetector
from src.classification.species_classifier import SpeciesClassifier
from src.risk_assessment.risk_calculator import RiskCalculator

# Initialize components
detector = BirdDetector()
classifier = SpeciesClassifier()
risk_calc = RiskCalculator()

# Process frame
detections = detector.detect_birds_in_frame(frame)
classifications = classifier.classify_birds_in_detections(detections, frame)
risks = risk_calc.calculate_risks(classifications)

# Get results
for risk in risks:
    print(f"Species: {risk.species}")
    print(f"Risk Level: {risk.risk_level}")
    print(f"Risk Score: {risk.risk_score}")
```

## API and Dashboard

### Dashboard & File Upload
- `GET /` — Main dashboard (web interface)
- `POST /upload` — Upload image or video for processing (used by dashboard)

### Streaming
- `GET /stream` — Real-time video stream with detection overlays

### API Endpoints
- `POST /api/detect` — Single image detection and risk assessment
- `GET /api/risk_trend` — Risk trend for current session
- `GET /api/session_summary` — Session summary (species distribution, risk levels)
- `GET /data/sample_videos/<filename>` — Download uploaded sample videos

> **Note:** The main workflow, dashboard and API are focused on the Airport7 dataset (7 classes: black_kite, brahminy_kite, cormorant, stork, egret, pigeon, crow). Fine-grained classification with CUB-200-2011 is available for research but not the default. 

## Running the Application

For local development:
```bash
python app.py
```
This will start the Flask development server at http://localhost:5000.

For production with better stream handling:
```bash
waitress-serve --host=0.0.0.0 --port=5000 app:app
```
This uses Waitress for a more robust production server. 
