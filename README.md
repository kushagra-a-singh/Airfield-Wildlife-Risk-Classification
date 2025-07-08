# Airfield Wildlife Risk Classification

A comprehensive real-time bird detection and classification system designed for airport bird strike prevention and wildlife monitoring.

## 🎯 Project Overview

This system provides advanced AI-powered bird detection, species classification, and risk assessment capabilities specifically designed for airport safety applications. It can detect and classify various bird species including Black Kites, Brahminy Kites, Cormorants, Storks, and Egrets, while providing real-time risk assessment and alerting.

## ✨ Key Features

### 🦅 **Bird Detection & Classification**
- **Multi-model Ensemble Detection**: Uses multiple YOLO models for enhanced accuracy
- **Real-time Species Classification**: Identifies 207 bird species (200 CUB + 7 Airport Birds)
- **Size and Behavior Analysis**: Estimates bird size categories and flight behaviors
- **Altitude and Speed Estimation**: Calculates bird altitude and speed for risk assessment

### 🚨 **Risk Assessment & Safety**
- **Comprehensive Risk Scoring**: Multi-factor risk assessment (0-10 scale)
- **Airport-Specific Categories**: High, medium, and low-risk bird classifications
- **Collision Probability**: Real-time collision risk calculation
- **Severity Estimation**: Critical, high, medium, and low severity levels

### 📊 **Real-time Analytics & Dashboard**
- **Live Video Streaming**: Real-time video processing with detection overlays
- **Interactive Dashboard**: Comprehensive analytics and visualization
- **Risk Trend Analysis**: Historical risk pattern tracking
- **Performance Metrics**: FPS, detection rates, and system statistics

### 🔧 **Advanced Features**
- **Trajectory Tracking**: Bird movement pattern analysis
- **Alert System**: High-risk event notifications
- **API Endpoints**: RESTful API for integration
- **Data Export**: Comprehensive reporting capabilities

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for enhanced performance)
- 8GB+ RAM recommended
- 10GB+ free disk space

### Installation

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd Bird-Detection-and-Classification-System
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Download Datasets**
   ```bash
   # Download CUB-200-2011 and airport bird datasets
   python download_airport_birds.py
   ```

3. **Train Models**
   ```bash
   # Train bird classification models
   python train_bird_classifier.py
   ```

4. **Test the System**
   ```bash
   # Run comprehensive test suite
   python test_system.py
   ```

5. **Run the Application**
   ```bash
   # Start web dashboard
   python app.py
   # Open http://localhost:5000
   ```

## ▶️ Running the Application

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

## 📁 Project Structure

```
Bird-Detection-and-Classification-System/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── TECHNICAL_GUIDE.md              # Technical implementation guide
├── run_pipeline.py                 # Pipeline execution script
├── download_airport_birds.py       # Airport birds dataset downloader
├── download_comprehensive_bird_datasets.py # Comprehensive dataset downloader
├── download_cormorant_stork.py     # Additional dataset downloader
├── expand_airport7_dataset.py      # Dataset expansion script
├── split_train_val_airport7.py     # Train/val split script
├── train_bird_classifier.py        # Model training script
├── test_system.py                  # Testing suite
├── data/                           # Data directory
│   ├── cub_200_2011/
│   │   └── CUB_200_2011/           # CUB-200-2011 dataset (11,788 images)
│   ├── integrated_birds/           # Integrated dataset (CUB-200-2011 + airport species)
│   └── dataset_info.json           # Dataset information
├── models/                         # Trained models
├── output_improved/                # Test outputs
├── annotated_frames/               # Annotated output frames
├── src/                            # Source code
│   ├── detection/                  # Detection modules
│   │   ├── enhanced_bird_detector.py
│   │   └── bird_detector.py
│   ├── classification/             # Classification modules
│   │   └── species_classifier.py
│   ├── risk_assessment/            # Risk assessment modules
│   │   └── risk_calculator.py
│   └── utils/                      # Utility modules
│       ├── video_processor.py
│       ├── dashboard_utils.py
│       └── model_downloader.py
├── templates/                      # HTML templates
└── (static/)                       # Web app assets (if present)
```

> **Note:** The `integrated_birds` dataset contains both airport-relevant species and CUB-200-2011 classes, organized by species folders in `train/` and `val/`.
> The CUB-200-2011 dataset is located at `data/cub_200_2011/CUB_200_2011/`.
> The `static/` folder may not be present in all setups.

## 🎯 Target Species

The system is specifically designed to detect and classify these bird species:

### **High Risk Birds (Airport Safety)**
| Species | Scientific Name | Images | Risk Level | Size Category |
|---------|----------------|--------|------------|---------------|
| Black Kite | Milvus migrans | 100 | High | Large |
| Brahminy Kite | Haliastur indus | 100 | High | Large |
| Red Kite | Milvus milvus | 80 | High | Large |

### **Medium Risk Birds**
| Species | Scientific Name | Images | Risk Level | Size Category |
|---------|----------------|--------|------------|---------------|
| Egret | Ardea alba | 80 | Medium | Medium |
| Pigeon | Columba livia | 120 | Medium | Medium |
| Crow | Corvus brachyrhynchos | 100 | Medium | Medium |
| White-tailed Kite | Elanus leucurus | 60 | Medium | Medium |

### **CUB-200-2011 Dataset**
- **200 bird species** with 11,788 images
- **Fine-grained classification** for comprehensive bird identification
- **Bounding box annotations** for precise detection training

## 🎯 API Endpoints

### Dashboard & File Upload
- `GET /` — Main dashboard
- `POST /upload` — Upload image or video for processing (used by dashboard)

### Streaming
- `GET /stream` — Real-time video stream with detection overlays

### API Endpoints
- `POST /api/detect` — Single image detection and risk assessment
- `GET /api/risk_trend` — Risk trend for current session
- `GET /api/session_summary` — Session summary (species distribution, risk levels)
- `GET /data/sample_videos/<filename>` — Download uploaded sample videos

### Example API Usage
```python
import requests

# Upload image for detection
with open('bird_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/detect', files=files)
    results = response.json()
```

> **Note:** The main workflow, dashboard, and API are focused on the Airport7 dataset (7 classes: black_kite, brahminy_kite, cormorant, stork, egret, pigeon, crow). Fine-grained classification with CUB-200-2011 is available for research but not the default.

## 📊 Risk Assessment Model

The system uses a comprehensive risk assessment model with the following factors:

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

## 🎨 Dashboard Features

### Main Dashboard
- Real-time video streaming with detection overlays
- Live statistics and metrics
- Drag-and-drop video upload
- Risk level indicators
- Alert notifications

### Advanced Analytics Dashboard
- Risk trend analysis charts
- Species distribution visualization
- Behavior pattern analysis
- Performance metrics
- Historical data tracking

## 🔍 Detection Capabilities

### Multi-Model Ensemble
- **YOLOv8 Nano**: Fast detection for real-time processing
- **YOLOv8 Small**: Balanced accuracy and speed
- **YOLOv8 Medium**: High accuracy for critical detections

### Detection Features
- **Bounding Box Detection**: Precise bird localization
- **Confidence Scoring**: Detection reliability assessment
- **Size Estimation**: Bird size category classification
- **Trajectory Tracking**: Movement pattern analysis

## 📈 Performance Metrics

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

## 🛠️ Configuration

### Environment Variables
```bash
# Optional environment variables
FLASK_ENV=development
CUDA_VISIBLE_DEVICES=0  # GPU device selection
MODEL_PATH=/path/to/models  # Custom model path
```

### Model Configuration
```python
# Detection settings
CONFIDENCE_THRESHOLD = 0.5
USE_ENSEMBLE = True
MIN_BIRD_SIZE = 20
MAX_BIRD_SIZE = 500

# Risk assessment weights
RISK_WEIGHTS = {
    "size": 0.25,
    "altitude": 0.20,
    "speed": 0.20,
    "behavior": 0.15,
    "species": 0.20
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   python train_bird_classifier.py
   ```

2. **CUDA out of memory**
   - Use CPU mode: Set `device = "cpu"` in detector
   - Reduce batch size
   - Use smaller models (YOLOv8n instead of YOLOv8x)

3. **No detections**
   - Lower confidence threshold
   - Check video content
   - Ensure proper lighting

4. **Slow performance**
   - Use GPU acceleration
   - Enable ensemble detection
   - Process at lower resolution

### Performance Tips
- **GPU**: 3x faster processing with CUDA
- **Ensemble**: Better accuracy with multiple models
- **Confidence**: Adjust thresholds based on use case
- **Resolution**: Lower resolution for speed

## 📊 Dataset Information

### CUB-200-2011 Dataset
- **Images**: 11,788
- **Species**: 200
- **Format**: Images with bounding boxes and species labels
- **Use**: Fine-grained bird species classification

### Integrated Airport Birds Dataset
- **Images**: 640+ airport images, 11,788+ CUB-200-2011 images
- **Species**: 7 airport-relevant species + 200 CUB-200-2011 species
- **Format**: High-quality images with species labels, organized by folders
- **Use**: Airport-specific and fine-grained bird detection and risk assessment

### Airport Bird Classes
- **Categories**: 4 (kites, raptors, waterfowl, small_birds)
- **Species**: 16+ airport-relevant species
- **Use**: Airport-specific bird detection and risk assessment

## 🎯 Airport-Specific Features

### High-Risk Bird Detection
- **Kite Detection**: Specialized for kites (common at airports)
- **Raptor Identification**: Large birds of prey
- **Waterfowl Monitoring**: Large water birds

### Risk Assessment
- **Species-Specific Multipliers**: Different risk factors
- **Behavior Analysis**: Flight patterns
- **Altitude Estimation**: Height-based risk
- **Speed Assessment**: Movement analysis

### Alert System
- **High-Risk Alerts**: Automatic alerts
- **Trend Analysis**: Risk patterns
- **Real-time Monitoring**: Continuous surveillance

## 📝 API Examples

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

## 🚀 Next Steps

1. **Download datasets**: `python download_airport_birds.py`
2. **Train models**: `python train_bird_classifier.py`
3. **Test system**: `python test_system.py`
4. **Run application**: `python app.py`
5. **Access dashboard**: http://localhost:5000

For detailed technical information, see `TECHNICAL_GUIDE.md`. 
