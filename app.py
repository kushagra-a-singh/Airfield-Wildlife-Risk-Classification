import json
import os

import cv2
import numpy as np
import torch
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from src.classification.species_classifier import SpeciesClassifier
from src.detection.enhanced_bird_detector import EnhancedBirdDetector
from src.risk_assessment.risk_calculator import RiskCalculator

UPLOAD_FOLDER = "data/sample_videos/"
ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tiff",
    "tif",
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


detector = EnhancedBirdDetector(use_ensemble=False)
risk_calc = RiskCalculator()
risk_trend = []
session_summary = {}


@app.route("/")
def index():
    return render_template("index.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_classifier(model_name):
    if model_name == "ensemble":
        return SpeciesClassifier(use_ensemble=True)
    else:
        classifier = SpeciesClassifier(use_ensemble=False)
        
        if model_name == "resnet18":
            from src.classification.species_classifier import ResNetBirdClassifier

            classifier.model = ResNetBirdClassifier(num_classes=7)
            model_path = "models/resnet18_airport7.pth"
            if os.path.exists(model_path):
                classifier.model.load_state_dict(
                    torch.load(model_path, map_location=classifier.device)
                )
                classifier.model.to(classifier.device)
                classifier.model.eval()
        elif model_name == "mobilenetv2":
            from src.classification.species_classifier import MobileNetBirdClassifier

            classifier.model = MobileNetBirdClassifier(num_classes=7)
            model_path = "models/mobilenetv2_airport7.pth"
            if os.path.exists(model_path):
                classifier.model.load_state_dict(
                    torch.load(model_path, map_location=classifier.device)
                )
                classifier.model.to(classifier.device)
                classifier.model.eval()
        return classifier


def build_session_summary(detections, classifications, risks):
    summary = {
        "total_detections": len(detections),
        "species_distribution": {},
        "risk_levels": {"low": 0, "moderate": 0, "high": 0},
        "models_used": set(),
    }
    for c in classifications:
        summary["species_distribution"][c.species] = (
            summary["species_distribution"].get(c.species, 0) + 1
        )
        summary["models_used"].add(getattr(c, "risk_category", "ensemble"))
    for r in risks:
        summary["risk_levels"][r.risk_level] += 1
    summary["models_used"] = list(summary["models_used"])
    return summary


@app.route("/upload", methods=["POST"])
def upload_file():
    global risk_trend, session_summary
    model = request.form.get("model", "ensemble")
    classifier = get_classifier(model)
   
    risk_trend = []
    session_summary = {}
    if "video" in request.files:
        file = request.files["video"]
        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)
        app.config["CURRENT_VIDEO"] = filename
        app.config["CURRENT_FILE_TYPE"] = "video"
        app.config["CURRENT_MODEL"] = model
        return jsonify({"success": True, "filetype": "video"})
    elif "image" in request.files:
        file = request.files["image"]
        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)
        app.config["CURRENT_IMAGE"] = filename
        app.config["CURRENT_FILE_TYPE"] = "image"
        app.config["CURRENT_MODEL"] = model
        frame = cv2.imread(filename)
        detections = detector.detect_birds_in_frame(frame, 0)
        classifications = classifier.classify_birds_in_detections(detections, frame)
        risks = risk_calc.calculate_risks(classifications)
        
        species_labels = {
            det.id: c.species for det, c in zip(detections, classifications)
        }
        frame_with_boxes = detector.draw_detections(frame, detections, species_labels)
        annotated_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"annotated_{file.filename}"
        )
        cv2.imwrite(annotated_path, frame_with_boxes)
        
        results = []
        for i, (detection, classification, risk) in enumerate(
            zip(detections, classifications, risks)
        ):
            
            pred_result = classifier.predict(
                frame[
                    int(detection.bbox[1]) : int(detection.bbox[3]),
                    int(detection.bbox[0]) : int(detection.bbox[2]),
                ]
            )
            results.append(
                {
                    "id": i,
                    "bbox": detection.bbox,
                    "confidence": detection.confidence,
                    "species": classification.species,
                    "size_category": classification.size_category,
                    "behavior": getattr(classification, "behavior", "Unknown"),
                    "model": pred_result.get("method", model),
                    "risk_level": risk.risk_level,
                    "risk_score": risk.risk_score,
                }
            )
        session_summary = build_session_summary(detections, classifications, risks)
        return jsonify(
            {
                "success": True,
                "filetype": "image",
                "annotated": f"annotated_{file.filename}",
                "results": results,
                "session_summary": session_summary,
            }
        )
    return jsonify({"success": False, "error": "No valid file uploaded"})


@app.route("/stream")
def stream():
    video_path = app.config.get("CURRENT_VIDEO", None)
    model = app.config.get("CURRENT_MODEL", "ensemble")
    classifier = get_classifier(model)
    if not video_path:
        return "No video uploaded", 400

    def generate():
        print("Starting video stream")
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        global risk_trend, session_summary
        risk_trend = []
        all_detections = []
        all_classifications = []
        all_risks = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detector.detect_birds_in_frame(frame, frame_id)
                classifications = classifier.classify_birds_in_detections(
                    detections, frame
                )
                risks = risk_calc.calculate_risks(classifications)
                all_detections.extend(detections)
                all_classifications.extend(classifications)
                all_risks.extend(risks)

                #update session summary after each frame for realtime access
                session_summary = build_session_summary(
                    all_detections, all_classifications, all_risks
                )

                species_labels = {
                    det.id: c.species for det, c in zip(detections, classifications)
                }
                frame_with_boxes = detector.draw_detections(
                    frame, detections, species_labels
                )

                if risks:
                    avg_risk = float(np.mean([r.risk_score for r in risks]))
                    risk_trend.append({"frame_id": frame_id, "risk_score": avg_risk})

                ret2, jpeg = cv2.imencode(
                    ".jpg", frame_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )
                if not ret2:
                    continue

                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
                frame_id += 1

            #if we exit the loop naturally yield the termination signal
            cap.release()
            session_summary = build_session_summary(
                all_detections, all_classifications, all_risks
            )
            termination_data = json.dumps(
                {
                    "status": "completed",
                    "summary": session_summary,
                    "risk_trend": risk_trend,
                    "total_frames": frame_id,
                }
            ).encode()
            print("Yielding stream end with summary and risk_trend")
            yield (
                b"--frame\r\nX-Stream-End: true\r\nContent-Type: application/json\r\n\r\n"
                + termination_data
                + b"\r\n"
            )
        except GeneratorExit:
            #if the generator is closed early by pressing stop..just update the summary and do not yield
            cap.release()
            session_summary = build_session_summary(
                all_detections, all_classifications, all_risks
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/risk_trend")
def api_risk_trend():
    global risk_trend
    return jsonify(risk_trend[-100:])


@app.route("/api/session_summary")
def api_session_summary():
    global session_summary
    return jsonify(session_summary)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    model = request.form.get("model", "ensemble")
    classifier = get_classifier(model)
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detections = detector.detect_birds_in_frame(frame, 0)
    classifications = classifier.classify_birds_in_detections(detections, frame)
    risks = risk_calc.calculate_risks(classifications)
    results = []
    for i, (detection, classification, risk) in enumerate(
        zip(detections, classifications, risks)
    ):
        pred_result = classifier.predict(
            frame[
                int(detection.bbox[1]) : int(detection.bbox[3]),
                int(detection.bbox[0]) : int(detection.bbox[2]),
            ]
        )
        results.append(
            {
                "id": i,
                "bbox": detection.bbox,
                "confidence": detection.confidence,
                "species": classification.species,
                "size_category": classification.size_category,
                "behavior": getattr(classification, "behavior", "Unknown"),
                "model": pred_result.get("method", model),
                "risk_level": risk.risk_level,
                "risk_score": risk.risk_score,
            }
        )
    return jsonify({"detections": len(detections), "results": results})


@app.route("/data/sample_videos/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/favicon.ico")
def favicon():
    return "", 204  


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    #for local development only:
    #python app.py
    #for production with better stream handling, run with:
    #waitress-serve --host=0.0.0.0 --port=5000 app:app
