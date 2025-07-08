"""
End-to-End Bird Detection, Classification, and Risk Assessment Pipeline
"""
import os
import cv2
from src.detection.bird_detector import BirdDetector
from src.classification.species_classifier import SpeciesClassifier
from src.risk_assessment.risk_calculator import RiskCalculator

SAMPLE_VIDEO = 'data/sample_videos/sample_birds.mp4'  

if __name__ == '__main__':
    detector = BirdDetector()
    classifier = SpeciesClassifier()
    risk_calc = RiskCalculator()

    if not os.path.exists(SAMPLE_VIDEO):
        print(f"Sample video not found: {SAMPLE_VIDEO}")
        exit(1)
    cap = cv2.VideoCapture(SAMPLE_VIDEO)
    frame_id = 0
    all_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect_birds_in_frame(frame, frame_id)
        classifications = classifier.classify_birds_in_detections(detections, frame)
        risks = risk_calc.calculate_risks(classifications)
        frame_with_boxes = detector.draw_detections(frame, detections)
        
        cv2.imshow('Bird Detection', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        all_results.append({
            'frame_id': frame_id,
            'detections': detections,
            'classifications': classifications,
            'risks': risks
        })
        frame_id += 1
        if frame_id >= 50:  
            break
    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_id} frames.")
    print("Sample risk assessment for last frame:")
    if all_results:
        for risk in all_results[-1]['risks']:
            print(risk)
    print("Pipeline complete.") 
