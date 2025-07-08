"""
Risk Assessment Module
Calculates risk index for detected and classified birds
Optimized for airport bird strike prevention
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.classification.species_classifier import BirdClassification


@dataclass
class BirdRiskAssessment:
    """Data class for bird risk assessment results"""

    species: str
    size_category: str
    risk_score: float  #0-10 scale
    risk_level: str  #low, moderate, high
    altitude: float  #meters
    speed: float  #m/s
    behavior: str
    scientific_name: str
    category: str  #airport risk category
    collision_probability: float
    severity_estimate: str  #low, medium, high, critical


class RiskCalculator:
    """
    Main risk calculator class
    Calculates risk index based on bird size, altitude, speed, and behavior
    Optimized for airport bird strike prevention
    """

    def __init__(self):
        #Load airport bird classes for risk assessment
        self.airport_bird_classes = self._load_airport_bird_classes()

        #Risk factors for size, altitude, speed, and behavior
        self.size_factors = {
            "small": 1.0,  # Sparrows, finches, small songbirds
            "medium": 2.0,  # Pigeons, crows, medium raptors
            "large": 3.0,  # Eagles, vultures, large waterfowl
        }

        #Altitude-based risk factors (in meters)
        self.altitude_factor = lambda alt: (
            1.0 if alt < 100 else (2.0 if alt < 500 else 3.0)
        )

        #Speed-based risk factors (in m/s)
        self.speed_factor = lambda spd: 1.0 if spd < 10 else (2.0 if spd < 20 else 3.0)

        # Behavior-based risk factors
        self.behavior_factors = {
            "gliding": 1.0,  #Steady flight
            "hovering": 1.5,  #Stationary flight (high risk)
            "flapping": 1.2,  #Active flight
            "soaring": 1.3,  #Thermal soaring
            "diving": 2.0,  #Rapid descent (very high risk)
            "unknown": 1.0,  #Default behavior
        }

        #Species-specific risk multipliers
        self.species_risk_multipliers = {
            # High-risk airport species
            "black_kite": 1.5,  # Common at airports
            "brahminy_kite": 1.5,  # Common at airports
            "red_kite": 1.5,  # Common at airports
            "eagle": 2.0,  # Very large, high risk
            "hawk": 1.8,  # Medium-large, high risk
            "falcon": 1.8,  # Fast, high risk
            "vulture": 2.0,  # Very large, high risk
            # Waterfowl (medium-high risk)
            "cormorant": 1.3,  # Large, common near water
            "stork": 1.4,  # Very large
            "egret": 1.3,  # Large, common near water
            "heron": 1.3,  # Large, common near water
            "goose": 1.4,  # Very large
            # Small birds (low-medium risk)
            "sparrow": 0.8,  # Small, low risk
            "finch": 0.8,  # Small, low risk
            "starling": 0.9,  # Small-medium
            "pigeon": 1.0,  # Medium, common
        }

        #Load airport bird classes and risk factors
        self.airport_bird_classes = self._load_airport_bird_classes()
        self.risk_factors = self._load_risk_factors()

        #Risk calculation weights
        self.weights = {
            "size": 0.25,
            "altitude": 0.20,
            "speed": 0.20,
            "behavior": 0.15,
            "species": 0.20,
        }

        #Risk thresholds
        self.risk_thresholds = {"low": 3.0, "moderate": 6.0, "high": 8.0}

        #Airport-specific risk categories
        self.airport_risk_categories = {
            "high": ["black_kite", "brahminy_kite", "stork"],
            "medium": ["cormorant", "egret"],
            "low": ["small_birds", "pigeons"],
        }

        self.risk_history = []

    def _load_airport_bird_classes(self) -> Dict:
        """Load airport-specific bird class mappings"""
        classes_path = Path("data/classes/airport_birds.json")
        if classes_path.exists():
            with open(classes_path, "r") as f:
                return json.load(f)
        else:
            #Fallback to default airport bird classes
            return {
                "kites": {
                    "black_kite": "Milvus migrans",
                    "brahminy_kite": "Haliastur indus",
                    "red_kite": "Milvus milvus",
                },
                "raptors": {
                    "eagle": "Aquila spp",
                    "hawk": "Accipiter spp",
                    "falcon": "Falco spp",
                    "vulture": "Gyps spp",
                },
                "waterfowl": {
                    "cormorant": "Phalacrocorax spp",
                    "stork": "Ciconia spp",
                    "egret": "Egretta spp",
                    "heron": "Ardea spp",
                    "duck": "Anas spp",
                    "goose": "Branta spp",
                },
                "small_birds": {
                    "sparrow": "Passer spp",
                    "finch": "Fringilla spp",
                    "starling": "Sturnus vulgaris",
                    "pigeon": "Columba spp",
                },
            }

    def _load_risk_factors(self) -> Dict[str, Any]:
        """Load risk factor definitions"""
        risk_factors = {
            "size_factors": {
                "small": {"risk_score": 2.0, "collision_prob": 0.3},
                "medium": {"risk_score": 5.0, "collision_prob": 0.6},
                "large": {"risk_score": 8.0, "collision_prob": 0.9},
            },
            "altitude_factors": {
                "low": {"threshold": 50, "risk_score": 8.0, "collision_prob": 0.9},
                "medium": {"threshold": 150, "risk_score": 5.0, "collision_prob": 0.6},
                "high": {"threshold": 300, "risk_score": 2.0, "collision_prob": 0.3},
            },
            "speed_factors": {
                "slow": {"threshold": 15, "risk_score": 3.0, "collision_prob": 0.4},
                "medium": {"threshold": 25, "risk_score": 6.0, "collision_prob": 0.7},
                "fast": {"threshold": 35, "risk_score": 9.0, "collision_prob": 0.9},
            },
            "behavior_factors": {
                "perching": {"risk_score": 1.0, "collision_prob": 0.1},
                "flying": {"risk_score": 4.0, "collision_prob": 0.5},
                "diving": {"risk_score": 7.0, "collision_prob": 0.8},
                "soaring": {"risk_score": 6.0, "collision_prob": 0.7},
            },
            "species_factors": {
                "black_kite": {"risk_score": 8.0, "collision_prob": 0.8},
                "brahminy_kite": {"risk_score": 7.0, "collision_prob": 0.7},
                "cormorant": {"risk_score": 6.0, "collision_prob": 0.6},
                "stork": {"risk_score": 9.0, "collision_prob": 0.9},
                "egret": {"risk_score": 4.0, "collision_prob": 0.4},
            },
        }

        return risk_factors

    def calculate_risk(self, classification) -> BirdRiskAssessment:
        """
        Calculate risk for a single bird classification
        Args:
            classification: BirdClassification object
        Returns:
            BirdRiskAssessment object
        """
        size = classification.size_category
        altitude = classification.altitude_estimate
        speed = classification.speed_estimate
        behavior = classification.behavior
        species = classification.species
        timestamp = classification.timestamp

        #Get species information
        species_info = self._get_species_info(species)
        scientific_name = species_info.get("scientific_name", "Unknown")
        category = species_info.get("category", "unknown")

        #Calculate base risk factors
        size_factor = self.size_factors.get(size, 1.0)
        altitude_factor = self.altitude_factor(altitude)
        speed_factor = self.speed_factor(speed)
        behavior_factor = self.behavior_factors.get(behavior, 1.0)

        #Apply species-specific multiplier
        species_multiplier = self.species_risk_multipliers.get(species, 1.0)

        #Calculate final risk score
        risk_score = (
            size_factor
            * altitude_factor
            * speed_factor
            * behavior_factor
            * species_multiplier
        )

        #Determine risk level
        risk_level = self._risk_level(risk_score)

        #Calculate collision probability
        collision_prob = self._calculate_collision_probability(
            classification,
            size_factor,
            altitude_factor,
            speed_factor,
            behavior_factor,
            species_multiplier,
        )

        #Determine severity
        severity = self._determine_severity(classification, risk_score)

        return BirdRiskAssessment(
            species=species,
            size_category=size,
            risk_score=risk_score,
            risk_level=risk_level,
            altitude=altitude,
            speed=speed,
            behavior=behavior,
            scientific_name=scientific_name,
            category=category,
            collision_probability=collision_prob,
            severity_estimate=severity,
        )

    def _get_species_info(self, species: str) -> Dict:
        """Get species information from airport bird classes"""
        for category, species_dict in self.airport_bird_classes.items():
            if species in species_dict:
                return {"scientific_name": species_dict[species], "category": category}

        return {"scientific_name": "Unknown", "category": "unknown"}

    def calculate_risks(self, classifications: List) -> List[BirdRiskAssessment]:
        """
        Calculate risk for a list of bird classifications
        Args:
            classifications: List of BirdClassification objects
        Returns:
            List of BirdRiskAssessment objects
        """
        risk_assessments = []

        for classification in classifications:
            try:
                risk_assessment = self._calculate_single_risk(classification)
                if risk_assessment:
                    risk_assessments.append(risk_assessment)
                    self.risk_history.append(risk_assessment)

            except Exception as e:
                print(f"Error calculating risk for {classification.species}: {e}")
                continue

        return risk_assessments

    def _calculate_single_risk(
        self, classification: BirdClassification
    ) -> BirdRiskAssessment:
        """Calculate risk for a single bird classification"""

        #Calculate individual risk factors
        size_risk = self._calculate_size_risk(classification.size_category)
        altitude_risk = self._calculate_altitude_risk(classification.altitude_estimate)
        speed_risk = self._calculate_speed_risk(classification.speed_estimate)
        behavior_risk = self._calculate_behavior_risk(classification.behavior)
        species_risk = self._calculate_species_risk(classification.species)

        #Calculate weighted risk score
        risk_score = (
            size_risk * self.weights["size"]
            + altitude_risk * self.weights["altitude"]
            + speed_risk * self.weights["speed"]
            + behavior_risk * self.weights["behavior"]
            + species_risk * self.weights["species"]
        )

        #Determine risk level
        risk_level = self._determine_risk_level(risk_score)

        #Calculate collision probability
        collision_prob = self._calculate_collision_probability(
            classification,
            size_risk,
            altitude_risk,
            speed_risk,
            behavior_risk,
            species_risk,
        )

        # Determine severity
        severity = self._determine_severity(classification, risk_score)
        category = self._get_airport_category(classification.species)

        return BirdRiskAssessment(
            species=classification.species,
            size_category=classification.size_category,
            risk_score=risk_score,
            risk_level=risk_level,
            altitude=classification.altitude_estimate,
            speed=classification.speed_estimate,
            behavior=classification.behavior,
            scientific_name=classification.scientific_name,
            category=category,
            collision_probability=collision_prob,
            severity_estimate=severity,
        )

    def _calculate_size_risk(self, size_category: str) -> float:
        """Calculate risk based on bird size"""
        size_factors = self.risk_factors["size_factors"]
        size_info = size_factors.get(size_category, size_factors["medium"])
        return size_info["risk_score"]

    def _calculate_altitude_risk(self, altitude: float) -> float:
        """Calculate risk based on altitude"""
        altitude_factors = self.risk_factors["altitude_factors"]

        if altitude <= altitude_factors["low"]["threshold"]:
            return altitude_factors["low"]["risk_score"]
        elif altitude <= altitude_factors["medium"]["threshold"]:
            return altitude_factors["medium"]["risk_score"]
        else:
            return altitude_factors["high"]["risk_score"]

    def _calculate_speed_risk(self, speed: float) -> float:
        """Calculate risk based on speed"""
        speed_factors = self.risk_factors["speed_factors"]

        if speed <= speed_factors["slow"]["threshold"]:
            return speed_factors["slow"]["risk_score"]
        elif speed <= speed_factors["medium"]["threshold"]:
            return speed_factors["medium"]["risk_score"]
        else:
            return speed_factors["fast"]["risk_score"]

    def _calculate_behavior_risk(self, behavior: str) -> float:
        """Calculate risk based on behavior"""
        behavior_factors = self.risk_factors["behavior_factors"]
        behavior_info = behavior_factors.get(behavior, behavior_factors["flying"])
        return behavior_info["risk_score"]

    def _calculate_species_risk(self, species: str) -> float:
        """Calculate risk based on species"""
        species_factors = self.risk_factors["species_factors"]
        species_info = species_factors.get(species, {"risk_score": 5.0})
        return species_info["risk_score"]

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score <= self.risk_thresholds["low"]:
            return "low"
        elif risk_score <= self.risk_thresholds["moderate"]:
            return "moderate"
        else:
            return "high"

    def _calculate_collision_probability(
        self,
        classification: BirdClassification,
        size_risk: float,
        altitude_risk: float,
        speed_risk: float,
        behavior_risk: float,
        species_risk: float,
    ) -> float:
        """Calculate collision probability"""
        #Get individual probabilities
        size_factors = self.risk_factors["size_factors"]
        altitude_factors = self.risk_factors["altitude_factors"]
        speed_factors = self.risk_factors["speed_factors"]
        behavior_factors = self.risk_factors["behavior_factors"]
        species_factors = self.risk_factors["species_factors"]

        size_prob = size_factors[classification.size_category]["collision_prob"]

        #Altitude probability
        if classification.altitude_estimate <= altitude_factors["low"]["threshold"]:
            altitude_prob = altitude_factors["low"]["collision_prob"]
        elif (
            classification.altitude_estimate <= altitude_factors["medium"]["threshold"]
        ):
            altitude_prob = altitude_factors["medium"]["collision_prob"]
        else:
            altitude_prob = altitude_factors["high"]["collision_prob"]

        #Speed probability
        if classification.speed_estimate <= speed_factors["slow"]["threshold"]:
            speed_prob = speed_factors["slow"]["collision_prob"]
        elif classification.speed_estimate <= speed_factors["medium"]["threshold"]:
            speed_prob = speed_factors["medium"]["collision_prob"]
        else:
            speed_prob = speed_factors["fast"]["collision_prob"]

        behavior_prob = behavior_factors[classification.behavior]["collision_prob"]
        species_prob = species_factors.get(
            classification.species, {"collision_prob": 0.5}
        )["collision_prob"]

        #Weighted average of probabilities
        collision_prob = (
            size_prob * self.weights["size"]
            + altitude_prob * self.weights["altitude"]
            + speed_prob * self.weights["speed"]
            + behavior_prob * self.weights["behavior"]
            + species_prob * self.weights["species"]
        )

        return min(1.0, max(0.0, collision_prob))

    def _determine_severity(
        self, classification: BirdClassification, risk_score: float
    ) -> str:
        """Determine collision severity"""
        #Consider size and species for severity
        if classification.size_category == "large" and risk_score > 7:
            return "critical"
        elif classification.size_category == "large" or risk_score > 6:
            return "high"
        elif classification.size_category == "medium" or risk_score > 4:
            return "medium"
        else:
            return "low"

    def _get_airport_category(self, species: str) -> str:
        """Get airport risk category for species"""
        for category, species_list in self.airport_risk_categories.items():
            if species in species_list:
                return category
        return "medium"

    def _risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on calculated score
        Risk levels:
        - Low (0-3): Unlikely to affect operations
        - Moderate (4-7): May need monitoring
        - High (8+): Potentially dangerous, alert needed
        """
        if risk_score < 4:
            return "Low"
        elif risk_score < 8:
            return "Moderate"
        else:
            return "High"

    def get_risk_statistics(self, risk_assessments: List[BirdRiskAssessment]) -> Dict:
        """Get comprehensive risk statistics"""
        if not risk_assessments:
            return {}

        levels = [r.risk_level for r in risk_assessments]
        scores = [r.risk_score for r in risk_assessments]
        species_list = [r.species for r in risk_assessments]
        categories = [r.category for r in risk_assessments]

        #Count high-risk species
        high_risk_species = [
            "black_kite",
            "brahminy_kite",
            "eagle",
            "hawk",
            "falcon",
            "vulture",
            "stork",
            "goose",
        ]
        high_risk_count = sum(
            1 for species in species_list if species in high_risk_species
        )

        return {
            "total_birds": len(risk_assessments),
            "risk_levels": {
                "low": levels.count("Low"),
                "moderate": levels.count("Moderate"),
                "high": levels.count("High"),
            },
            "risk_scores": {
                "average": np.mean(scores),
                "max": np.max(scores),
                "min": np.min(scores),
                "std": np.std(scores),
            },
            "species_distribution": {
                "unique_species": len(set(species_list)),
                "high_risk_species_count": high_risk_count,
                "most_common_species": (
                    max(set(species_list), key=species_list.count)
                    if species_list
                    else None
                ),
            },
            "category_distribution": {
                "kites": categories.count("kites"),
                "raptors": categories.count("raptors"),
                "waterfowl": categories.count("waterfowl"),
                "small_birds": categories.count("small_birds"),
                "unknown": categories.count("unknown"),
            },
            "alerts": {
                "high_risk_alerts": len(
                    [r for r in risk_assessments if r.risk_level == "High"]
                ),
                "moderate_risk_alerts": len(
                    [r for r in risk_assessments if r.risk_level == "Moderate"]
                ),
            },
        }

    def get_airport_specific_risks(
        self, risk_assessments: List[BirdRiskAssessment]
    ) -> Dict:
        """Get airport-specific risk analysis"""
        if not risk_assessments:
            return {}

        #Filter for airport-specific high-risk birds
        airport_high_risk = [
            "black_kite",
            "brahminy_kite",
            "eagle",
            "hawk",
            "falcon",
            "vulture",
            "stork",
            "cormorant",
            "egret",
        ]

        airport_risks = [r for r in risk_assessments if r.species in airport_high_risk]

        return {
            "airport_high_risk_birds": len(airport_risks),
            "airport_risk_species": list(set([r.species for r in airport_risks])),
            "highest_risk_species": (
                max(airport_risks, key=lambda x: x.risk_score).species
                if airport_risks
                else None
            ),
            "average_airport_risk": (
                np.mean([r.risk_score for r in airport_risks]) if airport_risks else 0
            ),
        }

    def get_risk_trends(self, window_size: int = 50) -> Dict[str, Any]:
        """Get risk trends over time"""
        if len(self.risk_history) < window_size:
            return {}

        recent_risks = self.risk_history[-window_size:]

        #Calculate trend statistics
        risk_scores = [r.risk_score for r in recent_risks]
        collision_probs = [r.collision_probability for r in recent_risks]

        #Trend analysis
        if len(risk_scores) > 1:
            risk_trend = np.polyfit(range(len(risk_scores)), risk_scores, 1)[0]
            collision_trend = np.polyfit(
                range(len(collision_probs)), collision_probs, 1
            )[0]
        else:
            risk_trend = 0
            collision_trend = 0

        return {
            "window_size": window_size,
            "average_risk_score": np.mean(risk_scores),
            "average_collision_probability": np.mean(collision_probs),
            "risk_trend": risk_trend,
            "collision_trend": collision_trend,
            "trend_direction": (
                "increasing"
                if risk_trend > 0
                else "decreasing" if risk_trend < 0 else "stable"
            ),
        }

    def get_high_risk_alerts(
        self, risks: List[BirdRiskAssessment]
    ) -> List[Dict[str, Any]]:
        """Get high-risk alerts"""
        alerts = []

        for risk in risks:
            if risk.risk_score > 8.0 or risk.collision_probability > 0.8:
                alerts.append(
                    {
                        "species": risk.species,
                        "risk_score": risk.risk_score,
                        "collision_probability": risk.collision_probability,
                        "severity": risk.severity_estimate,
                        "altitude": risk.altitude,
                        "speed": risk.speed,
                        "behavior": risk.behavior,
                        "alert_level": "critical" if risk.risk_score > 9.0 else "high",
                    }
                )

        return alerts
