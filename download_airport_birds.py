#!/usr/bin/env python3
"""
Download and integrate airport bird datasets for bird strike prevention.
Downloads images for Black Kite, Brahminy Kite, Egret, Pigeon, Crow, and other kite species.
Integrates with CUB-200-2011 dataset for comprehensive bird classification.
"""

import hashlib
import io
import json
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

#for iNaturalist api
import requests
from PIL import Image


class AirportBirdDownloader:
    """Downloads and integrates airport bird datasets"""

    def __init__(self):
        self.base_dir = Path("data")
        self.integrated_dir = self.base_dir / "integrated_birds"
        self.cub_dir = self.base_dir / "cub_200_2011" / "CUB_200_2011"

        self._create_directories()

        self.target_species = {
            # "black_kite": { ... },
            # "brahminy_kite": { ... },
            # "egret": { ... },
            # "pigeon": { ... },
            # "crow": { ... },
            # "red_kite": { ... },
            # "white_tailed_kite": { ... },
            "cormorant": {
                "scientific_name": "Phalacrocorax carbo",
                "common_name": "Great Cormorant",
                "search_terms": [
                    "great cormorant bird",
                    "Phalacrocorax carbo",
                    "cormorant flying",
                    "cormorant bird",
                ],
                "images_needed": 100,
                "risk_level": "medium",
            },
            "stork": {
                "scientific_name": "Ciconia ciconia",
                "common_name": "White Stork",
                "search_terms": [
                    "white stork bird",
                    "Ciconia ciconia",
                    "stork flying",
                    "stork bird",
                ],
                "images_needed": 100,
                "risk_level": "medium",
            },
        }

        self.unsplash_access_key = "UNSPLASH_ACCESS_KEY" 
        self.flickr_api_key = "FLICKR_API_KEY"  

    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.integrated_dir / "train",
            self.integrated_dir / "val",
            self.base_dir / "metadata",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")

    def download_from_inaturalist(
        self, species: str, search_terms: List[str], count: int
    ) -> List[str]:
        """Download images from iNaturalist API"""
        downloaded_paths = []

        for search_term in search_terms:
            if len(downloaded_paths) >= count:
                break

            try:
                url = "https://api.inaturalist.org/v1/observations"
                params = {
                    "q": search_term,
                    "per_page": min(200, count - len(downloaded_paths)),
                    "quality_grade": "research",
                    "has": ["photos"],
                }

                response = requests.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                for observation in data.get("results", []):
                    if len(downloaded_paths) >= count:
                        break

                    try:
                        photos = observation.get("photos", [])
                        if not photos:
                            continue

                        photo = photos[0]
                        image_url = photo["url"].replace("square", "large")

                        image_response = requests.get(image_url)
                        image_response.raise_for_status()

                        filename = f"{species}_{len(downloaded_paths):03d}.jpg"
                        save_path = self.integrated_dir / "train" / species / filename
                        save_path.parent.mkdir(exist_ok=True)

                        with open(save_path, "wb") as f:
                            f.write(image_response.content)

                        downloaded_paths.append(str(save_path))
                        print(f"  Downloaded: {filename}")

                        time.sleep(0.3)

                    except Exception as e:
                        print(f"  Error downloading image: {e}")
                        continue

            except Exception as e:
                print(f"  Error with iNaturalist API: {e}")
                continue

        return downloaded_paths

    def download_from_unsplash(
        self, species: str, search_terms: List[str], count: int
    ) -> List[str]:
        """Download images from Unsplash API"""
        downloaded_paths = []

        for search_term in search_terms:
            if len(downloaded_paths) >= count:
                break

            try:
                url = "https://api.unsplash.com/search/photos"
                headers = {"Authorization": f"Client-ID {self.unsplash_access_key}"}
                params = {
                    "query": search_term,
                    "per_page": min(30, count - len(downloaded_paths)),
                    "orientation": "landscape",
                }

                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()

                for photo in data.get("results", []):
                    if len(downloaded_paths) >= count:
                        break

                    try:
                        image_url = photo["urls"]["regular"]
                        image_response = requests.get(image_url)
                        image_response.raise_for_status()

                        filename = f"{species}_{len(downloaded_paths):03d}.jpg"
                        save_path = self.integrated_dir / "train" / species / filename
                        save_path.parent.mkdir(exist_ok=True)

                        with open(save_path, "wb") as f:
                            f.write(image_response.content)

                        downloaded_paths.append(str(save_path))
                        print(f"  Downloaded: {filename}")

                        time.sleep(0.1)

                    except Exception as e:
                        print(f"  Error downloading image: {e}")
                        continue

            except Exception as e:
                print(f"  Error with Unsplash API: {e}")
                continue

        return downloaded_paths

    def download_from_flickr(
        self, species: str, search_terms: List[str], count: int
    ) -> List[str]:
        """Download images from Flickr API"""
        downloaded_paths = []

        for search_term in search_terms:
            if len(downloaded_paths) >= count:
                break

            try:
                url = "https://www.flickr.com/services/rest/"
                params = {
                    "method": "flickr.photos.search",
                    "api_key": self.flickr_api_key,
                    "text": search_term,
                    "per_page": min(100, count - len(downloaded_paths)),
                    "format": "json",
                    "nojsoncallback": 1,
                    "sort": "relevance",
                }

                response = requests.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                for photo in data.get("photos", {}).get("photo", []):
                    if len(downloaded_paths) >= count:
                        break

                    try:
                        farm_id = photo["farm"]
                        server_id = photo["server"]
                        photo_id = photo["id"]
                        secret = photo["secret"]

                        image_url = f"https://live.staticflickr.com/{server_id}/{photo_id}_{secret}_b.jpg"

                        image_response = requests.get(image_url)
                        image_response.raise_for_status()

                        filename = f"{species}_{len(downloaded_paths):03d}.jpg"
                        save_path = self.integrated_dir / "train" / species / filename
                        save_path.parent.mkdir(exist_ok=True)

                        with open(save_path, "wb") as f:
                            f.write(image_response.content)

                        downloaded_paths.append(str(save_path))
                        print(f"  Downloaded: {filename}")

                        time.sleep(0.2)

                    except Exception as e:
                        print(f"  Error downloading image: {e}")
                        continue

            except Exception as e:
                print(f"  Error with Flickr API: {e}")
                continue

        return downloaded_paths

    def download_species_images(self, species: str, species_info: Dict) -> int:
        """Download images for a specific species"""
        print(
            f"\n🦅 Downloading {species_info['common_name']} ({species_info['scientific_name']})"
        )
        print(f"   Risk Level: {species_info['risk_level']}")
        print(f"   Target Images: {species_info['images_needed']}")

        downloaded_count = 0

        sources = [
            ("iNaturalist", self.download_from_inaturalist),
            ("Unsplash", self.download_from_unsplash),
            ("Flickr", self.download_from_flickr),
        ]

        for source_name, download_func in sources:
            if downloaded_count >= species_info["images_needed"]:
                break

            print(f"  📥 Trying {source_name}...")
            try:
                downloaded = download_func(
                    species,
                    species_info["search_terms"],
                    species_info["images_needed"] - downloaded_count,
                )
                downloaded_count += len(downloaded)
                print(f"  ✅ Downloaded {len(downloaded)} images from {source_name}")

            except Exception as e:
                print(f"  ❌ Error with {source_name}: {e}")
                continue

        print(f"  📊 Total downloaded: {downloaded_count} images")
        return downloaded_count

    def integrate_with_cub_dataset(self):
        """Integrate downloaded images with CUB dataset structure"""
        print("\n🔗 Integrating with CUB-200-2011 dataset...")

        #create train/val split for integrated dataset
        for split in ["train", "val"]:
            split_dir = self.integrated_dir / split
            split_dir.mkdir(exist_ok=True)

            cub_images_dir = self.cub_dir / "images"
            if cub_images_dir.exists():
                for class_dir in cub_images_dir.iterdir():
                    if class_dir.is_dir():
                        
                        integrated_class_dir = split_dir / class_dir.name
                        integrated_class_dir.mkdir(exist_ok=True)

                       
                        for img_file in class_dir.glob("*.jpg"):
                            if split == "train" or img_file.name.endswith(
                                ("_0001.jpg", "_0002.jpg")
                            ):
                                
                                import shutil

                                shutil.copy2(
                                    img_file, integrated_class_dir / img_file.name
                                )

        print("✅ Integration with CUB dataset completed")

    def create_metadata(self):
        """Create metadata files for the integrated dataset"""
        print("\n📋 Creating metadata...")

        metadata = {
            "dataset_info": {
                "name": "Integrated Bird Dataset (CUB-200-2011 + Airport Birds)",
                "description": "Combined dataset for bird detection and classification",
                "total_classes": 207,  #200 CUB + 7 airport birds
                "airport_birds": list(self.target_species.keys()),
                "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "airport_birds": self.target_species,
            "cub_classes": "200 species from CUB-200-2011 dataset",
        }

        metadata_dir = self.base_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = metadata_dir / "integrated_dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to: {metadata_path}")

    def download_all_species(self):
        """Download images for all target species"""
        print("🚀 Starting airport bird dataset download...")
        print("=" * 60)

        total_downloaded = 0

        for species in ["cormorant", "stork"]:
            species_info = self.target_species[species]
            downloaded = self.download_species_images(species, species_info)
            total_downloaded += downloaded
            species_info["downloaded_count"] = downloaded

        print(f"\n📊 Download Summary:")
        print(f"   Total species: 2 (cormorant, stork)")
        print(f"   Total images downloaded: {total_downloaded}")

        for species in ["cormorant", "stork"]:
            info = self.target_species[species]
            print(f"   {info['common_name']}: {info.get('downloaded_count', 0)} images")

        return total_downloaded

    def run(self):
        """Main execution function"""
        print("🦅 Airport Bird Dataset Downloader")
        print("=" * 60)

        if self.integrated_dir.exists() and any(
            (self.integrated_dir / "train").iterdir()
        ):
            print("✅ Dataset already exists!")
            print(f"📁 Location: {self.integrated_dir}")

            total_images = 0
            for species_dir in (self.integrated_dir / "train").iterdir():
                if species_dir.is_dir():
                    images = list(species_dir.glob("*.jpg"))
                    total_images += len(images)
                    print(f"   {species_dir.name}: {len(images)} images")

            print(f"📊 Total images found: {total_images}")

            print("\n📋 Creating metadata...")
            self.create_metadata()

            print("\n✅ Dataset is ready for use!")
            print("\nNext steps:")
            print("1. Train models: python train_bird_classifier.py")
            print("2. Test system: python test_system.py")
            return

        total_downloaded = self.download_all_species()

        if total_downloaded > 0:
            
            self.integrate_with_cub_dataset()

            self.create_metadata()

            print("\n✅ Dataset download and integration completed!")
            print(f"📁 Integrated dataset location: {self.integrated_dir}")
            print(f"📊 Total images: {total_downloaded} airport birds + CUB-200-2011")
            print("\nNext steps:")
            print("1. Train models: python train_bird_classifier.py")
            print("2. Test system: python test_system.py")
        else:
            print(
                "\n❌ No images were downloaded. Please check your API keys and internet connection."
            )


def main():
    """Main function"""
    downloader = AirportBirdDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
