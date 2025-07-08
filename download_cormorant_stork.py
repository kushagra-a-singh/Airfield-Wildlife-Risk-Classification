#!/usr/bin/env python3
"""
Download images for Cormorant and Stork for airport bird classification.
Saves images in data/airport7/train/.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import requests


class CormorantStorkDownloader:
    def __init__(self):
        self.base_dir = Path("data/airport7/train")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.target_species = {
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

    def download_from_inaturalist(
        self, species: str, search_terms: List[str], count: int
    ) -> List[str]:
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
                        save_path = self.base_dir / species / filename
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
                        save_path = self.base_dir / species / filename
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
                        save_path = self.base_dir / species / filename
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

    def download_all_species(self):
        print("🚀 Starting cormorant and stork dataset download...")
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
        print("🦅 Cormorant and Stork Dataset Downloader")
        print("=" * 60)
        self.download_all_species()
        print("\n✅ Download complete!")
        print(f"\nImages saved in: {self.base_dir}")
        print("\nNext steps:")
        print("1. Train models: python train_bird_classifier.py")
        print("2. Test system: python test_system.py")


def main():
    downloader = CormorantStorkDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
