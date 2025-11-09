#!/usr/bin/env python3
"""
Pipeline to associate each house with its respective images from the Houses dataset.

The dataset structure:
- Images are named: {house_id}_{room_type}.jpg
  where room_type is one of: bathroom, bedroom, kitchen, frontal
- Metadata file: HousesInfo.txt
  Format: bedrooms bathrooms area zipcode price (one line per house)
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class HouseMetadata:
    """House metadata structure."""
    house_id: int
    bedrooms: float
    bathrooms: float
    area: int
    zipcode: int
    price: int


@dataclass
class HouseImages:
    """House images structure."""
    house_id: int
    bathroom: Optional[str] = None
    bedroom: Optional[str] = None
    kitchen: Optional[str] = None
    frontal: Optional[str] = None


@dataclass
class House:
    """Complete house information with metadata and images."""
    house_id: int
    metadata: HouseMetadata
    images: HouseImages


class HouseImagePipeline:
    """Pipeline to associate houses with their images."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the pipeline.
        
        Args:
            dataset_path: Path to the 'Houses Dataset' directory
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path
        self.metadata_file = self.dataset_path / "HousesInfo.txt"
        
        # Validate paths
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
    
    def read_metadata(self) -> Dict[int, HouseMetadata]:
        """
        Read house metadata from HousesInfo.txt.
        
        Returns:
            Dictionary mapping house_id to HouseMetadata
        """
        metadata = {}
        
        with open(self.metadata_file, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                
                try:
                    bedrooms = float(parts[0])
                    bathrooms = float(parts[1])
                    area = int(parts[2])
                    zipcode = int(parts[3])
                    price = int(parts[4])
                    
                    metadata[line_num] = HouseMetadata(
                        house_id=line_num,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        area=area,
                        zipcode=zipcode,
                        price=price
                    )
                except ValueError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        return metadata
    
    def parse_image_filename(self, filename: str) -> Optional[tuple]:
        """
        Parse image filename to extract house_id and room_type.
        
        Args:
            filename: Image filename (e.g., "1_bathroom.jpg")
            
        Returns:
            Tuple of (house_id, room_type) or None if parsing fails
        """
        # Pattern: {house_id}_{room_type}.jpg
        pattern = r'^(\d+)_(bathroom|bedroom|kitchen|frontal)\.jpg$'
        match = re.match(pattern, filename)
        
        if match:
            house_id = int(match.group(1))
            room_type = match.group(2)
            return (house_id, room_type)
        
        return None
    
    def scan_images(self) -> Dict[int, HouseImages]:
        """
        Scan images directory and organize images by house_id.
        
        Returns:
            Dictionary mapping house_id to HouseImages
        """
        images_by_house = {}
        
        # Get all jpg files
        image_files = list(self.images_dir.glob("*.jpg"))
        
        for image_path in image_files:
            filename = image_path.name
            parsed = self.parse_image_filename(filename)
            
            if parsed is None:
                print(f"Warning: Skipping unparseable image: {filename}")
                continue
            
            house_id, room_type = parsed
            
            # Initialize HouseImages if not exists
            if house_id not in images_by_house:
                images_by_house[house_id] = HouseImages(house_id=house_id)
            
            # Set the image path
            setattr(images_by_house[house_id], room_type, str(image_path))
        
        return images_by_house
    
    def associate_houses_with_images(self) -> List[House]:
        """
        Main pipeline method: Associate each house with its images.
        
        Returns:
            List of House objects with complete information
        """
        print("Reading metadata...")
        metadata = self.read_metadata()
        print(f"Found metadata for {len(metadata)} houses")
        
        print("Scanning images...")
        images_by_house = self.scan_images()
        print(f"Found images for {len(images_by_house)} houses")
        
        print("Associating houses with images...")
        houses = []
        
        # Get all unique house IDs
        all_house_ids = set(metadata.keys()) | set(images_by_house.keys())
        
        for house_id in sorted(all_house_ids):
            house_metadata = metadata.get(house_id)
            house_images = images_by_house.get(house_id)
            
            # Create default metadata if missing
            if house_metadata is None:
                print(f"Warning: No metadata for house {house_id}, using defaults")
                house_metadata = HouseMetadata(
                    house_id=house_id,
                    bedrooms=0,
                    bathrooms=0,
                    area=0,
                    zipcode=0,
                    price=0
                )
            
            # Create default images if missing
            if house_images is None:
                print(f"Warning: No images found for house {house_id}")
                house_images = HouseImages(house_id=house_id)
            
            house = House(
                house_id=house_id,
                metadata=house_metadata,
                images=house_images
            )
            houses.append(house)
        
        print(f"Successfully associated {len(houses)} houses with their images")
        return houses
    
    def save_to_json(self, houses: List[House], output_path: str):
        """
        Save house-image associations to JSON file.
        
        Args:
            houses: List of House objects
            output_path: Path to output JSON file
        """
        output_data = []
        
        for house in houses:
            house_dict = {
                "house_id": house.house_id,
                "metadata": asdict(house.metadata),
                "images": {
                    "bathroom": house.images.bathroom,
                    "bedroom": house.images.bedroom,
                    "kitchen": house.images.kitchen,
                    "frontal": house.images.frontal
                }
            }
            output_data.append(house_dict)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved associations to {output_path}")
    
    def get_statistics(self, houses: List[House]) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            houses: List of House objects
            
        Returns:
            Dictionary with statistics
        """
        total_houses = len(houses)
        houses_with_all_images = sum(
            1 for h in houses 
            if all([h.images.bathroom, h.images.bedroom, h.images.kitchen, h.images.frontal])
        )
        houses_with_some_images = sum(
            1 for h in houses 
            if any([h.images.bathroom, h.images.bedroom, h.images.kitchen, h.images.frontal])
        )
        
        image_counts = {
            "bathroom": sum(1 for h in houses if h.images.bathroom),
            "bedroom": sum(1 for h in houses if h.images.bedroom),
            "kitchen": sum(1 for h in houses if h.images.kitchen),
            "frontal": sum(1 for h in houses if h.images.frontal)
        }
        
        return {
            "total_houses": total_houses,
            "houses_with_all_images": houses_with_all_images,
            "houses_with_some_images": houses_with_some_images,
            "houses_with_no_images": total_houses - houses_with_some_images,
            "image_counts_by_type": image_counts,
            "total_images": sum(image_counts.values())
        }


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Associate houses with their images from the Houses dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="temp_dataset/Houses Dataset",
        help="Path to the 'Houses Dataset' directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="house_image_associations.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about the dataset"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = HouseImagePipeline(args.dataset_path)
    
    # Run pipeline
    houses = pipeline.associate_houses_with_images()
    
    # Save to JSON
    pipeline.save_to_json(houses, args.output)
    
    # Print statistics if requested
    if args.stats:
        stats = pipeline.get_statistics(houses)
        print("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

