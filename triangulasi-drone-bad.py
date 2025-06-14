#!/usr/bin/env python3
"""
Tree Detection and Height Estimation System
==========================================

This module provides functionality for detecting trees in drone imagery,
estimating their heights using stereo vision techniques, and calculating
crown areas.

Author: Generated Code
Version: 2.0
"""

import os
import sys
import logging
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TreeDetectionConfig:
    """Configuration class for tree detection parameters."""
    
    def __init__(self):
        # Paths
        self.DATASET_PATH = Path('/home/ubuntu/CODE/PRATAPY/dataset-harist/DATASET_GK')
        self.MODEL_PATH = Path('runs/detect/treetopTrain/weights/best.pt')
        
        # Camera and sensor parameters
        self.DRONE_HEIGHT_M = 98.7
        self.GSD_CM_PER_PIX = 2.64  # Ground Sample Distance (cm/pixel)
        self.GSD_M_PER_PIX = self.GSD_CM_PER_PIX / 100.0
        
        # Camera specifications
        self.FOCAL_LENGTH_MM = 8.8
        self.PIXEL_SIZE_UM = 2.61
        self.IMG_RESOLUTION = (4864, 3648)
        
        # Color thresholds for crown detection (HSV)
        self.CROWN_COLOR_LOWER = np.array([25, 40, 40])
        self.CROWN_COLOR_UPPER = np.array([85, 255, 255])
        
        # Output settings
        self.OUTPUT_PREFIX = 'detected_'
        
    @property
    def focal_length_pixels(self) -> float:
        """Calculate focal length in pixels."""
        return (self.FOCAL_LENGTH_MM * 1000) / self.PIXEL_SIZE_UM


class GPSMetadata:
    """Class to handle GPS metadata extraction and processing."""
    
    @staticmethod
    def extract_gps_info(image_path: Path) -> Optional[Dict[str, float]]:
        """
        Extract GPS and altitude information from image EXIF data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing latitude, longitude, and altitude, or None if extraction fails
        """
        try:
            with Image.open(image_path) as image:
                exif = image._getexif()
                if not exif:
                    logger.warning(f"No EXIF data found in {image_path}")
                    return None
                
                gps_info = {}
                for tag_id, data in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo':
                        for gps_tag, gps_data in data.items():
                            sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                            gps_info[sub_tag] = gps_data
                
                return GPSMetadata._parse_coordinates(gps_info)
                
        except Exception as e:
            logger.error(f"Error reading EXIF data from {image_path}: {e}")
            return None
    
    @staticmethod
    def _parse_coordinates(gps_info: Dict) -> Optional[Dict[str, float]]:
        """Parse GPS coordinates from EXIF GPS info."""
        required_keys = ['GPSLatitude', 'GPSLongitude']
        if not all(key in gps_info for key in required_keys):
            return None
        
        # Convert latitude
        lat = GPSMetadata._convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        
        # Convert longitude
        lng = GPSMetadata._convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef') == 'W':
            lng = -lng
        
        # Extract altitude
        altitude = None
        if 'GPSAltitude' in gps_info:
            try:
                altitude = float(gps_info['GPSAltitude'])
            except (ValueError, TypeError):
                logger.warning("Could not parse GPS altitude")
        
        return {
            'latitude': lat,
            'longitude': lng,
            'altitude': altitude
        }
    
    @staticmethod
    def _convert_to_degrees(value: Tuple[float, float, float]) -> float:
        """Convert GPS coordinates from degrees, minutes, seconds to decimal degrees."""
        degrees, minutes, seconds = map(float, value)
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    @staticmethod
    def calculate_distance(coord1: Dict[str, float], coord2: Dict[str, float]) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula.
        
        Args:
            coord1: First coordinate dict with 'latitude' and 'longitude'
            coord2: Second coordinate dict with 'latitude' and 'longitude'
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        lat1, lon1 = radians(coord1['latitude']), radians(coord1['longitude'])
        lat2, lon2 = radians(coord2['latitude']), radians(coord2['longitude'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c


class TreeDetector:
    """Main class for tree detection and height estimation."""
    
    def __init__(self, config: TreeDetectionConfig):
        self.config = config
        self.model = self._load_model()
    
    def _load_model(self) -> YOLO:
        """Load the YOLO model for tree detection."""
        if not self.config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.MODEL_PATH}")
        
        logger.info(f"Loading model from {self.config.MODEL_PATH}")
        return YOLO(str(self.config.MODEL_PATH))
    
    def detect_trees(self, image_path: Path) -> Tuple[np.ndarray, Any]:
        """
        Detect trees in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (bounding boxes array, detection results)
        """
        try:
            results = self.model(str(image_path))
            boxes = results[0].boxes.xyxy.cpu().numpy()
            logger.info(f"Detected {len(boxes)} trees in {image_path.name}")
            return boxes, results
        except Exception as e:
            logger.error(f"Error detecting trees in {image_path}: {e}")
            return np.array([]), None
    
    def match_trees(self, boxes1: np.ndarray, boxes2: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Match trees between two images.
        
        Note: This is a simple implementation that matches by order.
        A more sophisticated approach would use Hungarian matching algorithm.
        
        Args:
            boxes1: Bounding boxes from first image
            boxes2: Bounding boxes from second image
            
        Returns:
            List of matched box pairs
        """
        n_matches = min(len(boxes1), len(boxes2))
        matches = [(boxes1[i], boxes2[i]) for i in range(n_matches)]
        
        if len(boxes1) != len(boxes2):
            logger.warning(f"Different number of trees detected: {len(boxes1)} vs {len(boxes2)}")
        
        return matches
    
    def estimate_height(self, bbox1: np.ndarray, bbox2: np.ndarray, 
                       baseline: float, altitude: float) -> Optional[float]:
        """
        Estimate tree height using stereo vision.
        
        Args:
            bbox1: Bounding box from first image
            bbox2: Bounding box from second image
            baseline: Distance between camera positions (meters)
            altitude: Camera altitude (meters)
            
        Returns:
            Estimated tree height in meters, or None if calculation fails
        """
        if not baseline or not altitude:
            return None
        
        # Calculate center points
        x1 = (bbox1[0] + bbox1[2]) / 2
        x2 = (bbox2[0] + bbox2[2]) / 2
        disparity_pixels = abs(x1 - x2)
        
        if disparity_pixels == 0:
            logger.warning("Zero disparity detected, cannot estimate height")
            return None
        
        # Calculate depth using stereo vision formula
        focal_pixels = self.config.focal_length_pixels
        depth = (focal_pixels * baseline) / disparity_pixels
        height = altitude - depth
        
        if height < 0:
            logger.warning(f"Negative height calculated: {height:.2f}m")
            return None
        
        return height
    
    def calculate_crown_area(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate crown area using color-based segmentation.
        
        Args:
            image: Input image
            bbox: Tree bounding box
            
        Returns:
            Tuple of (mask, area in square meters)
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        roi = image[y1:y2, x1:x2]
        
        # Convert to HSV and apply color thresholding
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.CROWN_COLOR_LOWER, self.config.CROWN_COLOR_UPPER)
        
        # Calculate area
        area_pixels = cv2.countNonZero(mask)
        area_m2 = area_pixels * (self.config.GSD_M_PER_PIX ** 2)
        
        return mask, area_m2
    
    def process_image_pair(self, img1_path: Path, img2_path: Path) -> Optional[Dict]:
        """
        Process a pair of images for tree detection and height estimation.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Dictionary containing processing results, or None if processing fails
        """
        logger.info(f"Processing pair: {img1_path.name} - {img2_path.name}")
        
        # Extract GPS metadata
        meta1 = GPSMetadata.extract_gps_info(img1_path)
        meta2 = GPSMetadata.extract_gps_info(img2_path)
        
        if not meta1 or not meta2:
            logger.error(f"Missing GPS metadata for image pair")
            return None
        
        # Calculate baseline from GPS coordinates
        baseline = GPSMetadata.calculate_distance(meta1, meta2)
        altitude = (meta1['altitude'] + meta2['altitude']) / 2 if meta1['altitude'] and meta2['altitude'] else None
        
        if not altitude:
            logger.warning("Using default drone height due to missing altitude data")
            altitude = self.config.DRONE_HEIGHT_M
        
        # Detect trees
        boxes1, results1 = self.detect_trees(img1_path)
        boxes2, results2 = self.detect_trees(img2_path)
        
        if len(boxes1) == 0 or len(boxes2) == 0:
            logger.warning("No trees detected in one or both images")
            return None
        
        # Match trees
        matches = self.match_trees(boxes1, boxes2)
        
        return {
            'matches': matches,
            'baseline': baseline,
            'altitude': altitude,
            'metadata': {'img1': meta1, 'img2': meta2},
            'boxes': {'img1': boxes1, 'img2': boxes2}
        }
    
    def visualize_results(self, image_path: Path, processing_result: Dict) -> None:
        """
        Visualize detection results on the image.
        
        Args:
            image_path: Path to the image to annotate
            processing_result: Results from process_image_pair
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        for i, (bbox1, bbox2) in enumerate(processing_result['matches']):
            # Estimate height
            height = self.estimate_height(
                bbox1, bbox2, 
                processing_result['baseline'], 
                processing_result['altitude']
            )
            
            # Calculate crown area
            _, area_m2 = self.calculate_crown_area(image, bbox1)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox1[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            label = f"Tree {i+1}: "
            if height is not None:
                label += f"H={height:.1f}m, "
            else:
                label += "H=N/A, "
            label += f"Crown={area_m2:.1f}m²"
            
            # Add text with background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save annotated image
        output_path = image_path.parent / f"{self.config.OUTPUT_PREFIX}{image_path.name}"
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved annotated image: {output_path}")
    
    def process_dataset(self) -> None:
        """Process all image pairs in the dataset directory."""
        if not self.config.DATASET_PATH.exists():
            logger.error(f"Dataset directory not found: {self.config.DATASET_PATH}")
            return
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg'}
        image_files = [
            f for f in self.config.DATASET_PATH.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.error(f"No image files found in {self.config.DATASET_PATH}")
            return
        
        image_files.sort()
        logger.info(f"Found {len(image_files)} images in dataset")
        
        # Process consecutive pairs
        pairs_processed = 0
        for i in range(0, len(image_files) - 1, 2):
            img1_path = image_files[i]
            img2_path = image_files[i + 1]
            
            try:
                result = self.process_image_pair(img1_path, img2_path)
                if result:
                    self.visualize_results(img1_path, result)
                    pairs_processed += 1
                else:
                    logger.warning(f"Failed to process pair: {img1_path.name} - {img2_path.name}")
            except Exception as e:
                logger.error(f"Error processing pair {img1_path.name} - {img2_path.name}: {e}")
        
        logger.info(f"Successfully processed {pairs_processed} image pairs")


def main():
    """Main function to run the tree detection system."""
    try:
        config = TreeDetectionConfig()
        detector = TreeDetector(config)
        detector.process_dataset()
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()