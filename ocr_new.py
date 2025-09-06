import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoWTooltipOCR:
    def __init__(self, lang='en'):
        """Initialize the WoW Tooltip OCR processor with PaddleOCR"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        
        # Color ranges for WoW item quality/text types (HSV)
        self.color_ranges = {
            'white': [(0, 0, 180), (180, 30, 255)],      # White text
            'blue': [(100, 50, 50), (130, 255, 255)],    # Rare items
            'green': [(40, 50, 50), (80, 255, 255)],     # Uncommon/equip text
            'purple': [(120, 50, 50), (160, 255, 255)],  # Epic items
            'orange': [(10, 100, 100), (25, 255, 255)],  # Legendary
            'red': [(0, 100, 100), (10, 255, 255)],      # Requirements/errors
            'yellow': [(20, 100, 100), (30, 255, 255)]   # Flavor text
        }
        
        # Regex patterns for data extraction
        self.patterns = {
            'item_level': r'Item Level (\d+)',
            'damage': r'(\d+)\s*-\s*(\d+)\s*Damage',
            'dps': r'\((\d+\.?\d*)\s*damage per second\)',
            'speed': r'Speed\s*(\d+\.?\d*)',
            'stats': r'\+(\d+)\s+(\w+)',
            'requirements': r'Requires Level (\d+)',
            'item_id': r'ID\s*(\d+)',
            'equip_effects': r'Equip:\s*(.+?)(?=\n|$)',
            'sell_price': r'Sell Price:\s*(\d+)',
        }

    def upscale_image(self, image: np.ndarray, scale_factor: float = 3.0) -> np.ndarray:
        """Upscale image using bicubic interpolation for better OCR accuracy"""
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        upscaled = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
        logger.info(f"Upscaled from {width}x{height} to {new_width}x{new_height}")
        return upscaled

    def extract_color_mask(self, image_hsv: np.ndarray, color_name: str) -> np.ndarray:
        """Extract mask for specific color range"""
        if color_name not in self.color_ranges:
            return None
            
        lower, upper = self.color_ranges[color_name]
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        mask = cv2.inRange(image_hsv, lower, upper)
        return mask

    def subtract_background(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Remove gradient background using morphological opening"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        result = cv2.subtract(gray, background)
        return result

    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
        """Apply CLAHE for local contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to reduce noise while preserving edges"""
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Try both adaptive methods and pick the best
        thresh_mean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        thresh_gaussian = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Use Gaussian as default (generally better for text)
        return thresh_gaussian

    def morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Light morphological operations to clean up text"""
        # Opening to remove small noise/glow artifacts
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
        
        # Gentle closing to connect fragmented characters (if needed)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        return closed

    def preprocess_image(self, image: np.ndarray, color_filter: str = None) -> np.ndarray:
        """Complete preprocessing pipeline following our refined strategy"""
        logger.info(f"Starting preprocessing - Input shape: {image.shape}, dtype: {image.dtype}")
        
        # Step 1: Upscale first
        processed = self.upscale_image(image, scale_factor=3.0)
        logger.info(f"After upscale: {processed.shape}")
        
        # Step 2: HSV color filtering (if specified)
        if color_filter:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            mask = self.extract_color_mask(hsv, color_filter)
            if mask is not None:
                # Apply mask to original image
                processed = cv2.bitwise_and(processed, processed, mask=mask)
                logger.info(f"Applied {color_filter} color filter: {processed.shape}")
        
        # Step 3: Background subtraction
        processed = self.subtract_background(processed)
        logger.info(f"After background subtraction: {processed.shape}")
        
        # Step 4: Contrast enhancement
        processed = self.enhance_contrast(processed)
        logger.info(f"After contrast enhancement: {processed.shape}")
        
        # Step 5: Noise reduction
        processed = self.reduce_noise(processed)
        logger.info(f"After noise reduction: {processed.shape}")
        
        # Step 6: Adaptive thresholding
        processed = self.adaptive_threshold(processed)
        logger.info(f"After thresholding: {processed.shape}")
        
        # Step 7: Light morphological cleanup
        processed = self.morphological_cleanup(processed)
        logger.info(f"Final processed shape: {processed.shape}, dtype: {processed.dtype}")
        
        return processed

    def detect_text_blocks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment tooltip into semantic blocks using color-based detection"""
        blocks = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract different colored text regions
        for color_name in ['blue', 'white', 'green', 'red', 'purple', 'orange']:
            mask = self.extract_color_mask(hsv, color_name)
            if mask is not None and np.sum(mask) > 100:  # Minimum pixels threshold
                # Find bounding boxes for this color
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get bounding box that encompasses all contours of this color
                    all_points = np.vstack(contours)
                    x, y, w, h = cv2.boundingRect(all_points)
                    
                    # Extract region with some padding
                    pad = 10
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(image.shape[1] - x, w + 2*pad)
                    h = min(image.shape[0] - y, h + 2*pad)
                    
                    blocks[color_name] = image[y:y+h, x:x+w]
                    logger.info(f"Extracted {color_name} text block: {w}x{h}")
        
        return blocks

    def run_ocr_on_image(self, image: np.ndarray) -> List[Dict]:
        """Run PaddleOCR on preprocessed image"""
        try:
            results = self.ocr.predict(image)
            if results and results[0]:
                return results[0]
            return []
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def extract_structured_data(self, ocr_results: List[Dict], color_context: str = None) -> Dict:
        """Extract structured data from OCR results using regex patterns"""
        # Combine all text from OCR results
        full_text = []
        for result in ocr_results:
            if len(result) >= 2:  # [bbox, (text, confidence)]
                text = result[1][0] if isinstance(result[1], tuple) else result[1]
                full_text.append(text)
        
        combined_text = ' '.join(full_text)
        logger.info(f"Combined text ({color_context}): {combined_text[:100]}...")
        
        extracted_data = {
            'raw_text': combined_text,
            'color_context': color_context
        }
        
        # Apply regex patterns to extract structured data
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                if pattern_name == 'damage' and len(matches[0]) == 2:
                    extracted_data['damage_min'] = int(matches[0][0])
                    extracted_data['damage_max'] = int(matches[0][1])
                elif pattern_name == 'stats':
                    stats = {}
                    for match in matches:
                        stat_value, stat_name = match
                        stats[stat_name.lower()] = int(stat_value)
                    extracted_data['stats'] = stats
                elif isinstance(matches[0], tuple):
                    extracted_data[pattern_name] = matches[0]
                else:
                    extracted_data[pattern_name] = matches[0]
        
        return extracted_data

    def process_tooltip(self, image_path: str) -> Dict:
        """Main processing function for a single tooltip image"""
        logger.info(f"Processing tooltip: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Strategy 1: Process whole image with different color filters
        results = {}
        
        # Process without color filter (all text)
        processed_full = self.preprocess_image(image.copy())
        ocr_results_full = self.run_ocr_on_image(processed_full)
        results['full_image'] = self.extract_structured_data(ocr_results_full, 'full')
        
        # Process with color-specific filters
        for color in ['blue', 'white', 'green', 'red', 'purple']:
            try:
                processed_color = self.preprocess_image(image.copy(), color_filter=color)
                ocr_results_color = self.run_ocr_on_image(processed_color)
                if ocr_results_color:  # Only add if we got results
                    results[f'{color}_text'] = self.extract_structured_data(
                        ocr_results_color, color
                    )
            except Exception as e:
                logger.warning(f"Failed to process {color} filter: {e}")
        
        # Strategy 2: Process text blocks separately
        try:
            blocks = self.detect_text_blocks(image)
            for block_name, block_image in blocks.items():
                processed_block = self.preprocess_image(block_image)
                ocr_results_block = self.run_ocr_on_image(processed_block)
                if ocr_results_block:
                    results[f'{block_name}_block'] = self.extract_structured_data(
                        ocr_results_block, f'{block_name}_block'
                    )
        except Exception as e:
            logger.warning(f"Block processing failed: {e}")
        
        return {
            'image_path': image_path,
            'processing_results': results,
            'timestamp': str(datetime.datetime.now())
        }

    def merge_results(self, results: Dict) -> Dict:
        """Merge results from different processing strategies into final structure"""
        merged = {
            'item_name': None,
            'item_type': None,
            'item_level': None,
            'quality': None,
            'damage': {},
            'stats': {},
            'requirements': {},
            'equip_effects': [],
            'item_id': None,
            'sell_price': None,
            'raw_ocr_data': results
        }
        
        # Priority order for data extraction (most reliable first)
        priority_sources = ['blue_text', 'white_text', 'full_image', 'green_text']
        
        for source in priority_sources:
            if source in results['processing_results']:
                data = results['processing_results'][source]
                
                # Extract item level
                if 'item_level' in data and not merged['item_level']:
                    merged['item_level'] = data['item_level']
                
                # Extract damage info
                if 'damage_min' in data and 'damage_max' in data:
                    merged['damage'] = {
                        'min': data['damage_min'],
                        'max': data['damage_max']
                    }
                if 'dps' in data:
                    merged['damage']['dps'] = float(data['dps'])
                if 'speed' in data:
                    merged['damage']['speed'] = float(data['speed'])
                
                # Extract stats
                if 'stats' in data:
                    merged['stats'].update(data['stats'])
                
                # Extract requirements
                if 'requirements' in data:
                    merged['requirements']['level'] = int(data['requirements'])
                
                # Extract item ID
                if 'item_id' in data and not merged['item_id']:
                    merged['item_id'] = data['item_id']
                
                # Extract sell price
                if 'sell_price' in data and not merged['sell_price']:
                    merged['sell_price'] = data['sell_price']
                
                # Extract equip effects (from green text usually)
                if 'equip_effects' in data:
                    if isinstance(data['equip_effects'], list):
                        merged['equip_effects'].extend(data['equip_effects'])
                    else:
                        merged['equip_effects'].append(data['equip_effects'])
        
        # Try to determine item name from raw text (usually first line)
        for source in ['blue_text', 'full_image']:
            if source in results['processing_results']:
                raw_text = results['processing_results'][source].get('raw_text', '')
                lines = raw_text.split('\n')
                if lines and lines[0].strip():
                    # Clean up item name
                    item_name = lines[0].strip()
                    # Remove common prefixes/suffixes
                    item_name = re.sub(r'^(Binds when picked up|Item Level \d+)', '', item_name).strip()
                    if item_name and not merged['item_name']:
                        merged['item_name'] = item_name
                        break
        
        # Determine quality based on processing results
        if 'purple_text' in results['processing_results']:
            merged['quality'] = 'Epic'
        elif 'blue_text' in results['processing_results']:
            merged['quality'] = 'Rare'
        elif 'green_text' in results['processing_results']:
            merged['quality'] = 'Uncommon'
        else:
            merged['quality'] = 'Common'
        
        return merged

    def process_batch(self, image_folder: str, output_file: str = 'wow_tooltips.json'):
        """Process multiple tooltip images and save to JSON"""
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise ValueError(f"Image folder not found: {image_folder}")
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = [
            f for f in image_folder.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {image_folder}")
        
        logger.info(f"Found {len(image_files)} images to process")
        
        processed_items = []
        
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
                
                # Process the tooltip
                raw_results = self.process_tooltip(str(image_file))
                
                # Merge and structure the results
                structured_item = self.merge_results(raw_results)
                
                processed_items.append(structured_item)
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                # Add error entry
                processed_items.append({
                    'image_path': str(image_file),
                    'error': str(e),
                    'processing_failed': True
                })
        
        # Save results to JSON
        output_data = {
            'metadata': {
                'total_processed': len(image_files),
                'successful': len([item for item in processed_items if 'error' not in item]),
                'failed': len([item for item in processed_items if 'error' in item]),
                'processing_timestamp': str(datetime.datetime.now())
            },
            'items': processed_items
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        return output_data

def main():
    """Main execution function"""
    output_path = 'wow_tooltips.json'
    
    # Initialize OCR processor
    ocr_processor = WoWTooltipOCR()
    
    input_path = Path("./images")
    
    if input_path.is_file():
        # Process single image
        logger.info("Processing single image...")
        try:
            raw_results = ocr_processor.process_tooltip(str(input_path))
            structured_item = ocr_processor.merge_results(raw_results)
            
            output_data = {
                'metadata': {
                    'total_processed': 1,
                    'processing_timestamp': str(datetime.datetime.now())
                },
                'items': [structured_item]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            
    elif input_path.is_dir():
        # Process batch
        logger.info("Processing image batch...")
        ocr_processor.process_batch(str(input_path), output_path)
    else:
        logger.error(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    main()