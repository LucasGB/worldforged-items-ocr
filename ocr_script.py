import cv2
import numpy as np
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ItemStats:
    """Structured representation of item statistics"""
    name: str = ""
    item_type: str = ""
    subtype: str = ""
    damage_range: Optional[Tuple[int, int]] = None
    dps: Optional[float] = None
    speed: Optional[float] = None
    level_requirement: Optional[int] = None
    item_level: Optional[int] = None
    primary_stats: Dict[str, int] = None
    secondary_stats: Dict[str, int] = None
    effects: List[str] = None
    binding: str = ""
    rarity: str = ""
    weapon_type: str = ""
    slot: str = ""
    
    def __post_init__(self):
        if self.primary_stats is None:
            self.primary_stats = {}
        if self.secondary_stats is None:
            self.secondary_stats = {}
        if self.effects is None:
            self.effects = []

class GameItemOCRProcessor:
    """Production-grade OCR processor for game items"""
    
    def __init__(self, config_path: str = "ocr_config.yaml"):
        self.ocr = None
        self.config = self._load_config(config_path)
        self._initialize_ocr()
        self._compile_patterns()
        self._load_vocabularies()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            'ocr_settings': {
                'use_textline_orientation': True,
                'lang': 'en',
                'det_db_thresh': 0.3,
                'det_db_box_thresh': 0.6,
                'rec_batch_num': 6
            },
            'preprocessing': {
                'resize_factor': 2.0,
                'gaussian_blur_kernel': 3,
                'adaptive_threshold_block_size': 11,
                'adaptive_threshold_c': 2
            },
            'text_correction': {
                'confidence_threshold': 0.7,
                'enable_spell_check': True,
                'enable_context_correction': True
            },
            'item_parsing': {
                'min_name_length': 3,
                'max_name_length': 50,
                'stat_confidence_threshold': 0.8
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    config[key] = {**value, **config[key]}
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def _initialize_ocr(self):
        """Initialize PaddleOCR with optimized settings"""
        try:
            self.ocr = PaddleOCR(**self.config['ocr_settings'])
            logger.info("OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            raise
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text parsing"""
        self.patterns = {
            # Damage patterns
            'damage_range': re.compile(r'(\d+)\s*[-–—]\s*(\d+)\s+[Dd]amage'),
            'single_damage': re.compile(r'(\d+)\s+[Dd]amage'),
            'dps': re.compile(r'\(?([\d.]+)\s+damage\s+per\s+second\)?', re.IGNORECASE),
            
            # Speed and timing
            'speed': re.compile(r'[Ss]peed\s+([\d.]+)'),
            'cooldown': re.compile(r'(\d+)\s+sec\s+cooldown', re.IGNORECASE),
            
            # Level requirements
            'level_req': re.compile(r'[Rr]equires\s+[Ll]evel\s+(\d+)'),
            'item_level': re.compile(r'[Ii]tem\s+[Ll]evel\s+(\d+)'),
            
            # Primary stats (Strength, Agility, etc.)
            'primary_stats': re.compile(r'[+]?(\d+)\s+(Strength|Agility|Intellect|Stamina|Spirit)', re.IGNORECASE),
            
            # Secondary stats (ratings, resistances)
            'secondary_stats': re.compile(r'[+]?(\d+)\s+(Critical Strike|Haste|Mastery|Versatility|dodge|parry|block|hit|expertise)', re.IGNORECASE),
            'rating_stats': re.compile(r'(\d+)\s+(\w+)\s+rating', re.IGNORECASE),
            
            # Item types and slots
            'weapon_types': re.compile(r'\b(Sword|Axe|Mace|Dagger|Staff|Bow|Gun|Crossbow|Wand|Fist Weapon|Polearm|Thrown)\b', re.IGNORECASE),
            'armor_types': re.compile(r'\b(Cloth|Leather|Mail|Plate)\b'),
            'slot': re.compile(r'\b(Main\s*[Hh]and|Off\s*[Hh]and|Two-Hand|Head|Neck|Shoulder|Back|Chest|Wrist|Hands|Waist|Legs|Feet|Finger|Trinket|Relic)\b', re.IGNORECASE),
            'slot_types': re.compile(r'\b(Cloth|Leather|Mail|Plate|Sword|Axe|Mace|Dagger|Staff|Bow|Gun|Crossbow|Wand|Fist Weapon|Polearm|Thrown)\b', re.IGNORECASE),

            # Binding and rarity
            'binding': re.compile(r'\b(Binds when picked up|Binds when equipped|Binds to account|Soulbound|Account Bound)\b', re.IGNORECASE),
            'rarity': re.compile(r'\b(Poor|Common|Uncommon|Rare|Epic|Legendary|Artifact|Heirloom)\b', re.IGNORECASE),
            
            # Effects (Equip, Use, Chance on hit, etc.)
            'equip_effect': re.compile(r'^[Ee]quip:\s*(.+)$'),
            'use_effect': re.compile(r'^[Uu]se:\s*(.+)$'),
            'chance_effect': re.compile(r'[Cc]hance\s+(?:on\s+hit|to|when)\s+(.+)', re.IGNORECASE),
            
            # Common OCR errors
            'number_fixes': [
                (r'\b1([1l])[o0]\b', r'110'),  # 11o -> 110, 1l0 -> 110
                (r'\b([2-9])[o0]([0-9])\b', r'\1\2'),  # 2o5 -> 205
                (r'\bO(\d+)\b', r'0\1'),  # O5 -> 05
                (r'\b([0-9]+)[Il1]\b', r'\g<1>1'),  # 51l -> 511
            ]
        }
    
    def _load_vocabularies(self):
        """Load game-specific vocabularies for spell checking"""
        # In production, these would be loaded from external files
        self.vocabularies = {
            'item_names': set(),  # Would be populated from database
            'stats': {
                'strength', 'agility', 'intellect', 'stamina', 'spirit',
                'critical', 'strike', 'haste', 'mastery', 'versatility',
                'dodge', 'parry', 'block', 'hit', 'expertise', 'resilience'
            },
            'weapon_types': {
                'sword', 'axe', 'mace', 'dagger', 'staff', 'bow', 'gun',
                'crossbow', 'wand', 'fist', 'weapon', 'polearm', 'thrown'
            },
            'common_words': {
                'damage', 'speed', 'level', 'requires', 'equip', 'use',
                'chance', 'increases', 'decreases', 'rating', 'resistance'
            }
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Apply preprocessing to improve OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for better text recognition
        resize_factor = self.config['preprocessing']['resize_factor']
        if resize_factor != 1.0:
            height, width = gray.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Noise reduction
        kernel_size = self.config['preprocessing']['gaussian_blur_kernel']
        if kernel_size > 0:
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Adaptive thresholding for better text contrast
        block_size = self.config['preprocessing']['adaptive_threshold_block_size']
        c_value = self.config['preprocessing']['adaptive_threshold_c']
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, c_value)
        
        if self.config['preprocessing'].get('invert_colors', False):
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def correct_ocr_text(self, text: str, confidence: float = 1.0) -> str:
        """Apply intelligent text correction"""
        if not text.strip():
            return text
            
        original_text = text
        
        # Apply number pattern fixes
        for pattern, replacement in self.patterns['number_fixes']:
            text = re.sub(pattern, replacement, text)
        
        # Context-aware corrections
        if self.config['text_correction']['enable_context_correction']:
            text = self._apply_context_corrections(text)
        
        # Spell checking for known vocabulary
        if (self.config['text_correction']['enable_spell_check'] and 
            confidence < self.config['text_correction']['confidence_threshold']):
            text = self._spell_check(text)
        
        if text != original_text:
            logger.debug(f"Text corrected: '{original_text}' -> '{text}'")
        
        return text
    
    def _apply_context_corrections(self, text: str) -> str:
        """Apply context-aware corrections based on game terminology"""
        corrections = {
            # Common OCR mistakes in game text
            'rainted': 'Tainted',
            'Binds when plcked up': 'Binds when picked up',
            'Binds when equlpped': 'Binds when equipped',
            'Mana per 5': 'Mana per 5 sec',
            'per second': 'per second',
            'Equlp': 'Equip',
            'ltem Level': 'Item Level',
            'Requlres': 'Requires',
            'Unlque': 'Unique',
            'Armour': 'Armor',  # Handle regional spellings
        }
        
        for mistake, correction in corrections.items():
            text = text.replace(mistake, correction)
        
        return text
    
    def _spell_check(self, text: str) -> str:
        """Simple spell checking against game vocabulary"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.vocabularies['stats'] or \
               clean_word in self.vocabularies['slot_types'] or \
               clean_word in self.vocabularies['common_words']:
                corrected_words.append(word)
            else:
                # Find best match in vocabularies
                best_match = self._find_best_vocabulary_match(clean_word)
                if best_match and self._similarity_score(clean_word, best_match) > 0.8:
                    # Preserve original case pattern
                    corrected_word = self._preserve_case(word, best_match)
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _find_best_vocabulary_match(self, word: str) -> Optional[str]:
        """Find best matching word in vocabularies"""
        all_vocab = set()
        for vocab in self.vocabularies.values():
            if isinstance(vocab, set):
                all_vocab.update(vocab)
        
        best_match = None
        best_score = 0
        
        for vocab_word in all_vocab:
            score = self._similarity_score(word, vocab_word)
            if score > best_score:
                best_score = score
                best_match = vocab_word
        
        return best_match if best_score > 0.6 else None
    
    def _similarity_score(self, a: str, b: str) -> float:
        """Calculate similarity score between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        """Preserve the case pattern of original word in replacement"""
        if original.isupper():
            return replacement.upper()
        elif original.islower():
            return replacement.lower()
        elif original.istitle():
            return replacement.title()
        else:
            return replacement
    
    def extract_text_with_confidence(self, image_path: str, use_preprocessing: bool = True) -> List[Tuple[str, float]]:
        """Extract text with confidence scores"""
        try:
            if use_preprocessing:
                # Save preprocessed image temporarily
                processed_img = self.preprocess_image(image_path)
                prefix = "white_bkground" if self.config['preprocessing'].get('invert_colors', False) else "black_bkground"
                temp_path = f"{prefix}_temp_processed_{Path(image_path).name}"
                cv2.imwrite(temp_path, processed_img)
                result = self.ocr.predict(temp_path)
                # Path(temp_path).unlink()  # Clean up
            else:
                result = self.ocr.predict(image_path)
            
            if not result or not hasattr(result[0], 'json'):
                logger.warning(f"No OCR results for {image_path}")
                return []
            
            json_data = result[0].json
            texts = json_data['res']['rec_texts']
            scores = json_data['res'].get('rec_scores', [1.0] * len(texts))
            
            # Ensure we have confidence scores for all texts
            if len(scores) < len(texts):
                scores.extend([1.0] * (len(texts) - len(scores)))
            
            # Apply text correction
            corrected_texts = []
            for text, confidence in zip(texts, scores):
                corrected_text = self.correct_ocr_text(text, confidence)
                corrected_texts.append((corrected_text, confidence))
            
            return corrected_texts
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return []
    
    def parse_item_stats(self, text_data: List[Tuple[str, float]]) -> ItemStats:
        """Parse extracted text into structured item statistics"""
        item = ItemStats()
        text_lines = [text for text, _ in text_data]
        
        # Find item name (heuristic: first non-binding, non-type line with decent length)
        for text, confidence in text_data:
            if (not item.name and 
                len(text.strip()) >= self.config['item_parsing']['min_name_length'] and
                len(text.strip()) <= self.config['item_parsing']['max_name_length'] and
                not any(pattern.search(text) for pattern in [
                    self.patterns['binding'], self.patterns['slot_types'], 
                    self.patterns['slot']]) and
                not text.strip().lower().startswith(('equip:', 'use:', 'requires'))):
                item.name = text.strip()
                break
        
        # Process each line for various attributes
        for text, confidence in text_data:
            if confidence < self.config['item_parsing']['stat_confidence_threshold']:
                logger.debug(f"Skipping low confidence text: {text} ({confidence:.3f})")
                continue
            
            # Damage parsing
            damage_match = self.patterns['damage_range'].search(text)
            if damage_match:
                item.damage_range = (int(damage_match.group(1)), int(damage_match.group(2)))
            
            # DPS parsing
            dps_match = self.patterns['dps'].search(text)
            if dps_match:
                item.dps = float(dps_match.group(1))
            
            # Speed parsing
            speed_match = self.patterns['speed'].search(text)
            if speed_match:
                item.speed = float(speed_match.group(1))
            
            # Level requirements
            level_req_match = self.patterns['level_req'].search(text)
            if level_req_match:
                item.level_requirement = int(level_req_match.group(1))
            
            item_level_match = self.patterns['item_level'].search(text)
            if item_level_match:
                item.item_level = int(item_level_match.group(1))
            
            # Primary stats
            for match in self.patterns['primary_stats'].finditer(text):
                value, stat = match.groups()
                item.primary_stats[stat.lower()] = int(value)
            
            # Secondary stats
            for match in self.patterns['secondary_stats'].finditer(text):
                value, stat = match.groups()
                item.secondary_stats[stat.lower()] = int(value)
            
            # Rating stats
            for match in self.patterns['rating_stats'].finditer(text):
                value, stat = match.groups()
                item.secondary_stats[f"{stat.lower()}_rating"] = int(value)
            
            slot_match = self.patterns['slot'].search(text)
            if slot_match:
                item.slot = slot_match.group(1)
            
            # Binding
            binding_match = self.patterns['binding'].search(text)
            if binding_match:
                item.binding = binding_match.group(1)
            
            # Effects
            equip_match = self.patterns['equip_effect'].search(text)
            if equip_match:
                item.effects.append(f"Equip: {equip_match.group(1)}")
            
            use_match = self.patterns['use_effect'].search(text)
            if use_match:
                item.effects.append(f"Use: {use_match.group(1)}")
            
            chance_match = self.patterns['chance_effect'].search(text)
            if chance_match:
                item.effects.append(f"Chance: {chance_match.group(1)}")
        
        return item
    
    def process_item_image(self, image_path: str, save_debug: bool = False) -> Dict[str, Any]:
        """Process a single item image and return structured data"""
        logger.info(f"Processing item image: {image_path}")
        
        # Extract text with confidence
        text_data = self.extract_text_with_confidence(image_path)
        
        if not text_data:
            logger.warning(f"No text extracted from {image_path}")
            return {"error": "No text extracted", "image_path": image_path}
        
        # Parse into structured format
        item_stats = self.parse_item_stats(text_data)
        
        result = {
            "image_path": image_path,
            "raw_text": [(text, conf) for text, conf in text_data],
            "item_stats": asdict(item_stats),
            "processing_metadata": {
                "text_lines_count": len(text_data),
                "avg_confidence": sum(conf for _, conf in text_data) / len(text_data) if text_data else 0,
                "preprocessed": True
            }
        }
        
        if save_debug:
            debug_path = f"debug_{Path(image_path).stem}.json"
            with open(debug_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Debug info saved to {debug_path}")
        
        return result
    
    def batch_process(self, image_paths: List[str], output_path: str = "processed_items.json") -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        results = []
        failed_count = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_item_image(image_path)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                failed_count += 1
                results.append({
                    "error": str(e),
                    "image_path": image_path
                })
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch processing complete. {len(results) - failed_count}/{len(image_paths)} successful. Results saved to {output_path}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = GameItemOCRProcessor()
    
    # Process single image
    # result = processor.process_item_image("image.png", save_debug=True)
    # print("Item Name:", result["item_stats"]["name"])
    # print("Damage Range:", result["item_stats"]["damage_range"])
    # print("Primary Stats:", result["item_stats"]["primary_stats"])
    
    # For batch processing (example)
    image_files = list(Path("images/").glob("*.png"))
    results = processor.batch_process([str(p) for p in image_files])