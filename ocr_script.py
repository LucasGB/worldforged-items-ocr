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
from item_stats import ItemStats, DamageInfo
from difflib import get_close_matches
from rapidfuzz import fuzz, process

canonical_words = [
    "Main-Hand",
    "Off-Hand",
    "One-Hand",
    "Two-Hand",
    "Wand",
    "Bow",
    "Crossbow",
    "Thrown",
    "Gun",
]

item_slots = [
    "Head", "Neck", "Shoulder", "Back", "Chest", "Wrist", "Hands", "Waist", "Legs", "Feet",
    "Finger", "Trinket", "Ranged", "Main-Hand", "Off-Hand", "One-Hand", "Two-Hand"
]

def map_by_similarity(s: str, words=canonical_words, cutoff=0.7) -> str:
    """
    Map input string `s` to the closest word in `words` by similarity.
    `cutoff` is the minimum similarity (0-1) to consider.
    """
    # normalize input
    s_clean = s.replace("_", " ").replace("-", " ").title()
    
    matches = get_close_matches(s_clean, words, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    else:
        return s_clean  # fallback: return original cleaned string

def to_camel_case(s: str) -> str:
    words = re.split(r'\s+', s)
    return words[0].lower() + ''.join(w.capitalize() for w in words[1:])

def fix_glued_number(s: str) -> str:
    # insert a space between a letter and a number
    s = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', s)
    # insert a space between a number and a letter (optional)
    s = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', s)
    return s


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            'id': re.compile(r'(?:\b[Ii][Dd]\s*)?(\d{6,})(?![:\w])\b'),
            # Damage patterns
            'damage_range': re.compile(r'\b([0-9OGBQDS]{1,3})\s*[-–—]?\s*([0-9OGBQDS]{1,3})\s*Damage\b', re.IGNORECASE),
            'single_damage': re.compile(r'(\d+)\s+[Dd]amage'),
            'dps': re.compile(r'\(?([\d.]+)\s*damage\s*per\s*second\)?', re.IGNORECASE),
            
            # Speed and timing
            'speed': re.compile(r'[Ss]peed\s*([\d.]+)'),
            'cooldown': re.compile(r'(\d+)\s+sec\s+cooldown', re.IGNORECASE),
            
            # Level requirements
            'level_req': re.compile(r'[Rr]equires\s*[Ll]evel\s*(\d+)'),
            'item_level': re.compile(r'[Ii]tem\s+[Ll]evel\s+(\d+)'),
            
            # Primary stats (Strength, Agility, etc.)
            'base_stats': re.compile(r'[+]?(\d+)\s*([Ss]trength|[Aa]gility|[Ii]ntellect|[Ss]tamina|[Ss]pirit|[Aa]rmor|[Bb]lock)', re.IGNORECASE),
            
            # Secondary stats (ratings, resistances)
            'equip_stats': re.compile(r'^(?:[Ee]quip(?:e)?|[Cc]hance on hit|[Cc]hance on cast)[:\s]+\s*(?:[Ii]ncreases?|[Ii]mproves?|[Rr]estores)\s+([a-zA-Z ]+?)(?=\s+by|\s+for|\s+up|\s+per|\.|$).*', re.IGNORECASE),
            
            # Item types and slots
            'slot': re.compile(r'\b(Main\s*[Hh]and|Off\s*[Hh]and|Two-Hand|Head|Neck|Shoulder|Back|Chest|Wrist|Hands|Waist|Legs|Feet|Finger|Trinket|Ranged)\b', re.IGNORECASE),
            'slot_types': re.compile(r'\b([Cc]loth|[Ll]eather|[Mm]ail|[Pp]late|[Ss]word|[Aa]xe|[Mm]ace|[Dd]agger|[Ss]taff|[Bb]ow|[Gg]un|[Cc]rossbow|[Ww]and|[Ff]ist Weapon|[Pp]olearm|[Tt]hrown)|[Ss]hield\b', re.IGNORECASE),

            # Binding and rarity
            'binding': re.compile(r'\b(Binds when picked up|Binds when equipped|Binds to account|Soulbound|Account Bound)\b', re.IGNORECASE),
            'rarity': re.compile(r'\b(Poor|Common|Uncommon|Rare|Epic|Legendary|Artifact|Heirloom)\b', re.IGNORECASE),
            
            # Effects (Equip, Use, Chance on hit, etc.)
            'use_effect': re.compile(r'^[Uu]se:\s*(.+)$'),
            'chance_effect': re.compile(r'(?:[Cc]hance\s*on\s*hit|[Cc]hance\s*on\s*[spell\s*]*cast)\s*:?(.+)', re.IGNORECASE),

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
            'slot_types': {
                'sword', 'axe', 'mace', 'dagger', 'staff', 'bow', 'gun',
                'crossbow', 'wand', 'fist', 'weapon', 'polearm', 'thrown'
            },
            'common_words': {
                'damage', 'speed', 'level', 'requires', 'equip', 'use',
                'chance', 'increases', 'decreases', 'rating', 'resistance'
            }
        }
    
    # FUZZY REGEX
    def extract_dps(self, text):
        """
        Extract DPS from OCR text using fuzzy matching.
        Returns a float if found, else None.
        """
        # Extract numbers
        number_matches = re.findall(r'[\d.]+', text)
        if not number_matches:
            return None

        # Lowercase text
        lower_text = text.lower()

        # Define keywords for DPS
        keywords = ["damage", "per", "second"]

        # Count how many keywords appear fuzzily in the text (not tokenized)
        found = 0
        for kw in keywords:
            # Check fuzzy match anywhere in the text
            if fuzz.partial_ratio(kw, lower_text) > 80:
                found += 1

        if found >= 2:  # require at least 2 keywords matched
            return float(number_matches[0])  # assume the first number is DPS
        return None

    def extract_binding(self, text, threshold=81):
        """
        Returns the best-matching binding phrase from OCR text.
        """
        BINDINGS = [
            "Binds when picked up",
            "Binds on equip",
            "Binds to realm",
        ]
                
        text_lower = text.lower()

        keyword_score = fuzz.partial_ratio(text_lower, "soulbound")
        if keyword_score > threshold:
            return "Binds when picked up"

        # ✅ Step 1: fuzzy keyword check ("binds" could be "pinds" or "bincs")
        keyword_score = fuzz.partial_ratio(text_lower, "binds")
        if keyword_score < threshold:
            return None
        
        
        # Use RapidFuzz to find the closest match among BINDINGS
        result = process.extractOne(text_lower, BINDINGS, score_cutoff=threshold)
        if result is None:
            return None
        match, score, _ = result  # safe unpack now
        return match

    def extract_chance_effects(self, text, threshold=81):
        """
        Returns the best-matching on-chance effects from OCR text.
        """
        CHANCE_EFFECTS = ["chance on hit", "chance on cast", "chance on spell cast"]
                
        text_lower = text.lower()[:20]  # Only first 20 chars to extract the effect prefix and reduce fuzzy computation

        # Find the closest match among CHANCE_EFFECTS
        result = process.extractOne(text_lower, CHANCE_EFFECTS, score_cutoff=threshold)
        if result is None:
            return None
        
        match, score, _ = result
        return match

    def extract_item_id(self, text, min_length=6, threshold=80):
        """
        Attempts to extract an item ID from OCR text using fuzzy matching.
        Returns the numeric ID as int if found, else None.
        """
        # Candidate patterns that may indicate an ID
        ID_KEYWORDS = ["id", "itemid", "temid"]
        tokens = re.split(r'\s+', text.lower())

        for i, token in enumerate(tokens):
            # Check if token matches a keyword fuzzily
            if any(fuzz.partial_ratio(token, kw) > threshold for kw in ID_KEYWORDS):
                # Merge all following tokens to capture split numbers
                candidate = "".join(tokens[i+1:])  # all tokens after keyword
                digits = re.sub(r'\D', '', candidate)
                if len(digits) >= min_length:
                    return int(digits)

        # Fallback: any number in text with min_length
        digits_only = re.findall(r'\d{%d,}' % min_length, text)
        if digits_only:
            return int("".join(digits_only))

        return None
    
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
                prefix = "black_bkground" if self.config['preprocessing'].get('invert_colors', False) else "white_bkground"
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
            
            # ID parsing
            item_id = self.extract_item_id(text)
            if item_id is not None:
                item.id = int(item_id)
                continue

            # Damage parsing
            fix_damage_range_ocr = str.maketrans({
                'O': '0', 'D': '0', 'Q': '9',
                'G': '6', 'B': '8', 'S': '5'
            })
            damage_match = self.patterns['damage_range'].search(text)
            if damage_match:
                if item.damage is None:
                    item.damage = DamageInfo()
                min_dmg_raw = damage_match.group(1)
                max_dmg_raw = damage_match.group(2)

                item.damage.min = int(min_dmg_raw.translate(fix_damage_range_ocr))
                item.damage.max = int(max_dmg_raw.translate(fix_damage_range_ocr))
                continue

            # DPS parsing
            dps_value = self.extract_dps(text)
            if dps_value is not None:
                if item.damage is None:
                    item.damage = DamageInfo()
                item.damage.damagePerSecond = dps_value
                continue
            
            # Speed parsing
            speed_match = self.patterns['speed'].search(text)
            if speed_match:
                if item.damage is None:
                    item.damage = DamageInfo()
                item.damage.speed = float(speed_match.group(1))
                continue
            
            # Level requirements
            level_req_match = self.patterns['level_req'].search(text)
            if level_req_match:
                item.required_level = int(level_req_match.group(1))
                continue
            
            # Item Level
            item_level_match = self.patterns['item_level'].search(text)
            if item_level_match:
                item.item_level = int(item_level_match.group(1))
                continue
            
            # Primary stats
            for base_stats_match in self.patterns['base_stats'].finditer(text):
                value, stat = base_stats_match.groups()
                item.base_stats[stat.lower()] = int(value)
            
            # Equip stats
            for equip_stats_match in self.patterns['equip_stats'].finditer(text):
                equip_effect_string = equip_stats_match.group(1)
                key = to_camel_case(equip_effect_string)
                item.equip_stats[key] = fix_glued_number(equip_stats_match.string.replace("Equip:", "").replace("Equipe:", "").strip())
            
            # Slot
            slot_match = self.patterns['slot'].search(text)
            if slot_match:
                item.slot = map_by_similarity(slot_match.group(1))
                continue

            # Slot Types
            slot_types_match = self.patterns['slot_types'].search(text)
            if slot_types_match:
                item.slot_type = slot_types_match.group(0)

                # Ascension renders 'Thrown' as slot when slotype is 'Thrown'. Here we map it back to 'Ranged'.
                ranged_weapons = [
                    "Thrown", 'Wand' 'Ranged' 'Gun', 'Bow', 'Crossbow'
                ]

                if item.slot_type in ranged_weapons:
                    item.slot = "Ranged"
                continue

            # Binding
            binding_match = self.extract_binding(text)
            if binding_match:
                item.binding = binding_match
                continue
            
            use_match = self.patterns['use_effect'].search(text)
            if use_match:
                item.equip_stats["onUse"] = f"Use: {use_match.group(1)}"
            
            chance_match = self.patterns['chance_effect'].search(text)
            if chance_match:
                key = self.extract_chance_effects(chance_match.group(0))
                key = to_camel_case(key) if key else "onChance"
                item.equip_stats[key] = chance_match.group(0)
        
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

        # Store result in S3

        
        if save_debug:
            prefix = "black_bkground" if self.config['preprocessing'].get('invert_colors', False) else "white_bkground"
            debug_path = f"debug/{prefix}_{Path(image_path).stem}.json"
            with open(debug_path, 'w') as f:
                json.dump(ItemStats.to_dynamodb_item(result["item_stats"]), f, indent=2)
            logger.info(f"Debug info saved to {debug_path}")
        
        # Return item_stats
        return result
    
    def batch_process(self, image_paths: List[str], output_path: str = "processed_items.json") -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        results = []
        failed_count = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_item_image(image_path, True)
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
    
    # For batch processing (example)
    image_files = list(Path("images/").glob("*.png"))
    results = processor.batch_process([str(p) for p in image_files])