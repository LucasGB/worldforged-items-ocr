import re
from rapidfuzz import fuzz, process
from typing import Optional, List, Dict, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class EquipStat:
    """Represents an extracted equipment statistic"""
    stat_type: str
    value: int
    raw_text: str
    confidence: float

class FuzzyEquipStatsExtractor:
    def __init__(self, similarity_threshold: int = 75):
        """
        Initialize the extractor with similarity threshold for fuzzy matching
        
        Args:
            similarity_threshold: Minimum similarity score (0-100) for pattern matching.
                                Higher values = stricter matching, lower values = more lenient.
                                75 is a good balance for OCR text with some errors.
        """
        self.similarity_threshold = similarity_threshold
        
        # Define stat types and their variations/synonyms
        self.stat_patterns = {
            'spellPower': [
                'spell power', 'spellpower', 'spell damage', 'spelldamage'
            ],
            'attackPower': [
                'attack power', 'attackpower'
            ],
            'criticalStrikeRating': [
                'critical strike rating', 'criticalstrike', 'critical'
            ],
            'defenseRating': [
                'defense rating', 'defence rating', 'defense', 'defence'
            ],
            'manaPerSec': [
                'mana per sec', 'mana per second',
                'restores mana', 'mana regeneration'
            ],
            'hasteRating': [
                'haste rating', 'haste', 'hastereating', 'hasterating'
            ],
            'hitRating': [
                'hit rating', 'hitrating', 'hitraeting',
            ],
            'parryRating': [
                'parry rating', 'parry chance', 'parry',
                'parrying', 'block rating'
            ],
            'stealthLevel': [
                'stealth level', 'stealth', 'effective stealth level',
                'stealth rating'
            ],
            'spellPenetration': [
                'spell penetration', 'spell pen', 'magic penetration',
                'spell piercing'
            ],
            'expertiseRating': [
                'expertise rating', 'expertise'
            ],
            'pvePower': [
                'pve Power'
            ],
            'pvpPower': [
                'pvp Power'
            ]
        }
        
        # Common OCR character corrections
        self.ocr_corrections = str.maketrans({
            'l': '1',
            'I': '1',
            'o': '0',
            'O': '0',
            'S': '5',
            'G': '6',
            'B': '8',
            'Z': '2',
            'g': '9'
        })
        
        # Regex patterns for different stat formats
        self.stat_regex_patterns = [
            r'equip:\s*increases?\s+([^b]+?)(?:\s+by\s+|\s+)(\d+)',
            r'equip:\s*improves?\s+([^b]+?)(?:\s+by\s+|\s+)(\d+)',
            r'equip:\s*restores?\s+(\d+)\s+(.+?)\s+per\s+sec',
            r'equip:\s*(.+?)\s+(\d+)',
            r'increases?\s+([^b]+?)(?:\s+by\s+|\s+)(\d+)',
            r'improves?\s+([^b]+?)(?:\s+by\s+|\s+)(\d+)',
            r'restores?\s+(\d+)\s+(.+?)\s+per\s+sec',
            r'(\+\d+)\s+(.+)',
            r'(.+?)\s+(\+?\d+)'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize the input text"""
        if not text:
            return ""
        
        # Remove extra whitespace, convert to lowercase, remove punctuation
        cleaned = re.sub(r'[^\w\s+]', ' ', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        return cleaned
    
    def apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR character corrections"""
        return text.translate(self.ocr_corrections)
    
    def extract_numbers(self, text: str) -> List[int]:
        """Extract all numeric values from text"""
        # Handle both regular numbers and numbers with + prefix
        numbers = re.findall(r'[+]?(\d+)', text)
        return [int(num) for num in numbers if num.isdigit()]
    
    # def find_best_stat_match(self, text_fragment: str) -> Tuple[Optional[str], float]:
    #     """
    #     Find the best matching stat type for a text fragment
        
    #     Returns:
    #         Tuple of (stat_type, confidence_score) or (None, 0.0)
    #     """
    #     best_match = None
    #     best_score = 0.0
        
    #     # Create a flat list of all patterns with their stat types
    #     all_patterns = []
    #     for stat_type, patterns in self.stat_patterns.items():
    #         for pattern in patterns:
    #             all_patterns.append((pattern, stat_type))
        
    #     # Find best match using rapidfuzz
    #     pattern_texts = [pattern for pattern, _ in all_patterns]
    #     result = process.extractOne(
    #         text_fragment,
    #         pattern_texts,
    #         scorer=fuzz.partial_ratio,  # Use partial ratio for better substring matching
    #         score_cutoff=self.similarity_threshold
    #     )
        
    #     if result:
    #         matched_pattern, score = result[0], result[1]
    #         # Find the stat type for this pattern
    #         stat_type = next((st for p, st in all_patterns if p == matched_pattern), None)
    #         return stat_type, score / 100.0
        
    #     return None, 0.0
    def find_best_stat_match(self, text_fragment: str) -> Tuple[Optional[str], float]:
        """Find the best matching stat type for a text fragment"""
        best_match = None
        best_score = 0.0
        
        # Normalize the input text
        normalized_text = text_fragment.lower().strip()
        
        for stat_type, patterns in self.stat_patterns.items():
            for pattern in patterns:
                # Try exact substring match first
                if pattern.lower() in normalized_text:
                    return stat_type, 1.0
                
                # Then try fuzzy matching with a higher threshold
                score = fuzz.ratio(pattern.lower(), normalized_text) / 100.0
                if score > best_score and score >= self.similarity_threshold:
                    best_match = stat_type
                    best_score = score
        
        return best_match, best_score
    
    def parse_with_regex(self, text: str) -> List[EquipStat]:
        """Parse stats using regex patterns"""
        stats = []
        
        for pattern in self.stat_regex_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 2:
                    # Handle different group arrangements
                    if 'restores' in pattern and 'per sec' in pattern:
                        # Special case for mana per sec
                        value_str, stat_desc = groups[0], groups[1]
                        if 'mana' in stat_desc:
                            stat_type = 'mana_per_sec'
                        else:
                            continue
                    else:
                        # Normal case
                        if groups[0].isdigit():
                            # Value comes first (like "+10 attack power")
                            value_str, stat_desc = groups[0], groups[1]
                        else:
                            # Stat description comes first
                            stat_desc, value_str = groups[0], groups[1]
                    
                    # Extract numeric value
                    value_match = re.search(r'(\d+)', value_str)
                    if not value_match:
                        continue
                    
                    value = int(value_match.group(1))
                    
                    # Find best matching stat type
                    stat_type, confidence = self.find_best_stat_match(stat_desc.strip())
                    
                    if stat_type and confidence > 0.5:
                        stats.append(EquipStat(
                            stat_type=stat_type,
                            value=value,
                            raw_text=match.group(0),
                            confidence=confidence
                        ))
        
        return stats
    
    def extract_stats_fuzzy(self, text: str) -> List[EquipStat]:
        """Extract stats using fuzzy matching when regex fails"""
        stats = []
        
        # Split text into meaningful chunks
        words = text.split()
        numbers = self.extract_numbers(text)
        
        if not numbers:
            return stats
        
        # Try to match stat patterns in sliding windows
        for window_size in range(2, min(8, len(words) + 1)):
            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i + window_size])
                
                # Skip windows without numbers
                window_numbers = self.extract_numbers(window)
                if not window_numbers:
                    continue
                
                # Remove numbers to get stat description
                stat_desc = re.sub(r'\+?\d+', '', window).strip()
                
                # Find best matching stat type
                stat_type, confidence = self.find_best_stat_match(stat_desc)
                
                if stat_type and confidence > 0.6:
                    value = window_numbers[0]  # Take first number in window
                    
                    stats.append(EquipStat(
                        stat_type=stat_type,
                        value=value,
                        raw_text=window,
                        confidence=confidence
                    ))
        
        return stats
    
    def extract_equipment_stats(self, text: str) -> List[EquipStat]:
        """
        Main method to extract equipment stats from OCR text
        
        Args:
            text: Raw OCR text containing equipment stats
            
        Returns:
            List of EquipStat objects with extracted statistics
        """
        if not text:
            return []
        
        all_stats = []
        
        # Step 1: Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Step 2: Try regex parsing first
        regex_stats = self.parse_with_regex(cleaned_text)
        all_stats.extend(regex_stats)
        
        # Step 3: If no stats found, apply OCR corrections and try again
        if not all_stats:
            corrected_text = self.apply_ocr_corrections(cleaned_text)
            corrected_stats = self.parse_with_regex(corrected_text)
            all_stats.extend(corrected_stats)
        
        # Step 4: Try fuzzy matching as fallback
        if not all_stats:
            fuzzy_stats = self.extract_stats_fuzzy(cleaned_text)
            all_stats.extend(fuzzy_stats)
        
        # Remove duplicates and sort by confidence
        unique_stats = {}
        for stat in all_stats:
            key = (stat.stat_type, stat.value)
            if key not in unique_stats or stat.confidence > unique_stats[key].confidence:
                unique_stats[key] = stat
        
        result = list(unique_stats.values())
        result.sort(key=lambda x: x.confidence, reverse=True)
        
        return result
    
    def extract_with_confidence(self, text: str) -> Tuple[List[EquipStat], float]:
        """
        Extract equipment stats with overall confidence score
        
        Args:
            text: Raw OCR text containing equipment stats
            
        Returns:
            Tuple of (list_of_stats, overall_confidence_score)
            overall_confidence is the average confidence of all extracted stats
        """
        if not text:
            return [], 0.0
        
        stats = self.extract_equipment_stats(text)
        
        if not stats:
            return [], 0.0
        
        # Calculate overall confidence as weighted average
        total_confidence = sum(stat.confidence for stat in stats)
        overall_confidence = total_confidence / len(stats)
        
        return stats, overall_confidence
    
    def extract_single_stat_with_confidence(self, text: str) -> Tuple[Optional[EquipStat], float]:
        """
        Extract the best single stat with confidence score
        
        Args:
            text: Raw OCR text containing equipment stats
            
        Returns:
            Tuple of (best_stat, confidence_score) or (None, 0.0)
        """
        if not text:
            return None, 0.0
        
        stats = self.extract_equipment_stats(text)
        
        if not stats:
            return None, 0.0
        
        # Return the stat with highest confidence
        best_stat = max(stats, key=lambda s: s.confidence)
        return best_stat, best_stat.confidence
    
    def batch_extract(self, texts: List[str]) -> List[List[EquipStat]]:
        """Extract stats from multiple texts"""
        return [self.extract_equipment_stats(text) for text in texts]
    
    def extract_to_dict(self, text: str) -> Dict[str, int]:
        """
        Extract stats and return as a simple dictionary
        
        Returns:
            Dictionary mapping stat types to values
        """
        stats = self.extract_equipment_stats(text)
        return {stat.stat_type: stat.value for stat in stats}


# Example usage and testing
def main():
    """Test the fuzzy equipment stats extractor"""
    extractor = FuzzyEquipStatsExtractor(similarity_threshold=75)
    
    # Test cases including the provided examples and OCR variations
    test_cases = [
        "Equip: Increases spell power by 9.",
        "Equip: Increases attack power by 10.",
        "Equip: Increases critical strike rating by 79.",
        "Equip: Increases defense rating by 89.",
        "Equip: Restores 5 mana per sec.",
        "Equip: Improves hit rating by 5",
        "Equip: Increases your parry rating by 89.",
        "Equip: Increases your effective stealth level by 1.",
        "Equip: Increases spell penetration by 9.",
        "Equip: Increases your expertise rating by 4.",
        # OCR error variations
        "Equlp: lncreases spell power by l2.",  # 'i' -> 'l', '2' looks like 'l2'
        "Equip: lncreases attack power by lO.",  # 'I' -> 'l', '0' -> 'O'
        "Equip: Increases crit rating by B9.",   # '8' -> 'B'
        "Equlp: Restores S mana per sec.",      # '5' -> 'S'
        "+15 Spell Power",                       # Alternative format
        "Attack Power +20",                      # Alternative format
    ]
    
    print("RapidFuzz Equipment Stats Extraction Results:")
    print("=" * 80)
    
    for text in test_cases:
        print(f"\nInput: '{text}'")
        stats = extractor.extract_equipment_stats(text)
        
        if stats:
            for stat in stats:
                print(f"  → {stat.stat_type}: {stat.value} (confidence: {stat.confidence:.2f})")
                print(f"    Raw match: '{stat.raw_text}'")
        else:
            print("  → No stats found")
    
    # Test confidence methods
    print(f"\n" + "="*50)
    print("Confidence Methods Examples:")
    
    # Multiple stats with confidence
    multi_stat_text = "Equip: Increases spell power by 15 and critical strike rating by 30."
    stats_list, overall_conf = extractor.extract_with_confidence(multi_stat_text)
    print(f"\nMultiple stats text: '{multi_stat_text}'")
    print(f"Overall confidence: {overall_conf:.2f}")
    for stat in stats_list:
        print(f"  → {stat.stat_type}: {stat.value} (conf: {stat.confidence:.2f})")
    
    # Single best stat with confidence
    single_stat_text = "Equlp: lncreases spell power by l5."  # OCR errors
    best_stat, conf = extractor.extract_single_stat_with_confidence(single_stat_text)
    print(f"\nSingle stat (with OCR errors): '{single_stat_text}'")
    if best_stat:
        print(f"Best stat: {best_stat.stat_type} = {best_stat.value} (confidence: {conf:.2f})")
    else:
        print("No stats found")
    
    # Dictionary output example
    print(f"\n" + "-"*40)
    print("Dictionary Output Example:")
    example_text = "Equip: Increases critical strike rating by 79."
    result_dict = extractor.extract_to_dict(example_text)
    print(f"Input: '{example_text}'")
    print(f"Output: {result_dict}")


if __name__ == "__main__":
    main()