import re
from rapidfuzz import fuzz, process
from typing import Optional, List, Tuple, Dict

class FuzzyItemLevelExtractor:
    def __init__(self, similarity_threshold: int = 70):
        """
        Initialize the extractor with similarity threshold for fuzzy matching
        
        Args:
            similarity_threshold: Minimum similarity score (0-100) for pattern matching
        """
        self.similarity_threshold = similarity_threshold
        
        # Target patterns to match against
        self.target_patterns = [
            "item level",
            "itemlevel", 
            "item lvl",
            "itemlvl",
            "level",
            "lvl"
        ]
        
        # Common OCR character substitutions
        self.ocr_corrections = str.maketrans({
            'l': '1',  # Common OCR mistake
            'I': '1',  # Common OCR mistake
            'o': '0',  # Common OCR mistake
            'O': '0',  # Common OCR mistake
            'S': '5',  # Sometimes S looks like 5
            'G': '6',  # Sometimes G looks like 6
            'B': '8',  # Sometimes B looks like 8
        })
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize the input text"""
        if not text:
            return ""
        
        # Remove extra whitespace and convert to lowercase
        cleaned = re.sub(r'\s+', ' ', text.lower().strip())
        return cleaned
    
    def extract_numbers(self, text: str) -> List[int]:
        """Extract all numeric values from text"""
        numbers = re.findall(r'\d+', text)
        return [int(num) for num in numbers if num.isdigit()]
    
    def find_best_pattern_match(self, text: str) -> Optional[Tuple[str, int, float]]:
        """
        Find the best matching pattern in the text using rapidfuzz
        
        Returns:
            Tuple of (matched_pattern, position, similarity_score) or None
        """
        if not text:
            return None
        
        # Split text into overlapping windows to catch patterns
        words = text.split()
        candidates = []
        
        # Create candidate strings of different lengths
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):  # Up to 3-word combinations
                candidate = ' '.join(words[i:j])
                candidates.append((candidate, i))
        
        # Also add the full text as a candidate
        candidates.append((text, 0))
        
        best_match = None
        best_score = 0
        
        for candidate, pos in candidates:
            # Find best match against all target patterns
            result = process.extractOne(
                candidate, 
                self.target_patterns,
                scorer=fuzz.ratio,
                score_cutoff=self.similarity_threshold
            )
            
            if result and result[1] > best_score:
                best_match = (result[0], pos, result[1])
                best_score = result[1]
        
        return best_match
    
    def extract_level_near_pattern(self, text: str, pattern_info: Tuple[str, int, float]) -> Optional[int]:
        """
        Extract item level number near the identified pattern
        
        Args:
            text: Original text
            pattern_info: Tuple of (pattern, position, score)
            
        Returns:
            Extracted level or None
        """
        pattern, position, score = pattern_info
        words = text.split()
        
        # Look for numbers in nearby words
        search_range = 3  # Look 3 words before and after
        start_idx = max(0, position - search_range)
        end_idx = min(len(words), position + search_range + 1)
        
        nearby_text = ' '.join(words[start_idx:end_idx])
        numbers = self.extract_numbers(nearby_text)
        
        # Filter for reasonable item level values
        reasonable_levels = [num for num in numbers if 1 <= num <= 200]
        
        if reasonable_levels:
            return reasonable_levels[0]  # Return first reasonable number
        
        return None
    
    def apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR character corrections"""
        return text.translate(self.ocr_corrections)
    
    def extract_with_regex_patterns(self, text: str) -> Optional[int]:
        """Try to extract using common regex patterns"""
        patterns = [
            r'item\s*level\s*(\d+)',
            r'itemlevel\s*(\d+)',
            r'item\s*lvl\s*(\d+)',
            r'itemlvl\s*(\d+)',
            r'level\s*(\d+)',
            r'lvl\s*(\d+)',
            r'(\d+)\s*level',
            r'(\d+)\s*lvl'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                level = int(match.group(1))
                if 1 <= level <= 200:
                    return level
        
        return None
    
    def extract_item_level(self, text: str) -> Optional[int]:
        """
        Main method to extract item level from OCR text
        
        Args:
            text: Raw OCR text that may contain errors
            
        Returns:
            Extracted item level as integer, or None if not found
        """
        if not text:
            return None
        
        # Step 1: Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Step 2: Try direct regex extraction first (fastest)
        direct_result = self.extract_with_regex_patterns(cleaned_text)
        if direct_result:
            return direct_result
        
        # Step 3: Apply OCR corrections and try regex again
        corrected_text = self.apply_ocr_corrections(cleaned_text)
        corrected_result = self.extract_with_regex_patterns(corrected_text)
        if corrected_result:
            return corrected_result
        
        # Step 4: Use fuzzy matching to find pattern
        pattern_match = self.find_best_pattern_match(cleaned_text)
        if pattern_match:
            level = self.extract_level_near_pattern(cleaned_text, pattern_match)
            if level:
                return level
        
        # Step 5: Try fuzzy matching on corrected text
        corrected_pattern_match = self.find_best_pattern_match(corrected_text)
        if corrected_pattern_match:
            level = self.extract_level_near_pattern(corrected_text, corrected_pattern_match)
            if level:
                return level
        
        # Step 6: Last resort - look for any reasonable numbers
        all_numbers = self.extract_numbers(cleaned_text)
        reasonable_numbers = [num for num in all_numbers if 1 <= num <= 200]
        
        if reasonable_numbers:
            # Prefer numbers in typical item level ranges (10-100)
            preferred = [num for num in reasonable_numbers if 10 <= num <= 100]
            return preferred[0] if preferred else reasonable_numbers[0]
        
        return None
    
    def batch_extract(self, texts: List[str]) -> List[Optional[int]]:
        """Extract item levels from multiple texts"""
        return [self.extract_item_level(text) for text in texts]
    
    def extract_with_confidence(self, text: str) -> Tuple[Optional[int], float]:
        """
        Extract item level with confidence score
        
        Returns:
            Tuple of (item_level, confidence_score)
        """
        if not text:
            return None, 0.0
        
        cleaned_text = self.preprocess_text(text)
        
        # Try direct regex first
        direct_result = self.extract_with_regex_patterns(cleaned_text)
        if direct_result:
            return direct_result, 1.0  # High confidence for direct matches
        
        # Try with OCR corrections
        corrected_text = self.apply_ocr_corrections(cleaned_text)
        corrected_result = self.extract_with_regex_patterns(corrected_text)
        if corrected_result:
            return corrected_result, 0.9  # Slightly lower confidence
        
        # Use fuzzy matching
        pattern_match = self.find_best_pattern_match(cleaned_text)
        if pattern_match:
            level = self.extract_level_near_pattern(cleaned_text, pattern_match)
            if level:
                confidence = pattern_match[2] / 100.0  # Convert to 0-1 scale
                return level, confidence
        
        return None, 0.0


# Example usage and testing
def main():
    """Test the fuzzy item level extractor"""
    extractor = FuzzyItemLevelExtractor(similarity_threshold=70)
    
    # Test cases including the provided examples and various OCR errors
    test_cases = [
        "ItemLevel43",           # Missing space
        "Item Level 29",         # Clean format
        "tem level 56",          # Missing 'I'
        "item lvl 82",           # Abbreviated
        "Level: 45",             # With punctuation
        "itemlevel67",           # No space
        "ITEM LEVEL 91",         # All caps
        "1tem Level 38",         # '1' instead of 'I'
        "Item Leve1 74",         # '1' instead of 'l'
        "ltem Level 55",         # 'l' instead of 'I'
        "Item Level5o",          # 'o' instead of '0'
        "ltam Lavel 63",         # Multiple OCR errors
        "Damage: 125 Item Level 67",  # Multiple numbers
        "Level 23 Defense 45",   # Multiple attributes
        "iten level B8",         # 'B' instead of '8'
        "garbage text",          # No pattern
        "",                      # Empty
        "Level 999",             # Out of range
    ]
    
    print("RapidFuzz Item Level Extraction Results:")
    print("=" * 60)
    print(f"{'Input Text':<25} {'Extracted Level':<15} {'Confidence':<10}")
    print("-" * 60)
    
    for text in test_cases:
        level, confidence = extractor.extract_with_confidence(text)
        level_str = str(level) if level else "None"
        confidence_str = f"{confidence:.2f}" if confidence > 0 else "0.00"
        print(f"{text:<25} {level_str:<15} {confidence_str:<10}")
    
    # Batch extraction example
    print(f"\nBatch extraction for provided examples:")
    provided_examples = ["ItemLevel43", "Item Level 29", "tem level 56"]
    results = extractor.batch_extract(provided_examples)
    
    for text, result in zip(provided_examples, results):
        print(f"'{text}' -> {result}")


if __name__ == "__main__":
    main()