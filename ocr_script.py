from paddleocr import PaddleOCR
import cv2
from matplotlib import pyplot as plt
import simplejson as json
import os

# Initialize PaddleOCR with updated parameter name
ocr = PaddleOCR(
    use_textline_orientation=True,  # Updated from use_angle_cls
    lang="en"
)

image_path = "image.png"
img = cv2.imread(image_path)

# Convert BGR to RGB for proper matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.axis('off')  # Optional: hide axes
plt.title('Input Image')
# plt.show()

# Perform OCR
result = ocr.predict(image_path)

# Save the full JSON output
json.dump(result[0].json, open("paddle_ocr_output.json", "w"), indent=4)

# Extract and display the recognized text
print("OCR Results:")
print("=" * 50)

# Access the recognized texts from the JSON structure
json_data = result[0].json
rec_texts = json_data['res']['rec_texts']

print(f"Total text lines detected: {len(rec_texts)}")
print("\nExtracted Text:")
print("-" * 30)

for idx, text in enumerate(rec_texts):
    print(f"{idx + 1:2d}. {text}")

# Optional: Create a cleaned version of the text
print("\n" + "=" * 50)
print("CLEANED TEXT OUTPUT:")
print("=" * 50)

def clean_text(text_list):
    """Clean and format the extracted text"""
    cleaned = []
    for text in text_list:
        # Basic cleaning
        text = text.strip()
        if text:  # Only add non-empty lines
            # Fix common OCR errors for game items
            text = text.replace("rainted", "Tainted")  # Common OCR error
            text = text.replace("11o", "110")  # Number recognition error
            text = text.replace("38.9 damage per second)", "(38.9 damage per second)")  # Fix parentheses
            cleaned.append(text)
    return cleaned

cleaned_texts = clean_text(rec_texts)
for text in cleaned_texts:
    print(text)

# Optional: Structure the data for game items
print("\n" + "=" * 50)
print("STRUCTURED ITEM DATA:")
print("=" * 50)

def parse_item_data(text_list):
    """Parse game item data into structured format"""
    item_data = {
        'name': '',
        'type': '',
        'damage': '',
        'speed': '',
        'dps': '',
        'stats': [],
        'requirements': [],
        'effects': [],
        'other': []
    }
    
    for text in text_list:
        text = text.strip()
        if not text:
            continue
            
        # Item name (usually one of the first longer strings without numbers)
        if not item_data['name'] and len(text) > 5 and not any(char.isdigit() for char in text) and 'Hand' not in text and 'Binds' not in text:
            item_data['name'] = text
        
        # Weapon type
        elif text in ['Sword', 'Axe', 'Mace', 'Dagger', 'Staff', 'Bow', 'Gun']:
            item_data['type'] = text
        
        # Hand type
        elif 'Hand' in text:
            item_data['type'] += f" ({text})" if item_data['type'] else text
        
        # Damage
        elif 'Damage' in text:
            item_data['damage'] = text
        
        # Speed
        elif text.startswith('Speed'):
            item_data['speed'] = text
        
        # DPS
        elif 'damage per second' in text:
            item_data['dps'] = text
        
        # Stats (Strength, Stamina, etc.)
        elif any(stat in text for stat in ['Strength', 'Stamina', 'Agility', 'Intellect', 'Spirit']):
            item_data['stats'].append(text)
        
        # Requirements
        elif text.startswith('Requires') or text.startswith('Item Level'):
            item_data['requirements'].append(text)
        
        # Effects
        elif text.startswith('Equip:') or 'chance' in text.lower() or 'damage to' in text.lower():
            item_data['effects'].append(text)
        
        # Other
        else:
            item_data['other'].append(text)
    
    return item_data

structured_data = parse_item_data(cleaned_texts)

for key, value in structured_data.items():
    if value:  # Only show non-empty fields
        if isinstance(value, list):
            if value:  # Only show non-empty lists
                print(f"{key.upper()}:")
                for item in value:
                    print(f"  - {item}")
        else:
            print(f"{key.upper()}: {value}")

# Save structured data as well
with open("structured_item_data.json", "w") as f:
    json.dump(structured_data, f, indent=4)

print(f"\nFiles saved:")
print(f"- paddle_ocr_output.json (raw OCR output)")
print(f"- structured_item_data.json (parsed item data)")