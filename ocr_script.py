from paddleocr import PaddleOCR
import cv2
from matplotlib import pyplot as plt
import simplejson as json

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

json.dump(result[0].json, open("paddle_ocr_output.json", "w"), indent=4)

# Print results
print("OCR Results:")
for idx, res in enumerate(result):
    if res:  # Check if results exist
        print(f"\nPage {idx + 1}:")
        for line in res:
            bbox = line[0]  # Bounding box coordinates
            text = line[1][0]  # Extracted text
            # confidence = line[1][1]  # Confidence score
            print(f"Text: '{text}'")
            # print(f"Text: '{text}' | Confidence: {confidence:.4f}")
    else:
        print(f"No text found on page {idx + 1}")