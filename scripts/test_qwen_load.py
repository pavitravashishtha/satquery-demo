import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image, ImageDraw

MODEL_PATH = "/home/pavitra/satquery/models/qwen3-vl-2b"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading model (4-bit quantized)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model loaded. Creating a local test image...")
img = Image.new("RGB", (400, 400), color=(34, 139, 34))
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 300, 300], fill=(70, 130, 180))
img.save("/home/pavitra/satquery/scripts/test_image.png")

messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe the colors and shapes in this image."}]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True
).to(model.device)

print("Running inference...")
output = model.generate(**inputs, max_new_tokens=100)
print("---RESULT---")
print(processor.decode(output[0], skip_special_tokens=True))
