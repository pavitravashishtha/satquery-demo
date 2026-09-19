from collections import Counter
import json
from PIL import Image
from peft import PeftModel
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Adjust these paths if your actual base model or images reside elsewhere
BASE_MODEL = "/home/pavitra/satquery/models/qwen2.5-vl-3b"
ADAPTER_PATH = "/home/pavitra/satquery/checkpoints/qwen2.5-vl-cdvqa-lora"
DATA_DIR = "/home/pavitra/satquery/data/CDVQA"
IMAGE_DIR = "/home/pavitra/satquery/data/SECOND"


def load_val_samples():
  with open(f"{DATA_DIR}/Val_questions.json") as f:
    questions = json.load(f)["questions"]
  with open(f"{DATA_DIR}/Val_images.json") as f:
    images = json.load(f)["images"]
  with open(f"{DATA_DIR}/Val_answers.json") as f:
    answers = json.load(f)["answers"]

  img_by_id = {i["id"]: i for i in images}
  ans_by_qid = {a["question_id"]: a for a in answers}

  samples = []
  for q in questions:
    img = img_by_id.get(q["img_id"])
    ans = ans_by_qid.get(q["id"])
    if img and ans:
      samples.append({
          "question": q["question"],
          "answer": ans["answer"],
          "file_name": img["file_name"],
      })
  return samples


print("Loading base model + adapter...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(BASE_MODEL)
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

samples = load_val_samples()
print(f"Evaluating on {len(samples)} validation samples...")

correct = 0
total = 0
answer_counts = Counter(s["answer"] for s in samples)
majority_answer = answer_counts.most_common(1)[0][0]
majority_correct = 0

# Evaluates the first 200 samples for a fast validation check
for s in samples[:200]:
  img1_path = f"{IMAGE_DIR}/im1/{s['file_name']}"
  img2_path = f"{IMAGE_DIR}/im2/{s['file_name']}"

  try:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
  except FileNotFoundError:
    continue

  messages = [{
      "role": "user",
      "content": [
          {"type": "image", "image": img1},
          {"type": "image", "image": img2},
          {"type": "text", "text": s["question"]},
      ],
  }]

  inputs = processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
  ).to(model.device)

  with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)

  prediction = (
      processor.decode(output[0], skip_special_tokens=True).strip().lower()
  )
  total += 1

  if s["answer"].lower() in prediction:
    correct += 1
  if majority_answer.lower() == s["answer"].lower():
    majority_correct += 1

print(f"\nModel accuracy: {correct}/{total} = {correct/total:.1%}")
print(
    f"Majority-baseline accuracy (always guessing '{majority_answer}'):"
    f" {majority_correct}/{total} = {majority_correct/total:.1%}"
)

