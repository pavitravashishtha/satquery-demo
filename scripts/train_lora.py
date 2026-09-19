import sys
import os
sys.path.insert(0, "/home/pavitra/satquery/scripts")
sys.path.insert(0, "/home/pavitra/satquery")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from cdvqa_dataset import CDVQADataset

# 1. Load Processor and Model
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"  # or your local model directory
processor = AutoProcessor.from_pretrained(model_id)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

# Critical for QLoRA: properly prepares quantized model for gradient checkpointing
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 2. Setup LoRA Config
peft_config = LoraConfig(
    r=8,           # Reduced from 16 to halve adapter memory on 6GB GPU
    lora_alpha=16, # Keep alpha == r for stable training
    # Attention projections only — FFN adapters exceed 6GB VRAM on RTX 4050 Laptop
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 3. Load Dataset
train_dataset = CDVQADataset(split="Train")

# 4. Correct Collate Function
def collate_fn(batch):
    texts = []
    images = []

    for sample in batch:
        q = sample.get("question") or sample.get("query") or ""
        a = sample.get("answer") or ""

        im1 = sample.get("im1") or sample.get("image1")
        im2 = sample.get("im2") or sample.get("image2")

        if isinstance(im1, str):
            im1 = Image.open(im1).convert("RGB")
        if isinstance(im2, str):
            im2 = Image.open(im2).convert("RGB")

        # Resize to 128x128 to keep sequences short and fit backward pass in 6GB VRAM
        im1 = im1.resize((128, 128), Image.BILINEAR)
        im2 = im2.resize((128, 128), Image.BILINEAR)

        images.append([im1, im2])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": q},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": a}],
            }
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(prompt)

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# 5. Define Training Arguments
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/pavitra/satquery/checkpoints/qwen2.5-vl-cdvqa-lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    # Speed optimizations
    optim="adamw_torch_fused",        # fused kernel is faster on CUDA
    dataloader_num_workers=0,          # single-process avoids worker crashes in Python 3.14
    dataloader_pin_memory=False,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn
)

if __name__ == "__main__":
    print(f"Starting full training run on {len(train_dataset)} samples...")
    print(f"Checkpoints and final LoRA adapter will be saved to: {OUTPUT_DIR}")
    trainer.train()
    print(f"Saving final LoRA adapter and processor to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Full training completed and adapter saved successfully!")
