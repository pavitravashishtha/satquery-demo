import json
import os
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/home/pavitra/satquery/data/CDVQA"
IM1_DIR = "/home/pavitra/satquery/data/SECOND/im1"
IM2_DIR = "/home/pavitra/satquery/data/SECOND/im2"


def load_cdvqa_split(split="Train"):
    split = split.capitalize()
    with open(f"{DATA_DIR}/{split}_questions.json") as f:
        questions_data = json.load(f)["questions"]
    with open(f"{DATA_DIR}/{split}_images.json") as f:
        images_data = json.load(f)["images"]
    with open(f"{DATA_DIR}/{split}_answers.json") as f:
        answers_data = json.load(f)["answers"]

    img_by_id = {img["id"]: img for img in images_data}
    answer_by_id = {a["question_id"]: a for a in answers_data}

    samples = []
    skipped_annotation = 0
    skipped_missing_file = 0
    for q in questions_data:
        img = img_by_id.get(q["img_id"])
        answer = answer_by_id.get(q["id"])
        if img is None or answer is None:
            skipped_annotation += 1
            continue

        file_name = img["file_name"]
        im1_path = os.path.join(IM1_DIR, file_name)
        im2_path = os.path.join(IM2_DIR, file_name)
        if not (os.path.exists(im1_path) and os.path.exists(im2_path)):
            skipped_missing_file += 1
            continue

        samples.append({
            "question_id": q["id"],
            "question": q["question"],
            "question_type": q["type"],
            "answer": answer["answer"],
            "file_name": file_name,
            "im1_path": im1_path,
            "im2_path": im2_path,
        })

    print(f"[{split}] skipped {skipped_annotation} unmatched annotations, "
          f"{skipped_missing_file} missing image files")
    return samples


class CDVQADataset(Dataset):
    def __init__(self, split="Train"):
        self.samples = load_cdvqa_split(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        im1 = Image.open(s["im1_path"]).convert("RGB")
        im2 = Image.open(s["im2_path"]).convert("RGB")
        return {
            "im1": im1,
            "im2": im2,
            "question": s["question"],
            "answer": s["answer"],
            "question_type": s["question_type"],
        }


if __name__ == "__main__":
    ds = CDVQADataset("Train")
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    print(f"Sample question: {sample['question']}")
    print(f"Sample answer: {sample['answer']}")
    print(f"im1 size: {sample['im1'].size}, im2 size: {sample['im2'].size}")
