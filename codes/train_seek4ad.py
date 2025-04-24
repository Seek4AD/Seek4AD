from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from datasets import Dataset
import json
from qwen_vl_utils import process_vision_info
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

model_path="/home/datasets/zk/Qwen2.5-VL-7B-Instruct/"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
 

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
 

model.enable_input_require_grads()


data_path = "/home/Seek4AD/train.json"
with open(data_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-1]
    test_data = data[-1:]

with open("train_data.json", "w") as f:
    json.dump(train_data, f)
with open("test_data.json", "w") as f:
    json.dump(test_data, f)
    

train_ds = Dataset.from_json("train_data.json")

def process_func(example):

    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 448, "resized_width": 448},
                {"type": "text", "text": "请详细描述这张图片的缺陷。"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {key: value.tolist() for key, value in inputs.items()}
    

    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    }








train_dataset = train_ds.map(process_func)

print(f"Train dataset size: {len(train_dataset)}")
print(train_dataset[0])  # 检查数据格式

from peft import LoraConfig, get_peft_model
 
config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    
)
 
# 将 LoRA 应用于模型
peft_model = get_peft_model(model, config)

 
args = TrainingArguments(
    output_dir="/home/datasets/Qwen2.5-VL-LoRA1-5epoch7B",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=74,
    learning_rate=1e-3,
    gradient_checkpointing=True,

)
 
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,  # there is need for your data
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
 
trainer.train()