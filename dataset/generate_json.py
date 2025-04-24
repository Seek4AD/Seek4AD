import os
import json
from collections import defaultdict

def generate_json_structure(dataset_root, output_path):
    """
    生成JSON结构并保存到指定路径
    
    参数说明：
    - dataset_root: 数据集根目录路径（需包含 train/test 子目录）
    - output_path: 生成的JSON文件保存路径（包含文件名）
    """
    result = {"train": defaultdict(list), "test": defaultdict(list)}
    
    for split in ["train", "test"]:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            continue
            
        for cls_name in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
                
            for specie_name in os.listdir(cls_path):
                specie_path = os.path.join(cls_path, specie_name)
                if not os.path.isdir(specie_path):
                    continue
                    
                anomaly = 1 if specie_name == "defect" else 0
                
                for img_file in sorted(os.listdir(specie_path)):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    record = {
                        "img_path": f"{cls_name}/{split}/{specie_name}/{img_file}",
                        "mask_path": "",
                        "cls_name": cls_name,
                        "specie_name": specie_name,
                        "anomaly": anomaly
                    }
                    result[split][cls_name].append(record)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "train": dict(result["train"]),
            "test": dict(result["test"])
        }, f, indent=2)

# 使用示例
if __name__ == "__main__":
    # 用户自定义路径
    dataset_path = "/path/to/your/dataset"  # 数据集根目录
    output_json_path = "/custom/path/your_dataset_index.json"  # 输出文件路径
    
    generate_json_structure(dataset_path, output_json_path)