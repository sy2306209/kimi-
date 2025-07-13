import os
import re
from datasets import load_from_disk, Dataset
from tqdm import tqdm


# HTML 标签清洗函数
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(
        r"[\U00010000-\U0010ffff\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        "", text)
    text = text.lower()
    # 去除换行符
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def add_label(text, name):
    if 'math' in name:
        text = '__label__math ' + text
    else:
        text = '__label__non-math ' + text
    return text


# 输入输出目录
data_dir = "./data"
output_dir = "./data_clean"
os.makedirs(output_dir, exist_ok=True)
all_train = []
all_test = []
# 遍历所有数据集目录
for dataset_name in os.listdir(data_dir):
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue
    print(f"正在处理: {dataset_name}")
    try:
        dataset = load_from_disk(dataset_path)


        # 清洗 text 字段
        def process(example):
            example["text"] = clean_text(example.get("text", ""))
            return example


        cleaned = dataset.map(process)
        # 保存
        save_path = os.path.join(output_dir, dataset_name)
        cleaned.save_to_disk(save_path)
        print(f"[成功] 已保存至 {save_path}")


        from random import shuffle

        labeled_texts = []
        for item in cleaned:
            if not item["text"]:
                continue
            labeled_text = add_label(item["text"], dataset_name)
            labeled_texts.append(labeled_text)
        shuffle(labeled_texts)
        split_idx = int(len(labeled_texts) * 0.75)
        train_texts = labeled_texts[:split_idx]
        test_texts = labeled_texts[split_idx:]
        txt_train_path = os.path.join(output_dir, f"{dataset_name}_train.txt")
        txt_test_path = os.path.join(output_dir, f"{dataset_name}_test.txt")
        with open(txt_train_path, "w", encoding="utf-8") as f:
            for line in train_texts:
                f.write(line + "\n")
        with open(txt_test_path, "w", encoding="utf-8") as f:
            for line in test_texts:
                f.write(line + "\n")
        all_train.extend(train_texts)
        all_test.extend(test_texts)
        print(f"[成功] 训练集TXT已保存至 {txt_train_path} 条数{len(train_texts)}")
        print(f"[成功] 测试集TXT已保存至 {txt_test_path} 条数{len(test_texts)}")
    except Exception as e:
        print(f"[失败] {dataset_name} 处理异常: {str(e)}")

shuffle(all_train)
shuffle(all_test)
all_train_path = os.path.join(output_dir, f"all_train.txt")
all_test_path = os.path.join(output_dir, f"all_test.txt")
with open(all_train_path, "w", encoding='utf-8') as f:
    for line in all_train:
        f.write(line + '\n')
with open(all_test_path, "w", encoding='utf-8') as f:
    for line in all_test:
        f.write(line + '\n')
