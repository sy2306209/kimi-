from datasets import load_dataset, Dataset
import os
from tqdm import tqdm

# 添加镜像配置（在代码最前面）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


dataset_names = ["open-web-math/open-web-math", "HuggingFaceFW/fineweb"]
print(os.environ['HF_ENDPOINT'])
# for name in tqdm(dataset_names):
#     try:
#         # 1. 流式加载1000条数据
#         dataset = load_dataset(name, split="train", streaming=True)
#         # subset = dataset.take(1000)
#         # local_data = Dataset.from_list(list(subset))
#         # # 保存到本地
#         # safe_name = name.replace("/", "__")
#         # save_path = f"./data/{safe_name}_sampled"
#         # os.makedirs(save_path, exist_ok=True)
#         # local_data.save_to_disk(save_path)
#         # print(f"[成功] 数据集 {name} 已保存至 {save_path}")
#     except Exception as e:
#         print(f"[失败] {name} 处理异常: {str(e)}")
for name in dataset_names:
    try:
        # 优先加载数据集配置（不下载完整数据）
        dataset = load_dataset(name, streaming=True, trust_remote_code=True)

        # 分批次处理10k样本
        batch_size = 1000
        all_samples = []
        for idx, sample in enumerate(dataset["train"]):
            if idx >= 10000: break
            all_samples.append(sample)
            if len(all_samples) % batch_size == 0:
                print(f"已处理 {len(all_samples)} 条数据")

        # 保存子集
        local_data = Dataset.from_list(all_samples)
        safe_name = name.replace("/", "__")
        local_data.save_to_disk(f"./data/{safe_name}_sampled")

    except Exception as e:
        print(f"[失败] {name}: {str(e)}")