import json
import fasttext
# 训练模型（此处保留您的原始代码）
model = fasttext.train_supervised(
    input="./data_clean/all_train.txt",
    epoch=3,
    lr=0.1,
    wordNgrams=2,
    verbose=2,
    minCount=1,
    loss='softmax'
)
model.save_model("./data_clean/fasttext_model.bin")

# 在测试集上评估
result = model.test("./data_clean/all_test.txt")
test_samples = result[0]
precision = round(result[1], 4)
recall = round(result[2], 4)

# 构建结果字典
results_dict = {
    "model_path": "./data_clean/fasttext_model.bin",
    "test_samples": test_samples,
    "precision": precision,
    "recall": recall,
    "hyperparameters": {  # 记录超参数便于复现
        "epoch": 10,
        "lr": 0.1,
        "wordNgrams": 2,
        "minCount": 1,
        "loss": "softmax"
    }
}
print(precision)
print(recall)
test_results = []

with open("./data_clean/all_test.txt", "r", encoding="utf-8") as test_file:
    for line in test_file:
        text = line.strip().split(' ', 1)[1]  # 移除标签前缀
        true_label = line.strip().split(' ')[0]

        # 模型预测
        pred_labels, pred_probs = model.predict(text, k=1)  # 获取Top3预测
        test_results.append({
            "text": text,
            "true_label": true_label,
            "pred_labels": [label.replace('__label__', '') for label in pred_labels],
            "pred_probs": [round(prob, 4) for prob in pred_probs]
        })

# 保存为JSON文件
with open("./data_clean/test_results.json", "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)  # 禁用ASCII编码确保中文支持
with open("./data_clean/details.json", "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=4, ensure_ascii=False)
