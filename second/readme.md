## 第一题

清洗数据时主要考虑如下方面：

```python
def clean_wiki_text(text):
    # 移除模板、链接、表格
    text = re.sub(r"\{\{.*?\}\}|\[\[.*?\]\]|\{\|.*?\|\}", "", text)
    # 移除引用标记
    text = re.sub(r'\[\d+\]', '', text)
    # 移除分类标记
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
    # 移除文件标记
    text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

除此之外，还考虑了中文繁体到简体的转换

遇到的问题主要在不熟悉啊下载的流程上，查阅资料后解决

## 第二题

清洗数据时主要考虑了将数据处理成纯文本格式（实际上没有太大必要）和处理为fasttext所需的数据格式上

fasttext要求一行为一条数据，以固定前缀+<标签名>开头

```python
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
```

效果评估时可根据设置的result_dic中对于超参数的不同设置来进行实验，通过精确率和召回率来判断效果

```python
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
```

运行时先后运行 download.py clean.py train.py即可