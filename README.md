## 第一题

清洗数据时主要考虑如下方面：

```python
def clean_wiki_text(text):
  # 使用BeautifulSoup解析HTML，提取纯文本（自动忽略<script>、<style>等标签内容）
    soup = BeautifulSoup(text, 'html.parser')  
    clean_text = soup.get_text()  # 获取去标签后的纯文本
    
    # 合并连续空白字符（包括换行符、制表符等）为单个空格，并去除首尾空白
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  
    
    # 综合正则表达式：移除维基百科特有噪声
    clean_text = re.sub(
        r"""
        \{\{.*?\}\}       |  # 模板标记（如{{Infobox}}）
        \{\|.*?\|\}       |  # 表格标记（如{| class="wikitable" |}）
        <.*?>             |  # 残留HTML标签（如<div>），BeautifulSoup未完全清除时补充处理
        \[\[(?:User|Talk|Category|File|Image):.*?\]\] |  # 非正文链接（用户页/讨论页/分类/文件）
        \[\d+\]           |  # 引用标记
        &[a-z]+;|&#\d+;   |  # HTML实体（如&amp;、&#32;）
        """,
        "",  
        clean_text,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE  
        # re.IGNORECASE: 忽略大小写（如{{infobox}}和{{Infobox}}等效）
        # re.DOTALL: 允许.匹配换行符（跨行模板）
        # re.VERBOSE: 允许正则换行和注释（提高可读性）[5,9](@ref)
    )
    
    return clean_text  
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
