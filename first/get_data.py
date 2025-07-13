from wiki_dump_extractor import WikiXmlDumpExtractor
from opencc import OpenCC
import json
from tqdm import tqdm
import re
from bs4 import BeautifulSoup

# 初始化繁简转换器
cc = OpenCC('t2s')


def clean_wiki_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = re.sub(
        r"""
        \{\{.*?\}\}       |  # 模板 {{...}}
        \{\|.*?\|\}       |  # 表格 {|...|}
        <.*?>             |  # HTML标签
        \[\[(?:User|Talk|Category|File|Image):.*?\]\] |  # 用户/讨论/分类/文件
        \[\d+\]           |  # 引用标记 [1]
        &[a-z]+;|&#\d+;
        """,
        "",
        clean_text,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE
    )
    return clean_text


extractor = WikiXmlDumpExtractor(file_path="zhwiki-20250601-pages-articles-multistream.xml.bz2")
output_data = []

for i, page in tqdm(enumerate(extractor.iter_pages())):
    if i >= 1000:
        break

    meta = {
        "id": page.page_id,
        "title": cc.convert(page.title),
        "url": f"https://zh.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
        "timestamp": page.timestamp.isoformat()  # 页面最后编辑时间
    }

    # 清洗正文并转换简体
    cleaned_text = clean_wiki_text(page.text)
    text_content = cc.convert(cleaned_text.strip())

    output_data.append({
        "text": text_content,
        "meta": meta
    })

# 保存为 JSON 文件
with open("wiki_1000.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
