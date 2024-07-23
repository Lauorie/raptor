import json
import numpy as np
from embeddings import EmbeddingsClient
from daily import Raptor
raptor = Raptor()

path_ = '/root/web_demo/HybirdSearch/es_app_0702/santi2.txt'
with open(path_, 'r', ) as f:
    data = f.read()

data = data[:5120]
# 将文本数据转换为列表，每段长度为 512
def split_text(text, max_len=512):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

docs_texts = split_text(data)
print(docs_texts[:2])

# Build tree
leaf_texts = docs_texts
results = raptor.recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
from loguru import logger
logger.info(f'results:\n{results}')

# from langchain_community.vectorstores import Chroma

# Initialize all_texts with leaf_texts
all_texts = leaf_texts.copy()

# Iterate through the results to extract summaries from each level and add them to all_texts
for level in sorted(results.keys()):
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)

logger.info(f'all_texts:\n{all_texts}')

# Now, use all_texts to build the vectorstore with Chroma
# vectorstore = Chroma.from_texts(texts=all_texts, embedding=raptor.embd)
# retriever = vectorstore.as_retriever()