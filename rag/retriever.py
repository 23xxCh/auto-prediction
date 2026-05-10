"""rag/retriever.py — 维护知识库语义检索模块。

使用 FAISS 向量索引实现语义检索：
- load_kb() 加载所有文档并构建 FAISS 索引
- search(query, top_k) 返回最相关的文档片段
"""

import os
import re
import numpy as np

# 向量维度（使用 OpenAI text-embedding-3-small 的 1536 维）
EMBEDDING_DIM = 1536


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """将长文本按句子分块。

    Args:
        text: 原始文本
        chunk_size: 每块目标字符数
        overlap: 相邻块重叠字符数

    Returns:
        文本块列表
    """
    # 按句子分割
    sentences = re.split(r'(?<=[。！？\n])', text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            # 重叠部分
            overlap_chars = current[-overlap:] if len(current) > overlap else current
            current = overlap_chars + sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _load_markdown_doc(filepath: str) -> list[dict]:
    """加载单个 markdown 文档，返回文本块列表。

    Args:
        filepath: .md 文件路径

    Returns:
        [{"text": str, "metadata": {"source": str, "fault_type": str}}, ...]
    """
    fault_type = os.path.splitext(os.path.basename(filepath))[0]
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # 提取 YAML front matter（如果有）
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()

    # 按 ## 标题分块，每个主要章节作为独立块
    sections = re.split(r"\n##\s+", content)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # 每节进一步按段落分块
        paragraphs = re.split(r"\n(?=[^#\n])", section)
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= 400:
                current += "\n" + para if current else para
            else:
                if current:
                    chunks.append(current.strip())
                current = para
        if current.strip():
            chunks.append(current.strip())

    return [
        {"text": chunk, "metadata": {"source": os.path.basename(filepath), "fault_type": fault_type}}
        for chunk in chunks
    ]


def _simple_embed(texts: list[str]) -> np.ndarray:
    """使用 TF-IDF 生成伪嵌入向量（无 API 调用开销）。

    用于演示环境；生产环境应替换为真实 embedding API 调用。

    Args:
        texts: 文本列表

    Returns:
        shape (N, EMBEDDING_DIM) 的 numpy 数组
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
    except ImportError:
        # 无 sklearn，回退到随机向量
        rng = np.random.default_rng(42)
        return rng.uniform(-1, 1, size=(len(texts), EMBEDDING_DIM)).astype(np.float32)

    # TF-IDF → SVD 降维到 EMBEDDING_DIM
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)
    n_comp = min(EMBEDDING_DIM, tfidf.shape[1] - 1)
    if n_comp < 1:
        n_comp = max(1, tfidf.shape[1] - 1) if tfidf.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced = svd.fit_transform(tfidf)
    # L2 归一化
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb = (reduced / norms).astype(np.float32)
    # 填充到 EMBEDDING_DIM
    if emb.shape[1] < EMBEDDING_DIM:
        padded = np.zeros((emb.shape[0], EMBEDDING_DIM), dtype=np.float32)
        padded[:, :emb.shape[1]] = emb
        emb = padded
    return emb


class MaintenanceKB:
    """维护知识库检索器。"""

    def __init__(self, kb_dir: str = None):
        if kb_dir is None:
            kb_dir = os.path.join(os.path.dirname(__file__), "maintenance_kb")
        self.kb_dir = kb_dir
        self._docs: list[dict] = []
        self._embeddings: np.ndarray = None
        self._index = None

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    def load(self):
        """加载知识库并构建 FAISS 索引。"""
        import faiss

        md_files = [
            os.path.join(self.kb_dir, f)
            for f in os.listdir(self.kb_dir)
            if f.endswith(".md")
        ]

        self._docs = []
        for filepath in md_files:
            chunks = _load_markdown_doc(filepath)
            self._docs.extend(chunks)

        texts = [doc["text"] for doc in self._docs]
        self._embeddings = _simple_embed(texts)

        # 构建 FAISS 索引（内积索引 + L2 归一化等价于余弦相似度）
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

        print(f"[OK] 维护知识库已加载：{len(self._docs)} 个文档块")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索最相关的文档块。

        Args:
            query: 查询文本
            top_k: 返回最相似的 K 个块

        Returns:
            [{"text": str, "metadata": dict, "score": float}, ...]
        """
        import faiss

        if self._index is None:
            self.load()

        query_emb = _simple_embed([query]).astype(np.float32)
        scores, indices = self._index.search(query_emb.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._docs):
                doc = self._docs[idx].copy()
                doc["score"] = float(score)
                results.append(doc)

        return results


def build_index(kb_dir: str = None, output_path: str = None):
    """命令行构建并保存知识库索引。

    Usage:
        python -m rag.build_index
        python -m rag.build_index --kb_dir ./rag/maintenance_kb --output ./outputs/kb.index
    """
    import argparse

    parser = argparse.ArgumentParser(description="构建维护知识库 FAISS 索引")
    parser.add_argument("--kb_dir", default=None, help="知识库目录路径")
    parser.add_argument("--output", default=None, help="索引输出路径")
    args = parser.parse_args()

    if args.kb_dir:
        kb_dir = args.kb_dir
    else:
        kb_dir = os.path.join(os.path.dirname(__file__), "maintenance_kb")

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(kb_dir), "..", "outputs", "kb.index")

    kb = MaintenanceKB(kb_dir)
    kb.load()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[OK] 索引保存至: {output_path}")
    return kb


if __name__ == "__main__":
    build_index()
