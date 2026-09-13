from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    # HTML 文本抽取：bytes -> Unicode -> 纯文本
    # （把 import 放在函数内，避免模块加载时引入重型依赖）
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    # 1) 把字节串解码成 Unicode 字符串。
    #    优先按 UTF-8 解码；若失败（说明页面不是 UTF-8），
    #    用 Resiliparse 的 detect_encoding() 探测真实编码后再解码。
    try:
        html_text = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)  # 返回编码名，例如 "cp1252"
        html_text = html_bytes.decode(encoding, errors="replace")

    # 2) HTML -> 纯文本（Resiliparse 会去掉标签、保留可见文字）
    return extract_plain_text(html_text)


def _find_lid_model() -> str:
    """按优先级在几个候选位置找 fastText 语言识别模型 lid.176.bin。"""
    candidates = [
        os.environ.get("FASTTEXT_LID_MODEL", ""),  # 显式指定
        "/shared-data/classifiers/lid.176.bin",  # 作业环境
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local-shared-data", "classifiers", "lid.176.bin"),  # 本地 dev
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "找不到 lid.176.bin。请设置环境变量 FASTTEXT_LID_MODEL 指向模型文件，"
        "或把模型放到 /shared-data/classifiers/lid.176.bin / local-shared-data/classifiers/lid.176.bin。"
    )


@lru_cache(maxsize=1)
def _load_lid_model(model_path: str):
    """懒加载 + 缓存模型，避免每次调用都重新读盘（126MB）。"""
    import fasttext

    return fasttext.load_model(model_path)


def run_identify_language(text: str) -> tuple[Any, float]:
    # 语言识别：fastText lid.176 -> (语言标识符, 置信度)
    # 说明：lid.176 预测结果是形如 ('__label__en',) 的标签，去掉 __label__ 前缀即得
    #       ISO 639-1/3 风格的代码（英语 "en"、简体中文 "zh"），无需额外重映射。
    model = _load_lid_model(_find_lid_model())
    # fastText 的 predict 一次只处理一行文本（遇到 \n 会抛异常），
    # 因此把换行折叠成空格再送入模型。
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    language = labels[0].replace("__label__", "")
    return language, float(scores[0])


# ---- PII 掩蔽用的正则（模块级编译一次） ----

# 电子邮件：本地部分 @ 域名，域名含至少一个点且 TLD 为 2+ 字母；
# 前后加负向断言防止匹配到更长标识符的中间片段。
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9._%+\-])"
)

# 美国电话号码：可选 +1 国家码 + 3-3-4 位数字，各部分可用 () / 空格 / - / . 分隔。
# 3-3-4 结构也覆盖纯 10 位连写（2831823829）。
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\d)")

# IPv4：四段 0-255。把每段写成 25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d 来保证数值范围合法。
_IPV4_RE = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
)


def _mask_with(pattern: re.Pattern[str], text: str, mask: str) -> tuple[str, int]:
    """通用掩蔽：把匹配到的子串替换成占位符，返回 (新字符串, 替换次数)。"""
    masked, n = pattern.subn(mask, text)
    return masked, n


def run_mask_emails(text: str) -> tuple[str, int]:
    # 把所有电子邮件替换为 |||EMAIL_ADDRESS|||
    return _mask_with(_EMAIL_RE, text, "|||EMAIL_ADDRESS|||")


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    # 把所有（美式格式的）电话号码替换为 |||PHONE_NUMBER|||
    return _mask_with(_PHONE_RE, text, "|||PHONE_NUMBER|||")


def run_mask_ips(text: str) -> tuple[str, int]:
    # 把所有 IPv4 地址替换为 |||IP_ADDRESS|||
    return _mask_with(_IPV4_RE, text, "|||IP_ADDRESS|||")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_model(env_var: str, filename: str) -> str:
    """在几个候选位置查找 fastText 模型文件。"""
    candidates = [
        os.environ.get(env_var, ""),
        os.path.join("/shared-data/classifiers", filename),
        os.path.join(_REPO_ROOT, "local-shared-data", "classifiers", filename),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"找不到 {filename}。请设置环境变量 {env_var}，或把模型放到 "
        f"/shared-data/classifiers/ 或 local-shared-data/classifiers/ 下。"
    )


@lru_cache(maxsize=4)
def _load_fasttext(model_path: str):
    import fasttext

    return fasttext.load_model(model_path)


def _predict_label(model_path: str, text: str) -> tuple[str, float]:
    """fastText 二分类：返回 (去掉 __label__ 前缀的标签, 置信度)。"""
    model = _load_fasttext(model_path)
    # fastText predict 只接受单行文本。
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    return labels[0].replace("__label__", ""), float(scores[0])


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    # Dolma / Jigsaw fastText NSFW 分类器，标签形如 nsfw / non-nsfw。
    label, score = _predict_label(
        _find_model("FASTTEXT_NSFW_MODEL", "dolma_fasttext_nsfw_jigsaw_model.bin"), text
    )
    # 统一标签命名：非 NSFW -> "non-nsfw"
    if "non" in label.lower():
        label = "non-nsfw"
    elif label.lower() in {"nsfw", "obscene", "toxic"} or "nsfw" in label.lower():
        label = "nsfw"
    return label, score


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    # Dolma / Jigsaw fastText 恶意言论分类器。
    label, score = _predict_label(
        _find_model(
            "FASTTEXT_TOXIC_MODEL", "dolma_fasttext_hatespeech_jigsaw_model.bin"
        ),
        text,
    )
    if "non" in label.lower():
        label = "non-toxic"
    else:
        label = "toxic"
    return label, score


def run_classify_quality(text: str) -> tuple[Any, float]:
    # 2.7 训练得到的质量分类器，标签为 __label__wiki / __label__cc。
    label, score = _predict_label(
        _find_model("CS336_QUALITY_MODEL", "quality_classifier.bin"), text
    )
    return label, score


# ---- Gopher 质量规则（题目 2.6） ----

_WORD_RE = re.compile(r"\S+")


def run_gopher_quality_filter(text: str) -> bool:
    """按 Gopher 论文的规则子集判断文本是否通过质量过滤器。

    规则：
      1. 词数必须落在 [50, 100000] 内；
      2. 平均词长必须在 [3, 10] 内；
      3. 以省略号 "..." 结尾的行占比必须 <= 30%；
      4. 含有至少一个字母字符的词占比必须 >= 80%。
    返回 True 表示通过（保留），False 表示应被移除。
    """
    words = _WORD_RE.findall(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100_000:
        return False

    # 平均词长（按去掉首尾标点后的 token 长度计算更贴近论文；这里用原始 token，
    # 与测试中的简单文本一致）。
    mean_word_length = sum(len(w) for w in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = [line for line in text.split("\n") if line.strip()]
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.30:
            return False

    words_with_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if words_with_alpha / num_words < 0.80:
        return False

    return True


# ---- 3.1 精确行去重 ----


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """精确行去重：只保留在整个语料库中只出现一次的行。

    第一遍用行内容的哈希统计词频（内存开销与行数而非总字节数成正比），
    第二遍只把出现次数为 1 的行写回输出目录。
    """
    import hashlib
    from collections import Counter

    output_directory = os.fspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    def _line_key(line: bytes) -> bytes:
        return hashlib.blake2b(line, digest_size=16).digest()

    line_counts: Counter[bytes] = Counter()
    contents: list[tuple[str, bytes]] = []
    for path in input_files:
        path = os.fspath(path)
        with open(path, "rb") as f:
            data = f.read()
        contents.append((os.path.basename(path), data))
        with open(path, "rb") as f:
            for line in f:
                line_counts[_line_key(line)] += 1

    for filename, data in contents:
        out_lines = [
            line
            for line in data.splitlines(keepends=True)
            if line_counts[_line_key(line)] == 1
        ]
        with open(os.path.join(output_directory, filename), "wb") as f:
            f.write(b"".join(out_lines))


# ---- 3.2 MinHash + LSH 模糊去重 ----


def _normalize_for_dedup(text: str) -> str:
    """按 RefinedWeb 做法归一化：NFD、去重音、小写、去标点、压缩空白。"""
    import re as _re
    import unicodedata

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _re.sub(r"[^\w\s]", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def _ngram_set(text: str, n: int) -> set[str]:
    tokens = _normalize_for_dedup(text).split()
    return {" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


_P = 2_147_483_647  # Mersenne prime 2**31 - 1，保证 int64 下乘法不溢出


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """用 MinHash + LSH 做模糊文档去重，并把保留的文档写到输出目录。"""
    import random

    import numpy as np

    assert num_hashes % num_bands == 0, "num_hashes 必须能被 num_bands 整除"
    rows = num_hashes // num_bands

    output_directory = os.fspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    filenames: list[str] = []
    texts: list[str] = []
    shingle_sets: list[set[str]] = []
    for path in input_files:
        path = os.fspath(path)
        with open(path, "r", errors="replace") as f:
            text = f.read()
        filenames.append(os.path.basename(path))
        texts.append(text)
        shingle_sets.append(_ngram_set(text, ngrams))

    rng = np.random.default_rng(0)
    # 用同一组随机系数对所有文档计算签名，保证不同文档的签名可比。
    a = rng.integers(1, _P, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, _P, size=num_hashes, dtype=np.int64)

    def signature(shingles: set[str]) -> "Any":
        import mmh3

        if not shingles:
            return np.full(num_hashes, _P - 1, dtype=np.int64)
        base = np.fromiter(
            (mmh3.hash(s, signed=False) % _P for s in shingles),
            dtype=np.int64,
            count=len(shingles),
        )
        sig = np.empty(num_hashes, dtype=np.int64)
        for i in range(num_hashes):
            sig[i] = int(((a[i] * base + b[i]) % _P).min())
        return sig

    signatures = [signature(s) for s in shingle_sets]

    # LSH：按 band 分桶，桶内任意两篇为候选重复对
    buckets: dict[tuple[int, ...], list[int]] = {}
    for doc_idx, sig in enumerate(signatures):
        for band in range(num_bands):
            key = tuple(int(x) for x in sig[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(doc_idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidate_pairs.add((members[i], members[j]))

    # 并查集聚类（只合并真实 Jaccard >= 阈值的候选对）
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    def jaccard(i: int, j: int) -> float:
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 and not s2:
            return 1.0
        inter = len(s1 & s2)
        union_size = len(s1) + len(s2) - inter
        return inter / union_size if union_size else 0.0

    for i, j in candidate_pairs:
        if jaccard(i, j) >= jaccard_threshold:
            union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(texts)):
        clusters.setdefault(find(idx), []).append(idx)

    keep: set[int] = set()
    rng_py = random.Random(336)
    for members in clusters.values():
        if len(members) == 1:
            keep.add(members[0])
        else:
            keep.add(rng_py.choice(sorted(members)))

    for idx in sorted(keep):
        with open(os.path.join(output_directory, filenames[idx]), "w") as f:
            f.write(texts[idx])
