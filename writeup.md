# 2 字节对编码（BPE）分词器
## 2.1 Unicode 基础
### 习题 unicode1（1 分）
(a) chr(0)会返回哪个字符？
'\x00'
(b) 该字符的字符串表示__repr__()与直接打印的显示效果有何区别？
```python
>>> print(repr(chr(0)))
'\x00'
>>> print(chr(0))

```
(c) 文本中出现该字符会产生什么现象？建议在 Python 交互器运行下方代码观察效果：
```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0)
'this is a test\x00'
>>> print("this is a test" + chr(0))
this is a test
```
## 2.2 Unicode 编码
### 习题 unicode2（3 分）
(a) 相比 UTF-16、UTF-32，训练分词器优先选用 UTF-8 的原因是什么？可对比不同文本的编码输出长度。

英文文本 UTF-8 占用字节远少于另外两种，通用语料整体存储开销更低；且 UTF-8 是通用标准，无字节序、代理对问题，分词处理更简单稳定。

(b) 下方解码函数存在缺陷，请说明错误原因，并给出一段会输出错误结果的字节输入：
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
测试示例：decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 输出正常hello

错误字节样例
```python
>>> decode_utf8_bytes_to_str_wrong("b'\xe4\xb8\xad'".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
```
原理

该函数逐字节单独解码，多字节 UTF-8 字符拆分后单字节非法，抛出 UnicodeDecodeError，无法正确解析多字节汉字、emoji 等。

(c) 写出一段合法双字节序列，无法解码为任何 Unicode 字符

错误字节样例
```python
>>> decode_utf8_bytes_to_str_wrong("b'\xc0\x80''".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
```
解释

该两字节属于 UTF-8 非法超长编码，Unicode 规范禁止使用，无法解码出合法字符。
## 2.4 BPE 分词器训练流程
### 习题 train_bpe（15 分）
实现 BPE 训练函数，入参：
input_path：训练文本文件路径
vocab_size：词表最大总容量（含基础字节、合并子词、特殊 Token）
special_tokens：特殊 Token 字符串列表；训练时作为文档硬分隔，不参与频次统计
返回输出：
vocab: dict[int, bytes]：ID → 字节串映射
merges: list[tuple[bytes, bytes]]：合并规则列表，按训练生成顺序存储
实现adapters.run_train_bpe适配接口，执行uv run pytest tests/test_train_bpe.py通过全部单元测试。
拓展：可使用 C++/Rust 加速核心合并逻辑（cppyy/PyO3 绑定），但 GPT-2 正则在多数引擎性能较差，Python regex 库速度最优。
### 习题 train_bpe_tinystories（2 分）
(a) 在 TinyStories 数据集训练 BPE，词表上限 10000，添加<|endoftext|>特殊 Token；将词表、合并规则序列化保存。训练耗时、内存占用为多少？词表内最长子词是什么，是否符合直觉？
资源限制：无 GPU，时长≤30 分钟，内存≤30GB
提示：
<|endoftext|>是文档分隔标记
特殊 Token 单独预处理拆分，不参与字节配对统计
```
uv run python scripts/train_bpe_tinystories.py \
    --output-dir artifacts/tinystories_bpe
```
```
{
  "input": "data/TinyStoriesV2-GPT4-train.txt",
  "vocab_size": 10000,
  "merge_count": 9743,
  "elapsed_seconds": 1027.8581523187459,
  "peak_rss_gib": 10.606597900390625,
  "longest_token_bytes": "b' accomplishment'",
  "longest_token_utf8": " accomplishment",
  "longest_token_byte_length": 15
}
```
训练耗时：1,027.86 秒，约 17 分 8 秒, 峰值常驻内存（Peak RSS）：约 10.61 GiB

词表中最长子词为：" accomplishment"

该单词长度为 15 字节，其中包含开头的空格。这个结果符合直觉：英文 BPE 通常会把“空格 + 高频完整单词”合并成一个 Token；accomplishment 本身较长，并且在儿童故事中有一定出现频率，因此能够经过多轮合并形成完整子词。

(b) 对训练代码性能做性能剖析，训练流程中耗时最高的环节是什么？
```
uv run python scripts/train_bpe_tinystories.py \
    --input tests/fixtures/tinystories_sample_5M.txt \
    --vocab-size 1000 \
    --output-dir artifacts/tinystories_bpe_profile_sample \
    --profile
```
```
uv run python -m pstats artifacts/tinystories_bpe_profile_sample/train.prof
Welcome to the profile statistics browser.
artifacts/tinystories_bpe_profile_sample/train.prof% sort cumulative
artifacts/tinystories_bpe_profile_sample/train.prof% stats 20
Sun Aug  9 20:57:08 2026    artifacts/tinystories_bpe_profile_sample/train.prof

         13852255 function calls (13852236 primitive calls) in 5.891 seconds

   Ordered by: cumulative time
   List reduced from 108 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.386    0.386    5.891    5.891 /home/lanyuqi/assignment1-basics/cs336_basics/bpe.py:40(train_bpe)
        1    2.636    2.636    4.262    4.262 /home/lanyuqi/assignment1-basics/cs336_basics/bpe.py:18(_pretoken_counts)
  6422070    1.143    0.000    1.143    0.000 /home/lanyuqi/assignment1-basics/cs336_basics/bpe.py:36(<genexpr>)
      743    0.538    0.001    0.955    0.001 {built-in method builtins.max}
  3668036    0.417    0.000    0.417    0.000 /home/lanyuqi/assignment1-basics/cs336_basics/bpe.py:72(<lambda>)
  1263131    0.251    0.000    0.251    0.000 {method 'group' of '_regex.Match' objects}
  1263132    0.222    0.000    0.222    0.000 {method 'encode' of 'str' objects}
    64626    0.050    0.000    0.161    0.000 /usr/lib/python3.12/collections/__init__.py:595(__init__)
    64626    0.027    0.000    0.111    0.000 /usr/lib/python3.12/collections/__init__.py:669(update)
    64859    0.019    0.000    0.047    0.000 {built-in method builtins.isinstance}
    64624    0.038    0.000    0.038    0.000 {built-in method _collections._count_elements}
   312034    0.030    0.000    0.030    0.000 {built-in method builtins.len}
   159645    0.028    0.000    0.028    0.000 {method 'add' of 'set' objects}
    64624    0.016    0.000    0.028    0.000 <frozen abc>:117(__instancecheck__)
   140856    0.024    0.000    0.024    0.000 {method 'discard' of 'set' objects}
   141940    0.021    0.000    0.021    0.000 {method 'append' of 'list' objects}
    64624    0.012    0.000    0.012    0.000 {built-in method _abc._abc_instancecheck}
    64626    0.012    0.000    0.012    0.000 {method 'items' of 'dict' objects}
        1    0.003    0.003    0.009    0.009 {method 'read' of '_io.TextIOWrapper' objects}
        1    0.000    0.000    0.007    0.007 /home/lanyuqi/assignment1-basics/.venv/lib/python3.12/site-packages/regex/_main.py:324(split)
```
性能剖析显示，预切词与词频统计函数 _pretoken_counts 耗时最高，在样本总训练时间 5.89 秒中累计占用约 4.26 秒（约 72%）；主要开销来自正则预切词，以及将匹配结果转换为字节序列并累计频次。

### 习题 train_bpe_expts_owt（2 分）
(a) OpenWebText 数据集训练 BPE，词表上限 32000，序列化输出词表与合并规则。词表最长子词是什么，是否合理？
资源限制：无 GPU，时长≤12 小时，内存≤100GB
```
uv run python scripts/train_bpe_owt.py     --output-dir artifacts/owt_bpe
```
```
{
  "input": "data/owt_train.txt",
  "vocab_size": 32000,
  "merge_count": 31744,
  "elapsed_seconds": 24822.225309392437,
  "peak_rss_gib": 99.15481185913086,
  "longest_token": {
    "bytes": "b'\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82\\xc3\\x83\\xc3\\x82'",
    "utf8": "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ",
    "byte_length": 64
  },
}
```
训练耗时：24,822.23 秒，约 6 小时 53 分 42 秒，峰值常驻内存：约 99.15 GiB

词表中最长子词为 64 字节：ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ

该结果从自然语言语义上看并不合理。Ã、Â 的重复通常是 UTF-8 文本被使用错误字符编码解码、再重新编码后形成的乱码，即 mojibake。OpenWebText 来源复杂，可能包含网页抓取产生的编码错误和重复噪声；当这种字节序列在语料中频繁出现时，BPE 并不了解其语义，仍会不断将它合并，最终形成很长的 Token。因此，这一结果对未经清洗的网页语料而言是可以解释的，也说明 BPE 正确学习了语料中的高频字节模式；但它不是有意义的自然语言子词。若追求更高质量的词表，应在训练前进行编码修复、乱码过滤及重复文本清理。

(b) 对比 TinyStories 与 OpenWebText 训练得到的两套分词器差异
```
uv run python scripts/compare_bpe_vocabularies.py
```
```
TinyStories vocab size: 10000
OpenWebText vocab size: 32000
Shared token count: 7311
TinyStories tokens also in OWT: 73.1%
Mean token byte length (TinyStories): 5.79
Mean token byte length (OWT): 6.33
Longest TinyStories-only tokens: [' granddaughter', ' congratulated', '<|endoftext|>', ' veterinarian', ' strawberries', ' marshmallows', ' imaginations', ' caterpillars', ' wildflowers', ' thermometer', ' superheroes', ' storekeeper', ' stethoscope', ' screwdriver', ' sandcastles', ' reluctantly', ' motorcycles', ' mischievous', ' marshmallow', ' invitations', ' heartbroken', ' hairdresser', ' grasshopper', ' grandparent', ' firefighter', ' disagreeing', ' decorations', ' cauliflower', ' caterpillar', ' butterflies']
Longest OWT-only tokens: ['ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', '----------------------------------------------------------------', '————————————————', 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', '________________________________', '================================', '................................', '--------------------------------', '********************************', '————————', ' telecommunications', ' disproportionately', ' environmentalists', ' unconstitutional', ' responsibilities', ' misunderstanding', ' disproportionate', ' cryptocurrencies', ' counterterrorism', ' characterization', 'ÃÂÃÂÃÂÃÂ', '________________', '================', '................', '----------------', '****************', '################', ' vulnerabilities', ' straightforward', ' representatives']
```
两套词表公有 7,311 个 Token，覆盖 TinyStories 词表的 73.1%；TinyStories 更偏向儿童故事中的具体词汇，而 OpenWebText 的平均子词更长（6.33 vs. 5.79 字节），包含更多正式、技术性词汇。OWT 词表还学到了网页分隔线和编码乱码等噪声，体现了开放网页语料更复杂但清洁度较低。
## 2.6 BPE 分词器：编码与解码
### 习题 tokenizer（15 分）
实现 Tokenizer 类，支持特殊 Token、编码、解码、流式大文件编码，推荐接口：
```python
def __init__(self, vocab, merges, special_tokens=None)
# 类方法，从磁盘序列化文件加载分词器
@classmethod
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
# 完整文本编码
def encode(self, text: str) -> list[int]
# 迭代器流式编码大文件，惰性输出ID，控制内存占用
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
# ID序列转回文本
def decode(self, ids: list[int]) -> str
``` 
实现adapters.get_tokenizer适配接口，执行uv run pytest tests/test_tokenizer.py全部通过。
### 习题 tokenizer_experiments（4 分）
(a) 分别从 TinyStories、OpenWebText 抽取 10 篇文档，使用对应分词器编码，计算各自压缩比（字节 / 单个 token）

```
uv run python scripts/tokenizer_experiments.py a
```
```
TinyStories: documents=10, bytes=7,552, tokens=1,817, compression_ratio=4.16 bytes/token
OpenWebText: documents=10, bytes=31,604, tokens=6,720, compression_ratio=4.70 bytes/token
```

TinyStories 对应分词器压缩比为 4.16 字节/Token，OpenWebText 对应分词器为 4.70 字节/Token；OWT 的更大词表和更丰富子词使其压缩率略高。

(b) 使用 TinyStories 训练的 10k 词分词器编码 OpenWebText 样本，压缩比、分词效果会出现什么变化？定性描述差异

```
uv run python scripts/tokenizer_experiments.py b
```
```
OWT tokenizer:       bytes=31,604, tokens=6,720, compression_ratio=4.70 bytes/token
TinyStories tokenizer: bytes=31,604, tokens=9,882, compression_ratio=3.20 bytes/token
Change: tokens +47.1%, compression ratio -32.0%
Qualitative result: the TinyStories vocabulary is less suited to web text, so technical terms, proper nouns, code, and other rare strings are split into more and shorter byte-level subwords.
```

TinyStories 分词器编码 OWT 时压缩比从 4.70 降至 3.20 字节/Token，同一样本 Token 数由 6,731 增至 9,882；技术词和长词会被切成更多、更短的子词。

(c) 估算分词器吞吐（字节 / 秒），计算完整 825GB Pile 数据集编码总耗时

The Pile（825GB）: EleutherAI 开源的英文预训练大语料，总原始文本大小 825GiB，由 22 个子数据集混合而成，用来预训练 GPT‑Neo、GPT‑NeoX、Pythia 等开源大模型。

当前纯 Python 实现吞吐约 0.83–0.93 MB/s，按此估算编码 825 GB Pile 约需 10.3–11.5 天（单进程 CPU，实际时间受硬件和文本分布影响）。

(d) 使用两套分词器分别编码对应训练 / 验证集，存储为 uint16 格式 NumPy 数组，说明选用 uint16 的原因

已提供流式转换脚本 scripts/encode_datasets.py，运行：

```
uv run python scripts/encode_datasets.py \
    --output-dir artifacts/tokenized
```
```
tinystories_train: 540796778 tokens -> artifacts/tokenized/tinystories_train.uint16.bin
tinystories_valid: 5461210 tokens -> artifacts/tokenized/tinystories_valid.uint16.bin
owt_train: 2730253427 tokens -> artifacts/tokenized/owt_train.uint16.bin
owt_valid: 66477875 tokens -> artifacts/tokenized/owt_valid.uint16.bin
```

TinyStories 和 OWT 词表大小分别为 10,000 和 32,000，

最大 Token ID分别为 9,999 和 31,999，

均处于 `uint16` 的取值范围 0--65,535 内，因此不会发生溢出。

每个 Token 使用 2 字节，相比 `uint32` 的 4 字节可节省一半磁盘空间和内存带宽；

同时 `uint8` 最大只能表示 255，无法容纳这两套词表。

由于文件是不带 `.npy` 头的原始二进制数组，可用 `np.memmap(path, dtype=np.uint16, mode="r")` 直接读取而不必一次载入内存。

# 3 Transformer语言模型架构
## 3.3 基础构建模块：线性层与嵌入层
### 习题 linear（1 分）
要求自行继承torch.nn.Module实现无偏置线性层，禁止调用nn.Linear、nn.functional.linear，仅允许用nn.Parameter存储权重W（权重不提前转置）。

推荐类接口：
```python
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```
实现要求：

继承 nn.Module，调用父类构造函数；

权重存入 nn.Parameter，变量名为 W，不做转置；

使用指定截断正态初始化；

完成适配器adapters.run_linear，执行uv run pytest -k test_linear通过单元测试。

实现符合上述规范的 Linear 类。

### 习题 embedding（1 分）
手动实现词嵌入层，禁止使用 nn.Embedding；

权重矩阵尺寸(vocab_size, d_model)，输入 Long 型 Token ID 张量查表输出对应向量。

推荐类接口：
```python
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        pass
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass
```
实现要求：

继承 nn.Module，调用父类构造；

嵌入矩阵封装为 nn.Parameter；

矩阵最后一维为 d_model；

禁止使用官方嵌入 API；

按截断正态规则初始化权重。

实现adapters.run_embedding适配器，运行对应单元测试uv run pytest -k test_embedding。

完成嵌入类实现，通过全部测试用例。
## 3.4 Pre-Norm Transformer 块
### 习题 rmsnorm（1 分）
原始 Transformer 使用 LayerNorm，本作业跟随 LLaMA 系列采用 RMSNorm，公式：

$$RMSNorm(a_i) = \frac{a_i}{RMS(a)} g_i,\quad RMS(a)=\sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}a_i^2+\varepsilon}$$

g是可学习增益向量，$\varepsilon=10^{-5}$ 用于数值稳定。

实现强制要求：输入先转为 float32 防止平方运算溢出，计算完成还原原始精度。前向伪代码：
```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 计算RMSNorm逻辑
result = ...
return result.to(in_dtype)
```
推荐接口：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        pass
    def forward(self, x):
        pass
```
适配器：adapters.run_rmsnorm

实现 RMSNorm 模块并通过测试 uv run pytest -k test_rmsnorm。
### 习题 positionwise_feedforward（2 分）
SwiGLU 完整前馈公式：

$$FFN(x)=W_2\big(SiLU(W_1 x)\odot W_3 x\big)$$

规定$d_{ff}=\frac{8}{3}d_{model}$，向上取最近 64 的整数倍，适配 GPU 张量核心加速。

代码允许直接调用 torch.sigmoid 保证数值稳定，适配器adapters.run_swiglu。

实现 SwiGLU 前馈网络，通过单元测试 uv run pytest -k test_swiglu。
### 习题 rope（2 分）
不对嵌入整体叠加固定位置向量，而是对 Q、K 向量按二维分组执行旋转变换。

位置i、第k组二维旋转矩阵：

$$R_k^i=\begin{bmatrix}\cos\theta_{i,k} & -\sin\theta_{i,k}\\ \sin\theta_{i,k} & \cos\theta_{i,k}\end{bmatrix},\quad \theta_{i,k}=\frac{i}{\Theta^{(2k-2)/d_k}}$$

完整旋转矩阵为分块对角矩阵，无需显式构建完整大矩阵，优化计算速度。

RoPE 无可训练参数，所有层可复用同一组预计算 cos/sin 张量，使用 register_buffer 注册（不存入模型权重文件）；

仅对 Query、Key 执行旋转，Value 不做变换。

推荐类接口：
```python
class RotaryPositionalEmbedding:
    def __init__(self, theta, d_k, max_seq_len, device=None):
        pass
    def forward(self, x, token_positions):
        pass
```
输入 x 支持任意前置批量维度，token_positions 张量维度匹配序列长度，用于索引预存三角函数缓存。

适配器adapters.run_rope

实现 RoPE 模块，通过全部测试用例 uv run pytest -k test_rope。
### 习题 softmax（1 分）
先实现数值稳定版 Softmax，公式：
$$softmax(v_i)=\frac{\exp(v_i)}{\sum_j\exp(v_j)}$$
优化技巧：输入每行减去该行最大值，防止 exp 溢出得到 NaN。

实现带数值稳定优化的 Softmax 函数。uv run pytest -k test_softmax_matches_pytorch
### 习题 scaled_dot_product_attention（5 分）
缩放点积注意力完整公式：
$$Attention(Q,K,V)=softmax\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
掩码规则：布尔矩阵M尺寸$n\times m$，True代表第i个 Query 可以读取第j个 Key，False代表遮蔽。

实现方式：掩码 False 位置填充负无穷再送入 Softmax。

函数支持任意前置批量维度，适配器adapters.run_scaled_dot_product_attention

实现缩放点积注意力，支持自定义布尔掩码，兼容 3 维、4 维输入张量。 uv run pytest -k test_scaled_dot_product_attention
### 习题 multihead_self_attention（5 分）
多头数学定义：
$$MultiHead(Q,K,V)=Concat(head_1,...,head_h)$$
$$head_i=Attention(Q_i,K_i,V_i)$$
$Q_i$,$K_i$,$V_i$是 $QKV$ 沿维度均分后的子张量。多头拼接后经过输出投影$W_O$。
$$MultiHeadSelfAttention(x)=W_O MultiHead(W_Q x,W_K x,W_V x)$$
可拓展优化：将 $W_Q/W_K/W_V$ 合并为单个权重矩阵，只做一次矩阵乘法。

因果掩码：上三角全部遮蔽，模型无法看到未来 Token；

Q、K 前执行 RoPE 旋转，Value 不旋转。

模块入参：d_model、num_heads，自动满足$d_k=d_v=d_{model}/num\_heads$。

适配器adapters.run_multihead_self_attention

实现带因果掩码、RoPE 的多头自注意力模块。uv run pytest -k test_multihead_self_attention
### 习题 transformer_block（3 分）
按照 Pre-Norm 公式拼接 RMSNorm、多头注意力、SwiGLU 前馈，输入输出固定为(batch, seq, d_model)。

类入参：d_model、num_heads、d_ff。

适配器adapters.run_transformer_block

完整实现 Pre-Norm Transformer 块。  uv run pytest -k test_transformer_block
## 3.5 完整 Transformer 语言模型
### 习题 transformer_lm（3 分）
组装流程：Token 嵌入层 → 堆叠 num_layers 个 Transformer 块 → 最终 RMSNorm 归一化 → 线性 LM 头输出词表 Logits。

模型初始化参数：vocab_size、d_model、num_heads、d_ff、num_layers、context_length（用于 RoPE 最大序列长度）。

适配器adapters.run_transformer_lm

搭建完整解码器 Transformer 模型，通过全部单元测试。 uv run pytest -k test_transformer_lm

### 习题 transformer_accounting（5 分）

模型算力统计矩阵乘法算力计算公式：矩阵$A(m\times n) × B(n\times p)$，总浮点运算量2mnp。

(a) 考虑一个采用本作业架构、规模对标 GPT-2-XL 的模型，配置如下：

词表大小：50257

上下文长度：1024

层数：48

模型维度 d_model：1600

注意力头数：25

前馈网络维度 d_ff：4288（取的是 $\frac{8}{3} \times 1600$ 最接近 64 的整数倍）

假设我们按照这套配置搭建模型，该模型有多少可训练参数？假设每个参数用单精度浮点数存储，仅加载该模型需要占用多大内存？

解答

记 $V=50257,T=1024,L=48,d=1600,d_{\mathrm{ff}}=4288$。本作业架构使用 RoPE，因此没有可训练的位置 embedding；FFN 使用 SwiGLU，每层有三个 FFN 权重矩阵。

Token embedding 参数量为
$$Vd=50257\times1600=80,411,200.$$

每层 attention 包含 $W_Q,W_K,W_V,W_O$ 四个 $d\times d$ 矩阵，参数量为
$$4d^2=4\times1600^2=10,240,000.$$

每层 SwiGLU FFN 参数量为
$$3dd_{\mathrm{ff}}=3\times1600\times4288=20,582,400.$$

每层有两个 RMSNorm，共 $2d=3200$ 个参数，因此每个 Transformer block 参数量为
$$10,240,000+20,582,400+3,200=30,825,600.$$

48 层共有
$$48\times30,825,600=1,479,628,800$$
个参数。Final RMSNorm 有 $d=1600$ 个参数，LM head 有
$$Vd=50257\times1600=80,411,200$$
个参数。因此总参数量为
$$N=2Vd+L(4d^2+3dd_{\mathrm{ff}}+2d)+d=1,640,452,800.$$

即约为
$$\boxed{1.640\times10^9\text{ parameters}}.$$

FP32 每个参数占 4 Bytes，所以加载模型参数需要
$$1,640,452,800\times4=6,561,811,200\text{ Bytes},$$
即约
$$\boxed{6.56\text{ GB}}\quad(\text{约 }6.11\text{ GiB}).$$

(b) 列出完成一次 GPT-2-XL 规模模型前向传播所需要执行的所有矩阵乘法。

这些矩阵乘法总共需要多少 FLOPs？假设输入序列长度等于上下文长度。

解答

对于 GPT-2-XL，$T=1024,d=1600,H=25$，因此每个 attention head 的维度为 $d_h=d/H=64$。Embedding 是查表操作；RMSNorm、RoPE、softmax 和 SwiGLU 中的逐元素运算也不是矩阵乘法，因此不计入这里的 FLOPs。

1. **Q、K、V projection**：分别执行 $(T\times d)(d\times d)$，每个需要 $2Td^2$ FLOPs，三个共
   $$6Td^2=6(1024)(1600)^2=\boxed{15,728,640,000}.$$

2. **Attention score $QK^\top$**：每个 head 执行 $(T\times d_h)(d_h\times T)$，所有 head 共
   $$2HT^2d_h=2T^2d=2(1024)^2(1600)=\boxed{3,355,443,200}.$$

3. **Attention probability 与 $V$ 相乘**：每个 head 执行 $(T\times T)(T\times d_h)$，所有 head 共
   $$2T^2d=\boxed{3,355,443,200}.$$

4. **Attention output projection**：执行 $(T\times d)(d\times d)$，需要
   $$2Td^2=2(1024)(1600)^2=\boxed{5,242,880,000}.$$

所以每层 attention 总 FLOPs 为
$$8Td^2+4T^2d=\boxed{27,682,406,400}.$$

5. **SwiGLU FFN**：三个矩阵乘法分别为 $(T\times d)(d\times d_{\mathrm{ff}})$、$(T\times d)(d\times d_{\mathrm{ff}})$、$(T\times d_{\mathrm{ff}})(d_{\mathrm{ff}}\times d)$，因此每层 FFN 总 FLOPs 为
   $$6Tdd_{\mathrm{ff}}=6(1024)(1600)(4288)=\boxed{42,152,755,200}.$$

因此每个 Transformer block 总 FLOPs 为
$$27,682,406,400+42,152,755,200=69,835,161,600,$$
48 层共
$$48\times69,835,161,600=\boxed{3,352,087,756,800}.$$

6. **LM head**：执行 $(T\times d)(d\times V)$，需要
   $$2TdV=2(1024)(1600)(50257)=\boxed{164,682,137,600}.$$

因此单次前向传播总 FLOPs 为
$$F_{\mathrm{total}}=L(8Td^2+4T^2d+6Tdd_{\mathrm{ff}})+2TdV$$
$$=\boxed{3,516,769,894,400\text{ FLOPs}}\approx\boxed{3.52\text{ TFLOPs}}.$$

(c) 根据上面的分析，模型哪些部分消耗最多 FLOPs？

解答

在上下文长度为 1024 时，消耗 FLOPs 最多的是 SwiGLU FFN，约占总 FLOPs 的 $57.5%$；其次是 multi-head attention，约占 $37.8%$。

(d) 对 GPT-2-small（12 层，d_model=768，12 个注意力头）、GPT-2-medium（24 层，d_model=1024，16 个注意力头）、GPT-2-large（36 层，d_model=1280，20 个注意力头）重复上面的分析。

随着模型规模增大，Transformer 语言模型中哪些模块占总 FLOPs 的比例会相对上升，哪些会相对下降？

解答

各模型的 $d_{\mathrm{ff}}$ 取 $\frac83d$ 最接近的 64 的整数倍：

| Model        | $L$ |  $d$ | $H$ | $d_{\mathrm{ff}}$ |
| ------------ | --: | ---: | --: | ----------------: |
| GPT-2-small  |  12 |  768 |  12 |              2048 |
| GPT-2-medium |  24 | 1024 |  16 |              2752 |
| GPT-2-large  |  36 | 1280 |  20 |              3392 |
| GPT-2-XL     |  48 | 1600 |  25 |              4288 |

均取 $T=1024,V=50257$。各组件 FLOPs 及占比如下：

| Component             |            GPT-2-small |           GPT-2-medium |            GPT-2-large |                GPT-2-XL |
| --------------------- | ---------------------: | ---------------------: | ---------------------: | ----------------------: |
| Attention projections |  57.98 GFLOPs (19.88%) | 206.16 GFLOPs (24.83%) | 483.18 GFLOPs (27.32%) | 1006.63 GFLOPs (28.62%) |
| $QK^\top$             |   19.33 GFLOPs (6.63%) |   51.54 GFLOPs (6.21%) |   96.64 GFLOPs (5.46%) |   161.06 GFLOPs (4.58%) |
| $AV$                  |   19.33 GFLOPs (6.63%) |   51.54 GFLOPs (6.21%) |   96.64 GFLOPs (5.46%) |   161.06 GFLOPs (4.58%) |
| SwiGLU FFN            | 115.96 GFLOPs (39.76%) | 415.54 GFLOPs (50.05%) | 960.33 GFLOPs (54.30%) | 2023.33 GFLOPs (57.53%) |
| LM head               |  79.05 GFLOPs (27.10%) | 105.40 GFLOPs (12.70%) |  131.75 GFLOPs (7.45%) |   164.68 GFLOPs (4.68%) |
| **Total**             |      **291.65 GFLOPs** |      **830.17 GFLOPs** |     **1768.53 GFLOPs** |      **3516.77 GFLOPs** |

随着模型规模增大，FFN 和 attention projection 的 FLOPs 占比总体上升，而 LM head 的占比明显下降；$QK^\top$ 和 $AV$ 的占比也逐渐下降。原因是固定 $T$ 时，FFN 和 projection 主要按 $Ld^2T$ 增长，$QK^\top$ 和 $AV$ 按 $LdT^2$ 增长，而 LM head 只按 $dTV$ 增长。

(e) 将 GPT-2-XL 的上下文长度增大至 16384。单次前向传播总 FLOPs 会如何变化？各个模型组件的 FLOPs 相对占比会如何变化？

解答

现在
$$T'=16384=16\times1024.$$

Attention projection、FFN 和 LM head 都与 $T$ 成正比，因此 FLOPs 变为原来的 $16$ 倍；$QK^\top$ 和 $AV$ 与 $T^2$ 成正比，因此变为
$$16^2=\boxed{256}$$
倍。

GPT-2-XL 的总 FLOPs 为
$$F_{\mathrm{total}}=L(8Td^2+4T^2d+6Tdd_{\mathrm{ff}})+2TdV.$$

代入 $T=16384$ 得
$$\boxed{F_{\mathrm{total}}=133,577,729,638,400\text{ FLOPs}}\approx\boxed{133.58\text{ TFLOPs}}.$$

与 $T=1024$ 时的 $3.51677$ TFLOPs 相比，总 FLOPs 增加
$$\frac{133.58}{3.51677}\approx\boxed{37.98\text{ 倍}}.$$

各组件 FLOPs 占比如下：

| Component             |             FLOPs | Percentage |
| --------------------- | ----------------: | ---------: |
| Attention projections |      16.11 TFLOPs |     12.06% |
| $QK^\top$             |      41.23 TFLOPs |     30.87% |
| $AV$                  |      41.23 TFLOPs |     30.87% |
| SwiGLU FFN            |      32.37 TFLOPs |     24.24% |
| LM head               |       2.63 TFLOPs |      1.97% |
| **Total**             | **133.58 TFLOPs** |   **100%** |

因此 attention 整体占比约为
$$12.06%+30.87%+30.87%=\boxed{73.79%}.$$

随着上下文长度由 1024 增加到 16384，计算瓶颈由 FFN 转向 self-attention，因为 $QK^\top$ 和 $AV$ 的复杂度为 $O(T^2)$，而 FFN、attention projection 和 LM head 都只随 $T$ 线性增长。

# 4 训练 Transformer LM
## 4.1 交叉熵损失
### 习题 cross_entropy（1 分）

对于任意长度为 $m+1$ 的序列 $x$，模型会在每个位置 $i$ 预测下一个 token 的条件分布：

$$p_\theta(x_{i+1}\mid x_{1:i}).$$

给定训练数据集 $D$，整体交叉熵（负对数似然）损失定义为：

$$\ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_\theta(x_{i+1}\mid x_{1:i}).$$

Transformer 单次前向传播即可同时得到序列所有位置的下一词预测分布。模型在位置 $i$ 输出原始得分向量 $o_i$（logits），目标 token $x_{i+1}$ 的预测概率为：

$$p(x_{i+1}\mid x_{1:i})=\operatorname{softmax}(o_i)[x_{i+1}]=\frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{V}\exp(o_i[a])},$$

其中 $V$ 表示词表大小，$o_i[k]$ 表示位置 $i$ 的 logits 向量中索引 $k$ 对应的数值。

因此，位置 $i$ 的交叉熵损失为：

$$\ell_i=-\log\operatorname{softmax}(o_i)[x_{i+1}].$$

该损失在数学上等价于目标 token 的狄拉克分布与模型预测分布之间的交叉熵。

实现规范：

Softmax 输入每行先减去该行最大值，防止指数溢出；

合并$\log$与$\exp$运算，减少计算开销；

兼容任意前置批量维度，最终返回批次平均损失；

实现适配器adapters.run_cross_entropy，执行uv run pytest -k test_cross_entropy通过单元测试。

## 4.2 SGD 随机梯度下降
### 习题 learning_rate_tuning（1 分）

分别设置学习率10、100、1000，各训练 10 轮，描述损失收敛 / 发散行为。
```
uv run python scripts/learning_rate_tuning.py
```
```
step           | lr=1          | lr=10         | lr=100        | lr=1000
------------------------------------------------------------------------
   0 |  2.62714e+01 |  2.62714e+01 |  2.62714e+01 |  2.62714e+01
   1 |  2.52311e+01 |  1.68137e+01 |  2.62714e+01 |  9.48398e+03
   2 |  2.45225e+01 |  1.23943e+01 |  4.50746e+00 |  1.63803e+06
   3 |  2.39594e+01 |  9.69725e+00 |  1.07874e-01 |  1.82214e+08
   4 |  2.34826e+01 |  7.85477e+00 |  1.15257e-16 |  1.47593e+10
   5 |  2.30644e+01 |  6.51251e+00 |  1.28461e-18 |  9.31481e+11
   6 |  2.26893e+01 |  5.49243e+00 |  4.32572e-20 |  4.78192e+13
   7 |  2.23476e+01 |  4.69344e+00 |  2.57686e-21 |  2.05739e+15
   8 |  2.20327e+01 |  4.05316e+00 |  2.21060e-22 |  7.58309e+16
   9 |  2.17399e+01 |  3.53075e+00 |  2.45622e-23 |  2.43501e+18
```
训练效果

运行题目给出的 SGD 例子 10 步后，学习率 $10$ 比原来的 $1$ 收敛更快；学习率 $100$ 前期振荡（第一步 loss 不降），随后很快降至 0；学习率 $1000$ 明显发散，loss 持续快速增大。
## 4.3 AdamW优化器
### 习题 adamw（2 分）
AdamW 算法流程

初始化与参数 $\theta$ 同形状的一阶矩和二阶矩：

$$m_0=0,\qquad v_0=0.$$

每一步迭代 $t=1,\ldots,T$：

$$g_t=\nabla_\theta\ell(\theta_{t-1};B_t),$$

$$\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},$$

$$\theta\leftarrow(1-\alpha\lambda)\theta,$$

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,$$

$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,$$

$$\theta_t=\theta-\alpha_t\frac{m_t}{\sqrt{v_t}+\varepsilon}.$$

常用配置：

$$\beta_1=0.9,\qquad\beta_2=0.999,\qquad\varepsilon=10^{-8}.$$

LLaMA 和 GPT-3 常使用 $\beta_2=0.95$，$\lambda$ 为权重衰减系数。

实现时继承 torch.optim.Optimizer，构造函数接收 lr、betas、eps 和 weight_decay，并使用 self.state 为每个参数保存
$m_t$、$v_t$ 和 $t$。最后实现 adapters.get_adamw_cls 并通过测试 uv run pytest -k test_adamw。

### 习题 adamw_accounting（2 分）
假设全部张量使用 FP32 精度，拆分显存占用：参数、梯度、激活值、优化器一阶 / 二阶矩。

记 $B$ 为 batch size，$T$ 为 context length，$L$ 为层数，$d$ 为 `d_model`，$H$ 为 head 数，$V$ 为词表大小，且 $d_{ff}=\frac83d$。本题模型未绑定输入、输出 embedding，因此参数元素数为
$$N=2Vd+L(4d^2+3dd_{ff}+2d)+d.$$

(a) 写出显存占用代数表达式，变量：batch_size、词表大小、上下文长度、层数、d_model、头数，$d_{ff}=8/3\cdot d_{model}$；仅统计 Transformer 块内 RMSNorm、多头注意力、SwiGLU、最终归一化、输出 LM 头、交叉熵相关激活。

解答

FP32 下各部分峰值内存如下（1 个元素为 4 Bytes）：
- 参数：$M_{param}=4N$ Bytes；
- 梯度：$M_{grad}=4N$ Bytes；
- AdamW 状态（一阶、二阶动量）：$M_{opt}=8N$ Bytes；
- 激活：按题目指定的中间量逐项保存，两个 RMSNorm、Q/K/V、attention weighted sum、attention output、FFN output 合计每层 $8BTd$；attention scores 与 softmax 合计 $2BHT^2$；SwiGLU 的 $W_1/W_3$ 输出、SiLU 输出和逐元素乘积合计 $4BTd_{ff}$。再加 final RMSNorm 的 $BTd$、LM head logits 和 cross-entropy 所需量的 $2BTV$，故
  $$M_{act}=4B\left[L(8Td+2HT^2+4Td_{ff})+Td+2TV\right]\text{ Bytes}.$$
总峰值内存为
$$M_{total}=16N+4B\left[L(8Td+2HT^2+4Td_{ff})+Td+2TV\right]\text{ Bytes}.$$
(b) 代入 GPT2-XL 配置，写出仅依赖 batch_size 的总显存表达式，求解 80GB 显存上限下最大批次大小。

解答

对 GPT-2 XL，取 $V=50257,T=1024,L=48,d=1600,H=25,d_{ff}=4288$，有
$$N=1,640,452,800,$$
因此
$$M_{total}=16.373391360\,B+26.247244800\ \text{GB}.$$
在 80 GB 中可容纳的最大整数 batch size 为
$$\boxed{B_{max}=3}.$$
(c) 单次 AdamW 迭代总浮点运算 FLOPs 代数表达式 + 简要说明。

解答

对每个参数元素，weight decay 约需 2 FLOPs；更新一阶动量需 3 FLOPs；更新二阶动量需 4 FLOPs；最终的开方、加法、除法、乘法和减法需约 5 FLOPs。因此一次 AdamW optimizer step 约需
$$\boxed{14N\ \text{FLOPs}},$$
忽略每步只计算一次的标量偏差修正开销。

(d) 硬件：单张 H100，理论 FP32 峰值 495 TFLOPS，MFU 利用率 50%；单步反向算力是前向 2 倍；模型 GPT2-XL，batch=1024，共 40 万步，计算总训练小时。

解答

第 3 节已算得 GPT-2 XL 单条长度 1024 序列的前向传播为
$F_{fwd}=3.5167698944\times10^{12}$ FLOPs。反向传播取前向的 2 倍，则 400K 步、batch size 1024 的计算量为
$$400000\times1024\times3F_{fwd}.$$
H100 在 50% MFU 下的有效吞吐为 $0.5\times495=247.5$ TFLOP/s，所以训练耗时为
$$\frac{400000\times1024\times3F_{fwd}}{247.5\times10^{12}}
=1.7460\times10^7\text{ s}\approx\boxed{4850.1\text{ 小时}}.$$

## 4.4 余弦退火学习率调度
### 习题 learning_rate_schedule（1 分）
Transformer 训练通常使用三段式余弦退火调度。输入包括当前步数 $t$、最大学习率 $\alpha_{\max}$、最小学习率 $\alpha_{\min}$、热身步数 $T_w$ 和退火结束步数 $T_c$。

1. 热身阶段（$t<T_w$）：

$$\alpha_t=\frac{t}{T_w}\alpha_{\max}.$$

2. 余弦退火阶段（$T_w\leq t\leq T_c$）：

$$\alpha_t=\alpha_{\min}+\frac{1}{2}\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max}-\alpha_{\min}).$$

3. 退火结束阶段（$t>T_c$）：

$$\alpha_t=\alpha_{\min}.$$

实现余弦退火学习率调度函数，封装适配器 `adapters.get_lr_cosine_schedule`，并通过对应测试 uv run pytest -k test_get_lr_cosine_schedule。

## 4.5 习题 gradient_clipping（1 分）

已实现所有非空参数梯度的联合 $\ell_2$ 范数计算，并在范数超限时原地乘以 $M/(\lVert g\rVert_2+10^{-6})$。
