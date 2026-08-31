# CS336 作业 3（缩放）：缩放定律

> 原文版本：26.0.5｜课程：Stanford CS336，2026 年春季
>
> 本文是 `cs336_assignment3_scaling.pdf` 的完整中文翻译。代码、命令、变量名与公式保留原样；图表说明和作业要求译为中文。

## 目录

1. [作业概述](#1-作业概述)
2. [缩放定律回顾](#2-缩放定律回顾)
3. [构建缩放定律](#3-构建缩放定律)
4. [参考文献](#参考文献)

---

<!-- 原 PDF 第 1 页 -->

2026 年春季

## 1 作业概述

在此作业中，您将获得一些关于语言模型缩放定律的实践经验。

### 场景设定

您负责训练 ClosedAI 的下一代语言模型。该模型的训练成本相当于一个小国家的 GDP，并会让南极冰架融化，因此您希望确保把事情做对。具体来说，您会获得一个固定的计算预算（以墙钟时间度量）。您的目标是产出一个训练损失最低的模型（即计算最优模型）。为了实现这一目标，您需要弄清楚如何在 (1) 训练更大的模型与 (2) 在更多 token 上训练之间取得最佳权衡。缩放定律（scaling laws）以经验方式将语言模型的训练损失与模型大小和用于训练的计算量联系起来，经常被用来外推这种权衡 [J. Kaplan et al., 2020; J. Hoffmann et al., 2022]。在本作业中，您将构建缩放定律，以估计在 48 个 B200 小时上训练的模型的计算最优模型大小及其对应的超参数。您不需要自己训练模型，而是通过查询一个训练 API，提交 (1) 模型超参数（即 Transformer 层数、嵌入大小、头数、批大小、学习率、训练 token 数等）和 (2) 实验的最大墙钟时间；API 将返回在相应超参数上训练得到的验证损失。拟合我们缩放定律的预算是额外的 12 个 B200 小时（我们大运行的 FLOPs 预算的 25%）。训练 API 接受范围广泛的超参数值，而您的工作之一就是弄清楚应该探索搜索空间的哪些部分，以便有效利用缩放定律预算。

### 代码结构

本作业说明可在 GitHub 上找到：github.com/stanford-cs336/assignment3-scaling

请使用 Git 克隆存储库。如果有任何更新，我们会通知您，您可以 git pull 获取最新信息。

### 提交方式

您将向 Gradescope 提交以下文件：

- writeup.pdf：完整描述您的拟合缩放定律的方法论，以及如何使用它来预测给定 FLOPs 预算下的最优模型大小。写报告应足够详细，以便可以重现您的结果。
- code.zip：包含您为拟合缩放定律和计算估计而编写的所有代码。

此外，请将您预测的最优超参数和预测的最终验证损失提交到 API。作业成绩的一部分将由您预测的最优模型的性能决定。

<!-- 原 PDF 第 2 页 -->

## 2 缩放定律回顾

我们将回顾 Chinchilla 论文 [J. Hoffmann et al., 2022] 中拟合缩放定律的方法之一。核心问题是：给定一个计算预算 𝐶（我们将用它来训练大语言模型），哪种超参数选择（模型大小、训练 token 数等）会带来最低的训练损失？主要挑战是如何从较小规模下的实验外推到更大规模。对于作业第二部分您自己的工作，欢迎您借鉴其他参考文献中的想法，例如 J. Kaplan et al. [1] 和 G. Yang et al. [3]。

### 2.1 来自 IsoFLOPs 曲线的缩放定律

回想一下，在包含 𝑁 个参数、在包含 𝐷 个 token 的数据集上训练 Transformer 的计算预算近似为 𝐶 = 6𝑁𝐷。J. Hoffmann et al. [2] 中 IsoFLOPs 缩放定律方法的工作原理如下：对于每个计算预算 𝐶，在给定计算预算 𝐶 的情况下训练不同规模 𝑁 的语言模型（数据规模为 𝐷 = 𝐶/(6𝑁)），得到最终训练损失 𝐿。

这会产生一组使用了相同 FLOPs 数量 𝐶𝑖 但模型规模 𝑁𝑖𝑗 不同的运行。经验上，J. Hoffmann et al. [2] 观察到，在固定计算预算 𝐶𝑖 下，最终训练损失 𝐿𝑖𝑗 与模型规模 𝑁𝑖𝑗 之间存在二次关系。一个直觉如下：当 𝑁𝑖 极小时，无论我们投入多少计算，模型都无法拟合数据，因此该区间内的最终训练损失很高。随着模型规模的增大，最终训练损失平滑下降，直到某一点之后，我们的模型变得过大，无法在 𝐶 FLOPs 内承担足够的梯度步来有效训练它（作为一个极端例子，当 𝑁𝑖 → ∞ 时，即使只走一步梯度也会超出我们的计算预算 𝐶，训练因此会在非常高的损失处停止）。寻找缩放定律的方法包括：为每个 𝐶𝑖 确定最优的模型和数据集规模，然后拟合一个幂律，在给定目标 𝐶（通常是一个比之前运行中使用的预算更大的预算）时预测 𝑁 和 𝐷。

为了拟合幂律，我们使用通过以下方法得到的数据点：对于每个预算 𝐶𝑖，利用使用该预算的一组运行（它的 "IsoFLOPs 曲线"），找到使训练损失最小的模型规模 𝑁opt(𝐶𝑖)。这个过程给了我们一系列对 ⟨𝐶𝑖, 𝑁opt(𝐶𝑖)⟩（以及相应的 𝐷opt 序列）。我们使用这些数据点来拟合幂律 𝑁opt ∝ 𝐶^𝑎 和 𝐷opt ∝ 𝐶^𝑏，并用这些幂律将计算最优的模型和数据集规模外推到我们的目标计算预算。

<!-- 原 PDF 第 3 页 -->

#### 题目：`chinchilla_isoflops`——IsoFLOPs 缩放定律（5 分）

编写一个脚本，使用一组训练运行的最终训练损失来重现上述拟合缩放定律的 IsoFLOPs 方法。对于本题，请使用文件 data/isoflops_curves.json 中给出的（合成）训练运行数据。该文件包含一个 JSON 数组，其中每个元素是描述一次训练运行的对象。以下是用于说明格式的前两次运行：

```json
[
  {
    "parameters": 49999999,
    "compute_budget": 6e+18,
    "final_loss": 7.192784500319437
  },
  {
    "parameters": 78730505,
    "compute_budget": 6e+18,
    "final_loss": 6.750171320661809
  },
  ...
]
```

在拟合缩放定律时，scipy 包（尤其是 scipy.optimize.curve_fit）可能会很有用，但您也可以随意使用任何您喜欢的曲线拟合方法。虽然 J. Hoffmann et al. [2] 为每条 IsoFLOP 曲线拟合一个二次函数来找到其最小值，但我们建议您直接取每个计算预算下训练损失最低的那次运行作为最小值。

**(a)** 展示您外推得到的计算最优模型大小，以及您获得的 ⟨𝐶𝑖, 𝑁opt(𝐶𝑖)⟩ 数据点。对于 10²³ FLOPs 的预算，您预测的最优模型大小是多少？对于 10²⁴ FLOPs 呢？

交付物：一张展示模型大小随计算预算变化的缩放定律图，图中显示用于拟合缩放定律的数据点，并外推至至少 10²⁴ FLOPs。然后用一句话回答您预测的最优模型大小。

**解答：** 对于每个计算预算，我选择最终训练损失最低的运行作为该
IsoFLOPs 曲线的最优点。得到的 $\langle C_i,N_{opt}(C_i)\rangle$ 数据点如下：

| $C_i$ (FLOPs) | $N_{opt}(C_i)$ (parameters) |
|---:|---:|
| $6\times10^{18}$ | $7.621\times10^8$ |
| $1\times10^{19}$ | $8.066\times10^8$ |
| $3\times10^{19}$ | $1.537\times10^9$ |
| $6\times10^{19}$ | $1.952\times10^9$ |
| $1\times10^{20}$ | $3.253\times10^9$ |
| $3\times10^{20}$ | $5.904\times10^9$ |
| $6\times10^{20}$ | $6.971\times10^9$ |
| $1\times10^{21}$ | $6.859\times10^9$ |
| $3\times10^{21}$ | $1.215\times10^{10}$ |

在 log-log 空间中使用最小二乘法拟合 $N_{opt}=AC^a$，得到

$$
N_{opt}(C)=1.1634C^{0.46868}.
$$

![计算最优模型大小随计算预算变化的缩放定律](results/chinchilla_isoflops/optimal_model_size.svg)

对于 $10^{23}$ FLOPs，预测的计算最优模型大小为
$7.01\times10^{10}$，即约 **700 亿参数**；对于 $10^{24}$ FLOPs，预测为
$2.06\times10^{11}$，即约 **2060 亿参数**。

**(b)** 展示您外推得到的计算最优数据集大小，以及来自训练运行的 ⟨𝐶𝑖, 𝐷opt(𝐶𝑖)⟩ 数据点。对于 10²³ 和 10²⁴ FLOPs 的预算，您预测的最优数据集大小是多少？

交付物：一张展示数据集大小随计算预算变化的缩放定律图，图中显示用于拟合缩放定律的数据点，并外推至至少 10²⁴ FLOPs。然后用一句话回答您预测的最优数据集大小。

**解答：** 根据 $D=C/(6N)$ 计算每个最优运行对应的数据集大小，得到：

| $C_i$ (FLOPs) | $D_{opt}(C_i)$ (tokens) |
|---:|---:|
| $6\times10^{18}$ | $1.312\times10^9$ |
| $1\times10^{19}$ | $2.066\times10^9$ |
| $3\times10^{19}$ | $3.253\times10^9$ |
| $6\times10^{19}$ | $5.123\times10^9$ |
| $1\times10^{20}$ | $5.123\times10^9$ |
| $3\times10^{20}$ | $8.469\times10^9$ |
| $6\times10^{20}$ | $1.435\times10^{10}$ |
| $1\times10^{21}$ | $2.430\times10^{10}$ |
| $3\times10^{21}$ | $4.116\times10^{10}$ |

在 log-log 空间中对 $D_{opt}=BC^b$ 进行最小二乘拟合，得到

$$
D_{opt}(C)=0.14326C^{0.53132}.
$$

![计算最优数据集大小随计算预算变化的缩放定律](results/chinchilla_isoflops/optimal_dataset_size.svg)

对于 $10^{23}$ FLOPs，预测的计算最优数据集大小为
$2.38\times10^{11}$ tokens，即约 **2380 亿 tokens**；对于 $10^{24}$ FLOPs，
预测为 $8.09\times10^{11}$ tokens，即约 **8090 亿 tokens**。

上述结果可以通过运行 `uv run python scripts/chinchilla_isoflops.py` 复现。

<!-- 原 PDF 第 4 页 -->

## 3 构建缩放定律

在本节中，您将使用通过我们的训练 API（第 3.2 节）在固定计算预算下查询到的训练运行来拟合缩放定律。您的主要目标是选择能够在 48 个 B200 小时上达到最低验证损失的模型大小和超参数；您还需要预测该模型将获得的验证损失。为了为预测的最优模型大小设置超参数，我们建议分析超参数如何影响较小规模设置下的验证损失。开始之前一定要仔细规划您的运行——一旦超出 12 个 B200 小时的缩放定律预算，训练 API 将拒绝进一步的请求。

### 3.1 训练运行细节

训练 API 会在 B200 GPU 上运行真实的训练任务。当您提交一个实验时，它会进入一个共享队列。排队中和运行中的实验都会在您的 12 小时缩放定律预算中预留完整的 max_runtime_seconds。当实验完成或失败时，API 会使用该实验实际报告的运行时间来重新计算您的预算，截断到至少 1 秒、至多 max_runtime_seconds。因此，如果您为一个 12 分钟就能完成的运行预留 30 分钟，未使用的 18 分钟将返还到您的剩余预算中。如果运行超时，则按 max_runtime_seconds 计费。

每个训练任务运行 n_evals 个块。API 每个块训练 total_train_tokens / n_evals 个 token，在每个块之后在验证集上进行评估，并在 status.val_losses 中报告验证损失的序列。对于已完成的实验，最终验证损失是该列表的最后一项。对于失败的超时实验，API 可能会返回部分验证损失，但这些不是完整的运行。

数据顺序是固定的。训练数据是经过分词处理的 DCLM 数据，最多 500B 个 token。没有数据重复遍历（epoching）。对于每个训练配置，API 使用相同的确定性训练 token 顺序。验证集也使用 2¹⁸ 个验证 token。最终的排行榜验证损失将使用更多 token。model_seed 字段只控制模型初始化，不控制数据顺序。

模型架构与作业 1 的架构基本一致，只有一些细微差别。我们在作业存储库中提供了该模型的代码供您参考：https://github.com/stanford-cs336/assignment3-scaling。训练代码仅供参考。您只能通过 API 训练模型。

模型在具有 32K 词表的、经过分词处理的 DCLM 数据上训练。模型的上下文长度为 512。API 支持 AdamW 和 SGD，两者都使用 warmup-cosine 学习率调度。下面的 API 示例使用 AdamW，权重衰减 0.01、梯度裁剪 1.0、预热比例 0.05、最终学习率比例 0.1。训练使用 512-token 序列上的平均下一 token 交叉熵损失。API 模型中没有 dropout。

<!-- 原 PDF 第 5 页 -->

### 3.2 训练 API

您将使用此训练 API 来查询您想要运行的缩放定律实验的最终验证损失。您的 API 密钥将是您的 8 位斯坦福 ID（例如 06123456）。请注意，您必须处于斯坦福网络中才能查询此 API，因此您可能需要使用 VPN。作为一个健全性检查，您应该能够看到 API 文档页面 http://hyperturing.stanford.edu:8000/docs，并通过 /budget 端点验证您的 API 密钥是否有效（更多细节见下文）。训练 API 有以下端点：

所有 API 请求都应在 X-API-Key 请求头中包含您的 API 密钥：

```python
>>> API_BASE_URL = "http://hyperturing.stanford.edu:8000"
>>> API_KEY = <YOUR_API_KEY_HERE>
>>> headers = {"X-API-Key": API_KEY}
>>> requests.get(f"{API_BASE_URL}/budget", headers=headers).json()
{'used_seconds': 30.0, 'remaining_seconds': 43170.0, 'total_budget_seconds': 43200.0}
```

您将把实验提交到队列中，轮询它们的状态，并在实验完成后读取验证损失。

#### POST /submit

给定一个训练配置，将一个训练运行加入队列并返回其实验 ID。请求体应该是一个 JSON 对象，包含以下键：

- architecture_config：描述 Transformer 架构的 JSON 对象。它包含键 attention_bias、head_dim、hidden_size、intermediate_size、num_attention_heads、num_hidden_layers、num_key_value_heads、rms_norm_eps、rope_theta、tie_word_embeddings、dtype 和 vocab_size。dtype 字段必须是 "float32" 或 "bfloat16"。

- optimizer_config：描述优化器的 JSON 对象。我们支持 AdamW 和 SGD 配置；下面的示例使用 AdamW。

- train_batch_size：整数训练批大小。

- val_batch_size：整数验证批大小。

- n_evals：训练期间要执行的验证评估次数。

- total_train_tokens：训练 token 的总数。

- max_runtime_seconds：为这个实验预留的最大墙钟运行时间。此值必须在 1 秒到 12 小时之间。

- model_seed：用于初始化模型的随机种子。

API 将 seq_len 固定为 512，将 n_validation_tokens 固定为 2¹⁸ ≈ 262k，并验证若干一致性条件。例如，hidden_size 必须等于 num_attention_heads * head_dim，num_attention_heads 必须能被 num_key_value_heads 整除，total_train_tokens 必须能被 512 * train_batch_size 整除。

响应是一个 JSON 对象，包含以下键：

- experiment_id：已排队实验的整数 ID。

- budget_summary：一个 JSON 对象，包含 used_seconds、remaining_seconds 和 total_budget_seconds。

如果您提交相同的训练配置两次，API 会返回 409 响应，并且不会预留额外的预算。如果 max_runtime_seconds 超过您的剩余预算，API 会返回 400 响应。

要查询 API，请发出 HTTP POST 请求：

```python
>>> import requests
>>> API_BASE_URL = "http://hyperturing.stanford.edu:8000"
>>> headers = {"X-API-Key": <YOUR_API_KEY_HERE>}
>>> training_config = {
...     "architecture_config": {
...         "attention_bias": False,
...         "head_dim": 64,
...         "hidden_size": 448,
...         "intermediate_size": 1280,
...         "num_attention_heads": 7,
...         "num_hidden_layers": 9,
...         "num_key_value_heads": 7,
...         "rms_norm_eps": 1e-6,
...         "rope_theta": 1_000_000,
...         "tie_word_embeddings": False,
...         "dtype": "bfloat16",
...         "vocab_size": 32_000,
...     },
...     "optimizer_config": {
...         "lr_scheduler": {
...             "peak_value": 3e-4,
...             "final_lr_frac": 0.1,
...             "warmup_frac": 0.05,
...             "init_value": 0.0,
...         },
...         "weight_decay": 1e-2,
...         "beta1": 0.9,
...         "beta2": 0.95,
...         "eps": 1e-8,
...         "eps_root": 1e-8,
...         "grad_clip_norm": 1.0,
...     },
...     "train_batch_size": 128,
...     "val_batch_size": 32,
...     "n_evals": 16,
...     "total_train_tokens": 1_048_576,
...     "max_runtime_seconds": 30.0,
...     "model_seed": 0,
... }
>>> requests.post(f"{API_BASE_URL}/submit", headers=headers, json=training_config).json()
{'experiment_id': 1, 'budget_summary': {'used_seconds': 30.0, 'remaining_seconds': 43170.0,
'total_budget_seconds': 43200.0}}
```

<!-- 原 PDF 第 6 页 -->

#### GET /budget

给定一个 API 密钥，返回您的实验已预留或已使用的墙钟预算。响应是一个 JSON 对象，包含以下键：

- used_seconds：您提交的实验已使用或已预留的总墙钟秒数。

- remaining_seconds：您缩放定律预算中剩余的墙钟秒数。

- total_budget_seconds：缩放定律预算中的总墙钟秒数。

要查询 API，请发出 HTTP GET 请求：

```python
>>> requests.get(f"{API_BASE_URL}/budget", headers=headers).json()
{'used_seconds': 30.0, 'remaining_seconds': 43170.0, 'total_budget_seconds': 43200.0}
```

#### GET /experiments

返回使用您的 API 密钥提交的所有实验的列表。每个实验包含以下键：

- experiment_id：/submit 返回的实验 ID。

- training_config：为此实验提交的训练配置。

- status：当前的实验状态。

status.status_type 字段是以下之一：

- "queued"：实验正在等待运行。

- "running"：实验正在运行。状态包含 val_losses，即到目前为止观察到的验证损失。实验在被排队后会进入 running 状态。这包括启动时间和抢占，因此您的实验显示为运行的时间可能比您指定的最大运行时间更长。

- "completed"：实验已完成。状态包含 used_runtime_seconds、val_losses 和 completed_at。

- "failed"：实验失败或超时。超时失败包含 partial_val_losses，您可以查看，但失败的实验要么崩溃了，要么超时了。

要查询 API，请发出 HTTP GET 请求：

```python
>>> requests.get(f"{API_BASE_URL}/experiments", headers=headers).json()
[{'experiment_id': 1, 'training_config': {...}, 'status': {'status_type': 'queued',
'queued_at': '...'}}]
```

#### GET /experiment/{experiment_id}

返回单个实验，响应格式与 /experiments 返回的条目相同。使用此端点轮询已提交的实验，直到其 status.status_type 为 "completed"：

```python
>>> requests.get(f"{API_BASE_URL}/experiment/1", headers=headers).json()
{'experiment_id': 1, 'training_config': {...}, 'status': {'status_type': 'completed',
'val_losses': [3.1, 2.9, ...], ...}}
```

对于已完成的实验，请使用 status.val_losses 中的最后一项作为该次运行的最终验证损失。

#### POST /final_submission

提交您预测的最优超参数和预测的最终验证损失。请求体应包含：

- training_config：您预测的最优训练配置。

- predicted_final_loss：该配置的预测最终验证损失。

您可以重新提交此端点；最新的提交会替换之前的提交。

```python
>>> final_submission = {
...     "training_config": training_config,
...     "predicted_final_loss": 2.75,
... }
>>> requests.post(f"{API_BASE_URL}/final_submission", headers=headers,
json=final_submission).json()
{'training_config': {...}, 'predicted_final_loss': 2.75, 'submitted_at': '...'}
```

#### GET /final_submission

返回您当前的最终提交，如果您还没有提交，则返回 None：

```python
>>> requests.get(f"{API_BASE_URL}/final_submission", headers=headers).json()
{'training_config': {...}, 'predicted_final_loss': 2.75, 'submitted_at': '...'}
```

<!-- 原 PDF 第 7 页 -->

### 3.3 排行榜

现在，您将使用训练 API 为 48 个 B200 小时的正式运行选择模型配置。目标是使最终模型的验证损失最小。

#### 题目：`scaling_laws`——构建缩放定律排行榜（50 分）

构建一条缩放定律，并用它选择您认为能在 48 个 B200 小时内达到最低验证损失的模型大小和超参数。您还需要预测该模型最终会得到的验证损失。为了构建缩放定律，您将使用训练 API 查询不同实验配置的最终验证损失（见第 3.2 节）；用于拟合缩放定律的实验总量不得超过 12 个 B200 小时。API 会强制执行这一硬性上限。

交付物：一份排版完整的报告，全面描述您拟合缩放定律所使用的方法，以及如何利用该缩放定律预测给定 FLOPs 预算下的最优模型大小和相应预测值。报告应解释各项设计决策，并提供足够细节，使他人能够复现您的方法和结果。

对于 48 个 B200 小时的正式运行，课程原则上不限制您报告的超参数。所有实验均在单张 B200 GPU 上运行。

建议至少考虑以下问题；报告还应说明您如何对每项因素作出决策：

- 在固定的 12 个 B200 小时缩放定律预算下，您如何决定查询哪些训练运行？
- 您如何拟合缩放定律？请描述所采用的具体方法。熟悉 Kaplan et al. [1] 和 Hoffmann et al. [2] 的方法可能会有所帮助。
- 缩放定律对实验数据的拟合效果如何？
- 对于给定的 48 个 B200 小时预算，缩放定律预测的最优模型大小是多少？预测损失是多少？
- 如果训练一个具有该预测最优参数量的模型，您会使用哪些超参数？估计给定模型配置的非嵌入参数量时，请使用

  $$
  N_{\mathrm{non\text{-}embedding}}=12n_{\mathrm{layer}}d_{\mathrm{model}}^2.
  $$

除报告外，还需通过 API 提交：（1）预测的最优超参数；（2）该模型的预测验证损失。作业成绩的一部分取决于最终预测模型的实际表现。
