# verl+

在 verl 的基础上，添加了对 Process Reward Model、LLM-as-a-Judge 和输出可视化的支持。

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Configuration](#configuration)

## Installation

### Prerequisites

- Python >= 3.11.0 （3.10 may raise some bug in deepseed)
- PyTorch >= 2.0.0 (2.6.0 is Recommended)
- CUDA >= 12.4
- ray==2.48.0
- vllm==0.8.5.post1

### Environment Setup

For conda users:
```bash
conda create -n verl_plus python=3.11
conda activate verl_plus
```

### Install from source

```bash
git clone https://github.com/BiNLP/verl
cd verl
pip install -e .
```

### Install dependencies

```bash
pip install vllm==0.8.5
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install -r requirements.txt
```


## Dataset Preparation

### GSM8K
```
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
```

### LogiQA
```
python3 examples/data_preprocess/logiqa.py --local_dir data/logiqa
```
### LogiQA for Tree Sampling
```
python3 examples/data_preprocess/logiqa_tree.py --local_dir data/logiqa_tree
```



## Usage


```bash

sh bash_scripts/*.sh

```



## Evaluation
### Method 1:
[LogiEval](https://github.com/BiNLP/LogiEval)
### Method 2 (Recommended):
使用 lighteval：
```
sh bash_scripts/eval.sh
```

## Configurations

### Vllm online inference sevice
```
sh bash_scripts/VllmBackend/start_vllm_server.sh
```

### Process Reward Model
```
PRM_PATH="/path/to/prm/model"
reward_model.worker_type = 'prm'
```
### PRM + LLM-as-a-Judge
```
reward_model.worker_type = 'judge'
```
### PRM + Async LLM-as-a-Judge
```
reward_model.worker_type = 'async_judge'
```


## Tree Sampling for RL Training
## TreeRL

Please preprocess the dataset before starting.

The startup script is located at ```bash_script/Step_TreeRL_LogiQA_GAE```:
```
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tree_gae \ # TreeRL 的 Value 并不使用 Critic Model，而是直接用 Backpropagation 之后的结点的 Value 作为 Value
    ...
   reward_model.reward_manager='tree' \ # 使用 Tree Sampling 策略，在生成的时候就计算了每个 Step 的 reward 存储在 Node 的结构中，这样方便后面我们提高效率加入 early stop strategy. 所以这里的 Reward Manage 只是负责将 step reward organize 成 reward tensor.
    trainer.tree_sampling=True \
    trainer.branch_level='step' \ # 'token' 的话就会每个 token 作为一个结点，对于长的推理序列，树会非常大！！！！
    trainer.step_reward_type='treerl' \ # 根据论文里面的公式，使用当前结点和 root 结点及其双亲的 Value 来计算 Reward；如果是 'fol'，则会将每一步翻译成 FOL 并使用 Z3 返回结果（Fragile）！！！！
    trainer.tree_rounds=1 \ # 分支轮数
    trainer.tree_top_k=1 \ # 每次分支出多少条，目前是只在最高熵的 step 进行分支

```
**TODO**
- 1、结点选择问题：反复采样 Top-k 仍然是那 k 个结点的问题；
- 2、FOL Reward 的脆弱性；

