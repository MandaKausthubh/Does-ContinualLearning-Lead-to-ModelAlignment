# LLM Continual Learning for Model Alignment

This project investigates whether continual learning via self-distillation can serve as a stable mechanism for debiasing and model alignment in Large Language Models.

## Overview

Large language models exhibit biases related to gender, race, and religion due to patterns in pretraining data. While supervised fine-tuning (SFT) can reduce harmful outputs, it is off-policy and often causes catastrophic forgetting. This project implements Self-Distillation Fine-Tuning (SDFT) to enable continual learning while preserving prior capabilities.

### Key Components

1. **Baseline Evaluation**: Establish initial bias metrics for pretrained models
2. **Standard Fine-Tuning (SFT)**: Traditional supervised fine-tuning baseline
3. **Self-Distillation Fine-Tuning (SDFT)**: Continual learning with distillation regularization
4. **Bias Measurement**: Comprehensive evaluation using sentiment and toxicity metrics

## Project Structure

```
./
├── configs/                  # Configuration files
│   ├── model_config.yaml    # Model and LoRA settings
│   ├── training_config.yaml # Training hyperparameters
│   └── eval_config.yaml     # Evaluation settings
│
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model loading and wrappers
│   ├── training/            # Training loops (SFT, SDFT)
│   ├── evaluation/          # Bias metrics and evaluation
│   ├── pipelines/           # Execution pipelines
│   └── utils/               # Utilities and helpers
│
├── experiments/             # Experiment outputs
│   ├── exp_01_baseline/
│   ├── exp_02_sft/
│   └── exp_03_sdft/
│
├── results/                 # Evaluation results and plots
│   ├── metrics/
│   ├── plots/
│   └── logs/
│
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.9+
- TPU access (for distributed training) or GPU
- HuggingFace account with access to LLaMA models

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-continual-alignment
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up HuggingFace:
```bash
huggingface-cli login
```

## Usage

### Quick Start

Run the full pipeline (baseline + SFT + SDFT + evaluation):

```bash
python main.py full --model llama-1b --device tpu
```

### Individual Commands

#### 1. Baseline Evaluation

Evaluate pretrained model to establish baseline bias metrics:

```bash
python main.py baseline \
    --model llama-1b \
    --datasets dolly oasst1 \
    --num-samples 500 \
    --device tpu
```

#### 2. Standard Fine-Tuning (SFT)

Run standard supervised fine-tuning:

```bash
python main.py sft \
    --model llama-1b \
    --phase1-dataset dolly \
    --phase2-dataset stereoset \
    --device tpu
```

To skip specific phases:
```bash
python main.py sft \
    --skip-phase1 \  # Skip general fine-tuning
    --skip-phase2    # Skip alignment fine-tuning
```

#### 3. Self-Distillation Fine-Tuning (SDFT)

Run fine-tuning with self-distillation regularization:

```bash
python main.py sdft \
    --model llama-1b \
    --phase1-dataset dolly \
    --phase2-dataset stereoset \
    --alpha 0.5 \
    --temperature 2.0 \
    --device tpu
```

Parameters:
- `--alpha`: Distillation weight (0-1). Higher = more regularization from reference model
- `--temperature`: Softmax temperature for distillation. Higher = softer distributions

#### 4. Evaluation

Evaluate a specific checkpoint:

```bash
python main.py evaluate \
    --model-path ./experiments/exp_02_sft/phase2/final \
    --model-name llama-1b \
    --datasets dolly oasst1 \
    --output ./results/eval_sft.json \
    --device tpu
```

#### 5. Comparison

Compare results across experiments:

```bash
python main.py compare \
    --experiments ./experiments/exp_01_baseline ./experiments/exp_02_sft ./experiments/exp_03_sdft \
    --output ./results/comparison.json
```

## Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
base_model:
  name: "meta-llama/Llama-3.2-1B"  # or 3B, 8B
  cache_dir: "./cache/models"
  torch_dtype: "bfloat16"

lora:
  enabled: true
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
device:
  type: "tpu"  # or cuda, cpu
  num_devices: 8

training:
  phase1:
    dataset: "dolly"  # or oasst1
    batch_size: 4
    learning_rate: 2.0e-5
    num_epochs: 3

  phase2:
    dataset: "stereoset"
    batch_size: 4
    learning_rate: 1.0e-5
    num_epochs: 5

  sdft:
    enabled: true
    alpha: 0.5
    temperature: 2.0
```

## Bias Measurement

### Metrics

1. **Sentiment Bias (B_sent)**:
   - Uses RoBERTa-base-sentiment classifier
   - B_sent = (1/N) * Σ[S(o_i^(m)) - S(o_i^(f))]

2. **Toxicity Bias (B_tox)**:
   - Uses Detoxify or RoBERTa-hate-speech classifier
   - B_tox = (1/N) * Σ[T(o_i^(m)) - T(o_i^(f))]

3. **Flip Rate (F)**:
   - Measures sensitivity to gender substitution
   - F = (1/N) * Σ[1[ŷ_i^(m) ≠ ŷ_i^(f)]]

### Gender Swapping

The implementation swaps:
- Pronouns: he↔she, his↔her, himself↔herself
- Common nouns: man↔woman, male↔female
- Names: Random male↔female name pairs

## Datasets

### Training Datasets

- **Dolly-15k**: Instruction-following dataset (~15K samples)
- **OASST1**: OpenAssistant conversations (~40K samples)
- **StereoSet**: Bias evaluation and alignment dataset (~10K samples)

### Data Preprocessing

- Tokenization with model's tokenizer
- Maximum sequence length: 512 tokens
- Padding and truncation
- Gender augmentation for bias testing

## Distributed Training (TPU)

The implementation supports 8x TPU training using PyTorch/XLA:

```python
# Example TPU launch
python main.py sft --device tpu --model llama-1b
```

Key features:
- Parallel loading with `torch_xla.distributed.parallel_loader`
- Synchronized metrics across devices
- Checkpoint saving from master process only

## Monitoring

### Weights & Biases

Training progress is automatically logged to WandB:
- Loss curves (total, CE, KL)
- Learning rate schedule
- Bias metrics during training

Configure in `configs/training_config.yaml`:
```yaml
logging:
  use_wandb: true
  project: "llm-continual-alignment"
```

### Local Logs

Logs are saved to:
- `./experiments/exp_*/logs/`
- `./results/logs/`

## Results

### Output Structure

```
experiments/
├── exp_01_baseline/
│   ├── baseline_results.json
│   └── logs/
├── exp_02_sft/
│   ├── phase1/
│   │   └── final/
│   ├── phase2/
│   │   └── final/
│   └── config.json
└── exp_03_sdft/
    ├── phase1/
    │   └── final/
    ├── phase2/
    │   └── final/
    └── config.json
```

### Analysis

View comparison results:
```bash
python main.py compare --experiments exp_01_baseline exp_02_sft exp_03_sdft
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size` in config
   - Increase `gradient_accumulation_steps`
   - Enable `gradient_checkpointing`

2. **TPU Connection**:
   ```bash
   # Check TPU status
   gcloud compute tpus list

   # Restart TPU if needed
   gcloud compute tpus start <tpu-name>
   ```

3. **Model Access**:
   - Ensure HuggingFace token has LLaMA access
   - Run `huggingface-cli login`

## Citation

If you use this code, please cite:

```bibtex
@software{llm_continual_alignment,
  title={LLM Continual Learning for Model Alignment},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- LLaMA models by Meta AI
- StereoSet by Nadeem et al.
- Dolly by Databricks
- OASST by OpenAssistant
