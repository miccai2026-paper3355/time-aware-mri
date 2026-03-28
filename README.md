# Time-Aware Multi-View MRI Benchmark - Evaluation Code

**Evaluation scripts for temporal reasoning in longitudinal brain MRI interpretation**

This repository contains inference code for evaluating vision-language models on the Time-Aware Multi-View MRI Benchmark, as described in our paper submitted to MICCAI 2026.

---

## 📋 Overview

This codebase implements the **Agentic Resident-Attending Protocol** for structured temporal MRI interpretation (Table 2 in paper). The workflow:

1. **Resident Agent**: Analyzes multi-timepoint, multi-sequence, multi-view MRI grids
2. **Output**: Structured JSON with reasoning steps and final diagnosis
3. **Evaluation**: Four clinical tasks (change segmentation, quantification, temporal ordering, progression localization)

### Supported Models

| Model | Implementation | File |
|-------|---------------|------|
| **GPT-4o/GPT-5** | OpenAI API | `openai_model.py` (not included) |
| **Gemini 2.5/3** | Google Gemini API | `gemini_model.py` |
| **Claude 3.5** | Anthropic API | `anthropic_model.py` |
| **InternVL 3/3.5** | HuggingFace Local | `internvl_model.py` |
| **Qwen3-VL** | HuggingFace Local | `qwen3_hf_local.py` |
| **MedGemma/PaliGemma** | HuggingFace Local | `paligemma_model.py` |
| **GLM-4V/5V** | ZhipuAI API | `glm_model.py` |

---

## 🗂️ Data Format

### Input JSON Structure

```json
{
  "qa_id": "UCSF-GBM_0001_single",
  "patient_id": "UCSF-0001",
  "dataset": "UCSF-GBM",
  "age": "56",
  "sex": "M",
  "question": ["Compare baseline and follow-up. Has the enhancing lesion increased, decreased, or remained stable?"],
  "options": {
    "A": "Increased",
    "B": "Decreased", 
    "C": "Stable",
    "D": "Cannot determine"
  },
  "answer": "A",
  "images": [
    {
      "path": "UCSF-0001/timepoint_0/T1CE_axial.png",
      "timepoint": 0,
      "sequence": "T1CE",
      "view": "axial",
      "filename": "T1CE_axial.png"
    },
    {
      "path": "UCSF-0001/timepoint_1/T1CE_axial.png",
      "timepoint": 1,
      "sequence": "T1CE",
      "view": "axial",
      "filename": "T1CE_axial.png"
    }
  ]
}
```

### Output JSONL Format

```json
{
  "qa_id": "UCSF-GBM_0001_single",
  "patient_id": "UCSF-0001",
  "model": "InternVL3-8B",
  "raw_text": "{\\"steps\\": [...], \\"answer\\": \\"A\\"}",
  "steps": [
    "Baseline T1CE shows heterogeneous enhancement in right frontal lobe",
    "Follow-up demonstrates increased size and irregular margins",
    "Interval increase in enhancing component consistent with progression"
  ],
  "answer": "A",
  "valid_json": true,
  "latency_s": 4.2
}
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision pillow transformers anthropic openai google-generativeai
```

### API Keys Setup

```bash
# For closed-source models
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export ZHIPUAI_API_KEY="your-zhipuai-key"

# For HuggingFace models (if using gated models)
export HF_TOKEN="your-hf-token"
```

### Basic Usage

#### **API-Based Models (Anthropic Claude)**

```bash
python anthropic_model.py \
    --samples benchmark_samples.json \
    --root /path/to/image/root \
    --out results/claude_results.jsonl \
    --model claude-3-5-sonnet-20241022 \
    --concurrency 4
```

#### **Local HuggingFace Models (InternVL)**

```bash
python internvl_model.py \
    --samples benchmark_samples.json \
    --root /path/to/image/root \
    --out results/internvl_results.jsonl \
    --model OpenGVLab/InternVL3-8B
```

#### **Local with Quantization (Qwen3-VL)**

```bash
python qwen3_hf_local.py \
    --samples benchmark_samples.json \
    --root /path/to/image/root \
    --out results/qwen3_results.jsonl \
    --model Qwen/Qwen2.5-VL-8B-Instruct
```

---

## 📊 Key Features

### 1. **Multi-View Grid Creation**

Each timepoint is presented as a grid combining:
- **Sequences**: T1, T2, FLAIR, T1CE, DWI, ADC
- **Views**: Axial, Coronal, Sagittal
- **Layout**: Sequences × Views matrix with labels

Example grid for 3 sequences × 3 views = 9 images per timepoint.

### 2. **Temporal Series Handling**

- Processes 3-4 serial timepoints per patient
- Spanning months to years (average 7.4 months)
- Automatically sorts chronologically
- Labels each timepoint in prompt

### 3. **Resume Support**

All scripts support resuming interrupted runs:
- Tracks completed `qa_id`s in output file
- Skips already processed samples
- Safe for cluster job interruptions

### 4. **Structured Output Schema**

```json
{
  "steps": ["reasoning step 1", "reasoning step 2", ...],
  "answer": "A",
  "answer_key": "A",
  "answer_option": "Increased"
}
```

### 5. **Dataset-Specific Path Resolution**

Handles different directory structures:
- Yale-BrainMets
- UCSF-GBM
- Lumiere-BrainMets
- UCSD-PTGBM
- OASIS-2
- RHUH-GBM

---

## 📁 Directory Structure

```
temporal-mri-benchmark/
├── anthropic_model.py      # Claude 3.5 runner
├── internvl_model.py        # InternVL 3/3.5 runner
├── qwen3_hf_local.py        # Qwen3-VL runner
├── paligemma_model.py       # PaliGemma/MedGemma runner
├── glm_model.py             # GLM-4V/5V runner
├── gemini_model.py          # Gemini 2.5/3 runner
├── benchmark_samples.json   # Sample input data
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔧 Model-Specific Notes

### **Anthropic Claude**
- Uses vision API with base64 image encoding
- Async processing with bounded concurrency
- Recommended concurrency: 4

### **InternVL**
- Requires dynamic image patching
- Uses 8-bit quantization for memory efficiency
- Processes images as pixel tensors
- May require A100 GPU (40GB+ VRAM)

### **Qwen3-VL**
- Uses `qwen_vl_utils` for vision processing
- 4-bit quantization supported
- Saves temporary grid images to `/tmp/`
- Auto-cleanup after processing

### **PaliGemma/MedGemma**
- Uses Gemma3Processor chat templates
- Requires HuggingFace token for gated models
- Images passed as nested lists
- 4-bit quantization available

### **GLM (ZhipuAI)**
- OpenAI-compatible API
- Base64 image encoding
- Async with Semaphore concurrency control

---

## 📊 Expected Performance

### Runtime (UCSF-GBM Subset, 1,192 samples)

| Model | Hardware | Time | Cost |
|-------|----------|------|------|
| GPT-4o | API | ~2 hours | ~$50 |
| Claude 3.5 | API | ~2 hours | ~$40 |
| Gemini 2.5 Pro | API | ~3 hours | ~$30 |
| InternVL3-8B | A100 40GB | ~6 hours | Free |
| Qwen3-VL-8B | A100 40GB | ~5 hours | Free |
| MedGemma-4B | A100 40GB | ~4 hours | Free |

---

## 🐛 Troubleshooting

### **CUDA Out of Memory**

For InternVL/Qwen3 on limited GPUs:
```bash
# Reduce batch size or use 4-bit quantization
python internvl_model.py --quantize 4bit ...
```

### **API Rate Limits**

```bash
# Reduce concurrency
python anthropic_model.py --concurrency 2 ...
```

### **Image Not Found**

Check path resolution in `resolve_image_path()` function. Each dataset has specific directory structure.

### **Invalid JSON Output**

Some models may not always return valid JSON. The `valid_json` flag tracks this:
```json
{"qa_id": "...", "valid_json": false, "raw_text": "..."}
```

---

## 📖 Citation

If you use this code or benchmark, please cite:

```bibtex
@inproceedings{temporal_mri_benchmark_2025,
  title={Time-Aware Multi-View MRI Benchmark for Temporal Reasoning in Longitudinal Neuroimaging},
  booktitle={MICCAI},
  year={2025}
}
```

---

## 📄 License

This code is released under MIT License. The benchmark data follows CC-BY-4.0.

---

## 🔗 Resources

- **Paper**: [Link will be added upon acceptance]
- **Dataset**: [Link will be added upon acceptance]
- **Documentation**: See individual script docstrings

---

## ⚠️ Important Notes

1. **API Costs**: Closed-source models incur API costs (~$30-50 per 1,000 samples)
2. **GPU Requirements**: Local models require A100 or similar (40GB+ VRAM)
3. **Data Privacy**: Ensure compliance when using API services with medical data
4. **Evaluation**: Results are for research purposes only, not clinical use

---

## 📧 Contact

For questions about the benchmark or code, please open an issue in this repository.

**Note**: This is an anonymous submission for MICCAI 2025 review. Full author information will be revealed upon acceptance.
