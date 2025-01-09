# Resource-Efficient Fine-Tuning for Machine Translation on Edge Devices

This repository contains the implementation and findings from our project. The primary focus is on enabling high-quality Chinese-to-English translations using constrained hardware environments, such as edge devices or personal computers.

## Project Overview
Pre-trained machine translation models, while powerful, require significant computational resources. This project investigates techniques to fine-tune such models for domain-specific tasks with limited resources. The approaches evaluated include:
- **Traditional Fine-Tuning**: Best translation performance but resource-intensive.
- **LoRA (Low-Rank Adaptation)**: Memory-efficient but with moderate performance.
- **Layer Freezing**: Balanced trade-off between performance and efficiency.

## Key Highlights
- Models used: **mBART**, **M2M100**, **Marian MT**, and **Chinese-Alpaca**.
- Dataset: **UM-Corpus** with a focus on **Science** and **Education** domains.
- Metrics: Translation quality measured using **BLEU scores**.

## Problem Statement
The project addresses the challenge of fine-tuning large language models for machine translation tasks within resource-constrained environments. The goal is to:
- Minimize dependence on large servers.
- Enable edge devices to perform domain-specific, high-quality translations.

## Methods and Results
### Fine-Tuning Techniques
1. **Traditional Fine-Tuning**: Full parameter updates for the best BLEU scores.
2. **LoRA**: Only updates LoRA matrices, achieving the most efficient memory usage (~2GB).
3. **Layer Freezing**: Freezes a portion of model layers to balance efficiency and accuracy.

### Evaluation
- **Best BLEU Scores**: Achieved using mBART with traditional fine-tuning.
- **Efficiency**: LoRA demonstrated the lowest resource usage, ideal for edge scenarios.

| Method            | GPU Usage  | BLEU Score (Avg) | Training Time |
|--------------------|------------|------------------|---------------|
| Traditional Fine-Tuning | High (~7GB) | Best            | Long          |
| LoRA              | Low (~2GB) | Moderate         | Short         |
| Layer Freezing    | Medium (~4.5GB) | Balanced      | Moderate      |

## Repository Structure
- `data/`: Dataset files (UM-Corpus subset for Science and Education domains).
- `scripts/`: Code for fine-tuning, LoRA implementation, and evaluation.
- `docs/`: Documentation and presentation slides.

##  Setup and Usage
### Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers

### Steps
1. Clone the repository:
   ```bash

