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


### **Project Structure**  

We provide two main notebooks: one for the **education corpus domain** and one for the **science domain corpus**.  

1. **Load and Preprocess**  
   - Import required libraries, load data, initialize models, and define evaluation metrics.  

2. **Baseline Evaluation**  
   - Evaluate the original mBART and M2M100 models on a **50,000-sample dataset** to establish baseline BLEU scores.  

3. **Hyperparameter Tuning**  
   - Perform grid search on a smaller **10,000-sample dataset** to identify optimal configurations for each fine-tuning technique.  

4. **Fine-Tune mBART**  
   - Fine-tune the mBART model using the best hyperparameters on the full **50,000-sample dataset**.  

5. **Fine-Tune M2M100**  
   - Fine-tune the M2M100 model using the best hyperparameters on the full **50,000-sample dataset**.  

6. **Evaluation and Analysis**  
   - Assess model performance using **BLEU scores**, analyze errors (Word-Level, Structural, Other), and compare GPU resource usage.  

7. **Extension: Quantization and Small Models**  
   - Evaluate a pre-quantized **LLaMA** model and a compact **Chinese-to-English MT model** (e.g., Marian MT) for low-resource scenarios.  

---

### **Usage**  

1. **Set Up Environment**  
   - Ensure required libraries are installed (see Dependencies section).  

   **Recommendation**: Upload the notebooks to **Google Colab** and use an **A100 GPU** for optimal performance.  


2. **Download Data**  
   - Place the dataset in the working directory or Colab content folder. Update the paths to the `.txt` files accordingly.  
   - Provided datasets:  
     - `Bi-Education.txt` for education domain.  
     - `Bi-Science.txt` for science domain.  
   - Feel free to experiment with other domain-specific corpora!  

3. **Run Notebook**  
   - Execute the notebook cells sequentially.  

4. **View Results**  
   - Analyze BLEU scores, error types, GPU usage, and model performance comparisons.  

---

### **Dependencies**  

The following libraries are required:  

- `datasets`  
- `optimum`  
- `auto-gptq`  
- `sentencepiece`  
- `bitsandbytes`  
- `sacremoses`  
- `sacrebleu`  
- `transformers`  
- `peft`  
- `nltk`  
- `tqdm`  
- `pandas`  
- `torch`  

To install all dependencies in your Colab environment, run:  

```bash
!pip install datasets optimum auto-gptq sentencepiece bitsandbytes sacremoses sacrebleu transformers peft nltk tqdm pandas torch
```


