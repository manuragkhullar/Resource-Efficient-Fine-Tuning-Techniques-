# **CIS5300 Final Project: From BART to Edge**  

---

## **Description**  

This project focuses on fine-tuning large language models (LLMs) for **Chinese-to-English translation** in the **education domain**. We compare two state-of-the-art models—**mBART** and **M2M100**—using various fine-tuning techniques such as **LoRA** and **Layer Freezing**. The goal is to identify the most effective and resource-efficient approach for achieving high-quality translations tailored to this domain.

---

## **Project Structure**  

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

## **Usage**  

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

## **Dependencies**  

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

---

## **Results**  

The notebook includes detailed results with tables, figures, and analysis. Key findings include:  

- **BLEU scores** for all models and fine-tuning methods.  
- **Error analysis** to identify common translation challenges.  
- **Resource usage** comparison (GPU efficiency across methods).  

The project concludes with recommendations for achieving the best balance between **accuracy** and **efficiency** for Chinese-to-English translation tasks.  