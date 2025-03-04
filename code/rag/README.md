# **CS 145 Mini Project 3 - README**


## **🚀 Overview**
This project implements a **Retrieval-Augmented Generation (RAG) system** for the **Meta KDD Cup 2024 CRAG challenge**. The system integrates **vLLM**, **Meta-Llama 3.2-3B-Instruct**, and **Google Cloud VM** to efficiently answer complex questions using an LLM-powered pipeline.

---

## **📁 Project Structure**

```
CS_145_Mini_Project_3/
│── rag/                     # Codebase for RAG system
│   ├── generate.py          # Generates model predictions
│   ├── evaluate.py          # Evaluates model performance
│   ├── evaluation_model.py  # Evaluates model performance
│   ├── rag_baseline.py      # RAG model implementation
│   ├── vanilla_baseline.py  # Vanilla baseline implementation
│── data/                    # Contains dataset files
│   ├── crag_task_1_dev_v4_release.jsonl.bz2
│── output/                  # Stores generated predictions & evaluation results
│── requirements.txt         # Python dependencies
│── README.md                # This file
```

---

## **🔹 Prerequisites**
### **1️⃣ Google Cloud VM Setup**
1. Go to [Google Cloud Compute Engine](https://console.cloud.google.com/compute/instances).
2. Create a new VM instance with:
   - **Machine Type:** `n1-standard-4` 
   - **GPU:** Tesla T4
   - **Boot Disk:** Deep Learning VM (CUDA 12.1)
   - **Firewall:** Allow HTTP & HTTPS
3. **Connect via SSH:**
   ```bash
   ssh -i ~/.ssh/my-gcloud-key your-username@YOUR_VM_EXTERNAL_IP
   ```

### **2️⃣ Install Dependencies**
1. **Clone the repository & navigate to the project folder:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
   cd CS_145_Mini_Project_3
   ```
2. **Install dependencies:**
   ```bash
   pip install -r rag/requirements.txt
   pip install --upgrade openai 
   ```

### **3️⃣ Authenticate with Hugging Face**
1. **Log in to Hugging Face:**
   ```bash
   huggingface-cli login
   ```
2. **Verify model access:**
   ```bash
   python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"
   ```
   ✅ No errors mean authentication is successful.

---

## **🚀 Running the Model**
### **1️⃣ Start vLLM Server** (Keep this running in a separate terminal)
```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --gpu_memory_utilization=0.85 \
    --tensor_parallel_size=1 \
    --dtype="half" \
    --max_model_len 8192 \
    --port=8088 \
    --enforce-eager
```
✅ **Leave this running** while making API calls.

### **2️⃣ Generate Model Predictions**
📌 Open a **new terminal** and run:
```bash
cd ~/CS_145_Mini_Project_3/rag
python generate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --split 1 \
    --model_name "rag_baseline" \
    --llm_name "meta-llama/Llama-3.2-3B-Instruct" \
    --is_server \
    --vllm_server "http://localhost:8088/v1"
```
✅ This generates responses and saves them in `output/`.

### **3️⃣ Evaluate the Model**
```bash
python evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name "rag_baseline" \
    --llm_name "meta-llama/Llama-3.2-3B-Instruct" \
    --is_server \
    --vllm_server "http://localhost:8088/v1" \
    --max_retries 10
```
✅ This computes accuracy, hallucination rate, and missing rate.

---

## **📂 Output Files**
After execution, the output files will be located in:
```
CS_145_Mini_Project_3/output/rag_baseline/meta-llama-3.2-3B-Instruct/
│── predictions.json          # Generated answers
│── evaluation_results.json   # Accuracy and performance metrics
```
✅ **Download these files for submission.**


