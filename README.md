# 🧠 Logic & Reasoning QA Finetuner for Ethos ML Challenge

This repository contains the code for a machine learning project designed to solve the **Ethos ML reasoning challenge**.  
It fine-tunes a `roberta-large` model on a multiple-choice question-answering task to predict the correct solution for logic and reasoning problems.

---

## 🚀 Features

- **Model Finetuning:** Leverages the powerful `roberta-large` architecture from Hugging Face’s `transformers` library.  
- **Multiple-Choice QA:** Formulates the reasoning task as a multiple-choice classification problem.  
- **PyTorch-based:** Built using PyTorch for training and inference.  
- **GPU Accelerated:** Automatically uses a CUDA-enabled GPU if available.  
- **Model Saving:** Automatically saves the fine-tuned model weights for later inference.  

---

## 🛠️ Setup

Follow these steps to set up the environment and run the project.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AshAryan12104/ethos_ml
cd ethos_ml
```

---

### 2️⃣ Create a Virtual Environment (Recommended)
It’s good practice to create a virtual environment to manage dependencies.

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate  
# On Windows:
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Authenticate with Hugging Face
To download the `roberta-large` model, you need to authenticate with Hugging Face.

Run the following command and paste your Hugging Face access token when prompted:

```bash
python -m huggingface_hub.commands.cli login
```

---

## 🏃‍♀️ Usage

### 1️⃣ Place Your Data
Ensure your `train.csv` and `test.csv` files are located in the project’s root directory  
—or update the paths in the **CONFIG** section of `main.py`.

### 2️⃣ Run the Script
Execute the main script:

```bash
python main.py
```

This will:
- Load the training and test datasets  
- Fine-tune `roberta-large` for 3 epochs  
- Save the fine-tuned model to a folder named `roberta_finetuned_model`  
- Generate predictions on `test.csv`  
- Save the final predictions to `output.csv`

---

## 🔐 Download Pretrained Model Weights

To run evaluation without retraining, download the pretrained model checkpoints from the link below:

📦 [Download Finetuned Model Weights from Google Drive](https://drive.google.com/drive/folders/1OujZbnsBpx6BOZapGWryvRcDxpbf5VJU?usp=sharing)


### 📂 Where to Place the Files:

After downloading, place them in the root directory.

---

## 🤖 Model

This project uses **RoBERTa-Large**, a Transformer-based language model with **355 million parameters**, fine-tuned for logical reasoning and QA tasks specific to the Ethos ML Challenge.

---

## 📄 License

This project is developed as part of a Hackathon and is intended for academic and demonstration use only.


