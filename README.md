# 🦖 REX-2: 50M Parameter Story Engine (IQROGUEREX)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

🚀 **REX-2** is a **50M parameter Transformer-based language model**, trained entirely from scratch for **short-form creative storytelling**.
It showcases a complete **end-to-end LLM pipeline**, from architecture design to deployment.

---

## 🔗 Project Links

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/IqRogueRex/REX-2-50M)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/iqroguerex-cpu/REX-2-50M)

---

## 🌐 Live Demo

👉 **Try it here:**
https://rextwo50mbychinmay.streamlit.app/

---

## ✨ Features

### 🧠 Custom Transformer Model

* Decoder-only **GPT-style architecture**
* Built from scratch (no pre-trained weights)
* Lightweight yet effective design

### ✍️ Creative Story Generation

* Generates **short, structured stories**
* Maintains **basic grammar and narrative flow**
* Designed for **child-like storytelling tone**

### ⚙️ Efficient Training Pipeline

* Multi-stage training approach
* Optimized for **limited compute (Tesla T4)**
* Balanced performance vs resource usage

### 🚀 Interactive Deployment

* **Streamlit-based UI**
* Hosted with **Hugging Face integration**
* Easy-to-use interface for real-time generation

### 📉 Optimized Performance

* Final loss: **2.2 – 2.5**
* Improved coherence via fine-tuning

---

## 🧠 Model Architecture

| Component       | Value                    |
| --------------- | ------------------------ |
| Model Type      | Decoder-only Transformer |
| Parameters      | 50.3 Million             |
| Layers          | 6 Transformer Blocks     |
| Attention Heads | 8                        |
| Embedding Size  | 512                      |
| Context Window  | 256 tokens               |
| Tokenizer       | GPT-2 (50,257 vocab)     |

---

## 📈 Training Pipeline

### 🔹 1. Base Training

* Dataset: **TinyStories**
* ~3,000 steps
* Learned basic grammar and structure

### 🔹 2. Stream Polishing

* Streaming-based data exposure
* Reduced overfitting
* Improved generalization

### 🔹 3. Fine-Tuning

* Targeted optimization
* Final loss: **2.2 – 2.5**
* Better narrative coherence

---

## 🧪 Sample Output

```text
Once upon a time, a little robot named Rex found a glowing stone in the forest.
When he touched it, the trees began to whisper secrets of the stars...
```

---

## 🛠️ Tech Stack

* **Model Framework:** PyTorch
* **Frontend/UI:** Streamlit
* **Tokenizer:** GPT-2 Tokenizer
* **Deployment:** Hugging Face + Streamlit Cloud
* **Dataset:** TinyStories

---

## 📂 Project Structure

```
REX-2-50M/
├── app.py              # Streamlit UI + HF integration
├── model.py            # Transformer architecture
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚡ Installation & Local Setup

```bash
# Clone repository
git clone https://github.com/ChinmayChatradamath/REX-2-50M.git

# Navigate to project
cd REX-2-50M

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚠️ Limitations

* Context drift after ~60 tokens
* Occasional character inconsity
* Logical gaps in longer outputs
* Limited world knowledge (trained on small dataset)

---

## 📈 Use Cases

* 🧪 Small-scale LLM research
* ✍️ Creative story generation
* 🎓 Educational demonstrations
* ⚙️ End-to-end ML pipeline showcase

---

## 🌍 Why This Project Matters

Most modern LLMs rely on massive datasets and compute.

**REX-2 proves that:**

* 💡 Meaningful language models can be built at smaller scales
* ⚙️ Strong engineering can compensate for limited resources
* 🧠 Learning-by-building is a powerful approach to AI

---

## 👨‍💻 Author

**Chinmay V Chatradamath**

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🚀 Future Improvements

* Scale to **100M+ parameters**
* Improve long-context coherence
* Add instruction tuning
* Expand dataset diversity
* Build multi-model comparison dashboard

---

⭐ **If you found this interesting, consider giving it a star!**
