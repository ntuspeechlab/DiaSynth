DiaSynth: Synthetic Dialogue Generation Framework [NAACL'25]

DiaSynth is a synthetic dialogue generation framework designed for low-resource dialogue applications. By leveraging Large Language Models (LLMs) and Chain-of-Thought (CoT) reasoning, DiaSynth creates high-quality, persona-driven dialogues for various domains.

🚀 Key Features:
	•	Scalable Synthetic Dialogue Generation 
	•	Persona-Conditioned & Topic-Specific Conversations 
	•	Multi-Domain Dialogue Generation (SAMSum, DialogSum, etc.) 
	•	Pre-trained LLM Support (Phi-3, LLaMA, GPT-4o, etc.)

📥 Installation

1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/DiaSynth.git
cd DiaSynth
```

2️⃣ Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3️⃣ Install llama-cpp-python (for GGUF model)

CMAKE_ARGS="-DGGML_CUDA=on -DCUDAToolkit_ROOT=/usr/local/cuda-11.8" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

4️⃣ Download & Configure Model Files

Download the Phi-3 GGUF Model
	•	Phi-3-mini-128k-instruct (4GB)
	•	Place it inside the folder:

```bash
mkdir -p models/phi3
mv model.gguf models/phi3/
```

Download Tokenizer
	•	Download from microsoft/Phi-3-mini-128k-instruct
	•	Save it in:

```bash
mkdir -p models/phi3/tokenizer
mv tokenizer.json models/phi3/tokenizer/
```

🚀 Usage

🔹 Generate Synthetic Dialogues

Run the following commands to generate synthetic dialogues using Phi-3:

```bash
python3.10 -m diasynth.main --topics_file_path "topics.txt" --n_sub_topics 6 --stage="scratch" \
    --csv_path CSV_PATH --dialogue_base "dialoguesum"
```

🔹 Modify Topics
	•	Update topics.txt to include your custom topics for dialogue generation.

🎯 Features

✅ Flexible Topic Expansion – Generate subtopics for broader coverage. <br>
✅ Persona-Based Conversations – Customize dialogues with realistic characters. <br>
✅ Multi-Turn Dialogue Generation – Supports structured and free-flowing interactions. <br>
✅ Scalable Data Synthesis – Generate thousands of dialogues efficiently. <br>
✅ Pretrained Model Fine-tuning – Improve performance with synthetic data.

📊 Example Outputs
```
Example: Generated Dialogue (Healthcare Domain)

Doctor: Good morning! How can I assist you today?
Patient: I've been feeling dizzy and fatigued for the past few days.
Doctor: I see. Have you experienced any headaches or blurred vision?
Patient: Yes, occasionally. It gets worse in the afternoon.
Doctor: Based on your symptoms, we might need to check your blood pressure.
```

<!-- 📌 Citation

If you use DiaSynth in your research, please cite:

@inproceedings{suresh2025diasynth,
  title={DiaSynth: Synthetic Dialogue Generation Framework for Low-Resource Dialogue Applications},
  author={Suresh, Sathya Krishnan and Mengjun, Wu and Pranav, Tushar and Chng, Eng Siong},
  booktitle={Proceedings of the NAACL},
  year={2025}
} -->