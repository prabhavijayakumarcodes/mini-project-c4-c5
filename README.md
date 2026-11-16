# Mini Project (C4 & C5) — Insurance Claim Prediction + M&A RAG Assistant

| Part | Area | Notebook | Weight |
|---|---|---|---|
| A | Machine Learning — predict `insuranceclaim` | [`Part_A_Insurance_Claim_Prediction.ipynb`](Part_A_Insurance_Claim_Prediction.ipynb) | 30% |
| B | LangChain + RAG — Q&A over the M&A Playbook PDF only | [`Part_B_MA_RAG_Assistant.ipynb`](Part_B_MA_RAG_Assistant.ipynb) | 70% |

## What's inside

**Part A — Insurance Claim Prediction**
- Task 1: dataset understanding (head/tail, shape, columns, dtypes, missing values) + answers on numerical/categorical variables and target balance
- Task 2: univariate EDA — age, BMI, charges, insurance claims
- Task 3: leakage-safe preprocessing — stratified train/test split, `StandardScaler` + `OneHotEncoder(region)` inside a scikit-learn `Pipeline`, with the written answer on why transformers are fitted on training data only
- Task 4: Logistic Regression (main, interpretable) + Random Forest (comparison) — predictions, probabilities, accuracy/precision/recall/F1/ROC-AUC, confusion matrices, ROC curves

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.870 | 0.898 | 0.884 | 0.904 |
| Random Forest | 0.951 | 0.944 | 0.975 | 0.959 | 0.992 |

The notebook also answers the 5 extra healthcare questions printed at the end of the brief (page 5), interpreted for a diabetes-prediction model.

**Part B — M&A Knowledge Assistant (RAG)**
- Task 1: `PyPDFLoader` → text cleaning → `RecursiveCharacterTextSplitter`; chunk size/overlap experiment (chunk stats + retrieval hit-rate) and the chunking questions
- Task 2: Google Gemini embeddings, cosine-similarity demo and explanations
- Task 3: FAISS vector store built **only** from the M&A PDF (checked with an assertion) and saved locally
- Task 4: LangChain LCEL pipeline — question → retrieve top-k chunks → prompt → Gemini → answer **plus retrieved sources** (page, chunk id, similarity)
- Task 5: grounded prompt — context only, no hallucination, explicit *"The information is not available in the provided M&A Playbook."*, concise answers with `[p.X]` citations; demonstrated with out-of-scope questions (e.g. the TargetCo case study, which is not in this PDF)

## Project structure

```
├── Part_A_Insurance_Claim_Prediction.ipynb   # executed, outputs included
├── Part_B_MA_RAG_Assistant.ipynb             # run it with your own Gemini API key
├── data/
│   ├── insurance.csv                         # Part A dataset
│   └── MA_Due_Diligence_Playbook.pdf         # Part B knowledge source (the only one)
├── requirements.txt
└── README.md
```

## How to run

### Option 1 — Locally (macOS / Linux)

```bash
git clone https://github.com/prabhavijayakumarcodes/mini-project-c4-c5.git
cd mini-project-c4-c5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Part B needs a **Google Gemini API key** (free at <https://aistudio.google.com/apikey>). Export it before starting Jupyter, or paste it when the notebook asks:

```bash
export GOOGLE_API_KEY="your-key-here"
jupyter notebook
```

Open each notebook and choose **Kernel → Restart & Run All**.

On Windows, activate the environment with `.venv\Scripts\activate` and set the key with `set GOOGLE_API_KEY=your-key-here`.

### Option 2 — Google Colab

1. Open <https://colab.research.google.com> → **File → Open notebook → GitHub** and paste this repository's URL.
2. In the first cell, run:
   ```python
   !git clone https://github.com/prabhavijayakumarcodes/mini-project-c4-c5.git
   %cd mini-project-c4-c5
   !pip install -q -r requirements.txt
   ```
3. Run all cells. Part B asks for your Gemini API key (hidden input). Alternatively add it as a Colab secret and set `os.environ["GOOGLE_API_KEY"]`.

### Notes
- If Google renames a model, change `EMBEDDING_MODEL` / `LLM_MODEL` in the setup cell of Part B.
- Part B saves the FAISS index to `faiss_index_ma_playbook/` (ignored by git); it is rebuilt on every run.
- Never commit your API key — the notebook reads it from the environment or a hidden prompt.
