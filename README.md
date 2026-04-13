# Neurax Embeddings

>A small Streamlit dashboard for generating and visualizing word embeddings using custom Word2Vec training or pre-trained GloVe (Twitter 25d).

---

## Features
- Train a custom Word2Vec model from a text corpus (configurable vector size, window, min count, epochs)
- Load pre-trained GloVe (twitter-25) vectors and visualize selected words
- Dimensionality reduction: PCA (linear) or t-SNE (non-linear), 2D/3D views
- Interactive Plotly visualizations and semantic nearest-neighbor lookup
- Lightweight preprocessing pipeline with lemmatization and stopword removal (regex tokenizer fallback included)

---

## Prerequisites
- Python 3.8+
- Git (for version control / pushing to remote)

## Quick Setup (Windows PowerShell)

```powershell
# 1) Create & activate a virtual environment (if you don't have one)
python -m venv .venv
& .venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install -r requirement.txt

# 3) Run the app
streamlit run app.py
```

Notes:
- `matplotlib` is optional but recommended if you want gradient-styled DataFrame cells. It's listed in `requirement.txt`.
- If you prefer to use full NLTK tokenizers, ensure NLTK data is available:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

The app ships with a regex-based tokenizer fallback to avoid runtime failures when NLTK tokenizers are not present.

---

## Usage
1. Pick a model mode in the sidebar: `Custom Word2Vec` or `Pre-trained GloVe (Twitter 25d)`.
2. For `Custom Word2Vec`: provide a corpus (sample text or upload `.txt`) and tune hyperparameters.
3. For `Pre-trained GloVe`: provide comma-separated words to visualize.
4. Choose dimensionality reduction (`PCA` or `t-SNE`) and view dimension (2D/3D).
5. Click **Process & Visualize** to train/load, project vectors, and display the interactive plot.
6. Use the **Semantic Similarity Engine** to query a word and view nearest neighbors.

---

## Files
- `app.py` — main Streamlit app
- `requirement.txt` — Python dependencies
- `.gitignore` — recommended ignores (virtualenv, caches, datasets, models)

---

## Git / Publishing
- Keep your virtualenv out of git by ensuring `.venv/` or `venv/` is in `.gitignore`.
- Commit only source and dependency files (e.g., `requirement.txt`).
- To push to GitHub:

```powershell
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

For large model files use Git LFS or external storage (S3, Hugging Face, etc.).

---

## Troubleshooting
- LookupError for `punkt_tab` or `punkt`: install NLTK data (see notes above) or use the app's fallback tokenizer.
- `Styler.background_gradient` error: install `matplotlib`.
- If you accidentally committed `.venv`:

```powershell
git rm -r --cached .venv
git commit -m "Stop tracking virtualenv"
git push
```

---

## Development notes
- Streamlit API changes: `use_container_width` has been replaced by `width='stretch'`/`width='content'`. This repo uses `width='stretch'`.
- The app caches NLTK model downloads using Streamlit's caching helpers to avoid repeated downloads.

---

## Contributing
Feel free to open issues or PRs. For significant changes, add tests and update this README accordingly.

## License
Choose a license for your project (MIT, Apache-2.0, etc.) and add a `LICENSE` file if you plan to publish.
