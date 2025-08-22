# data_mining_project

# What Leaderboards Reveal About LLM Evolution

This repository analyzes open leaderboard data to trace the evolution of Large Language Models (LLMs) over time. The main workflow lives in `main.ipynb`, which loads leaderboard datasets, performs exploratory analysis, identifies structural shifts, clusters capability profiles, and builds simple predictive models with explainability.

---

## 📦 Contents

```
.
├── main.ipynb               # End-to-end analysis notebook
├── data/                    # Place input CSVs here (see below)
└── README.md                # This file
```

**Expected input files** (as referenced in the notebook):
- - `llm_models.csv`
- `open-llm-leaderboard.csv`

**Example output files** (produced by the notebook):
- - `top5_benchmarks_trend_forecast.png`

---

## 🛠️ Setup

Create a fresh environment (conda recommended) and install dependencies:

```bash
conda create -n llm-evo python=3.10 -y
conda activate llm-evo
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, you can create one with the following minimal set (detected from the notebook):

```txt
python>=3.10
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
ruptures
shap
requests
psutil
huggingface_hub
python-dateutil
scipy
jupyter
```

---

## 🚀 How to Run

1. **Prepare data**: Download or export the leaderboard CSVs and place them under `data/` (or update file paths in the notebook to match your location). Typical filenames used:
   - llm_models.csv
- open-llm-leaderboard.csv

2. **Launch Jupyter**:
   ```bash
   jupyter notebook main.ipynb
   ```

3. **Execute cells top-to-bottom**. The notebook will:
   - Load and clean leaderboard data
   - Compute longitudinal trends across core benchmarks (e.g., instruction-following, reasoning, math)
   - Perform **k-means clustering** and **PCA** on benchmark scores to reveal capability families
   - Run **change-point detection** (via `ruptures`) to identify structural breaks in performance timelines
   - Train lightweight classifiers (e.g., `XGBoostClassifier`) on metadata to predict model creators
   - Use **SHAP** to explain model predictions and identify influential metadata features
   - Save key figures (e.g., `top5_benchmarks_trend_forecast.png`) into `results/`

> Notes:
> - Some steps may require adjusting file paths.
> - If additional APIs are used (e.g., `huggingface_hub`), configure credentials as needed.

---

## 📊 Key Analyses (as organized in the notebook)

- **Market & Release Dynamics**: How the number of models and submissions has changed since 2024–2025.
- **Benchmark Trajectories**: Domain-level progress across instruction-following and reasoning-heavy tasks.
- **Clustering & PCA**: Capability profiles separating early, general-purpose, and reasoning-focused models.
- **Change-Point Detection**: Structural breaks in late-2024 indicating punctuated progress.
- **Predictive Modeling + SHAP**: Metadata-based creator classification and feature attribution.

---

## 🔧 Customization

- Swap in your own CSVs under `data/` and update the paths in the “Load Data” cell.
- Tune k-means `n_clusters`, PCA components, or XGBoost hyperparameters in their respective cells.
- Extend the pipeline to include forecasting or deployment cost analyses.

---

## 📁 Reproducibility Tips

- Fix random seeds for clustering and model training.
- Log software versions:
  ```python
  import sys, numpy, pandas, sklearn, xgboost
  print(sys.version); print(numpy.__version__, pandas.__version__, sklearn.__version__, xgboost.__version__)
  ```
- Store generated artifacts in `results/` for consistent references.

---

## 🙌 Acknowledgments

- Open LLM Leaderboard data and related community benchmarks.
- Libraries detected in this project: datetime, dateutil, huggingface_hub, math, matplotlib, numpy, os, pandas, psutil, re, requests, ruptures, scikit-learn, scipy, seaborn, shap, sklearn, time, torch, transformers, warnings, xgboost.

---

## 📜 License

Specify your license (e.g., MIT).

---

*Generated on 2025-08-22.*
