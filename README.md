# ML CI/CD Pipeline — Iris Classification Model

This repository implements an automated **Continuous Integration & Continuous Deployment (CI/CD)** pipeline for an end-to-end Machine Learning workflow using **GitHub Actions** and **GitHub Pages**.

---

## 📌 Repository Structure

```text
.
├── model.py                # ML model training, prediction & evaluation script
├── test_model.py           # Automated test suite using pytest
├── requirements.txt        # Project dependencies (scikit-learn, pytest, numpy)
├── index.html              # Deployment dashboard for GitHub Pages
│
└── .github/
    └── workflows/
        └── cicd.yml        # GitHub Actions CI/CD workflow pipeline
```

---

## 🤖 ML Model & Dataset Overview

- **Machine Learning Model**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Dataset**: Fisher's Iris Flower Dataset (`sklearn.datasets.load_iris`)
  - **Samples**: 150 instances
  - **Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
  - **Target Classes**: 3 (`setosa`, `versicolor`, `virginica`)
- **Model Accuracy**: **100.00%** on test split (threshold &ge; 90%)

---

## 🔄 CI/CD Pipeline Architecture

The pipeline is defined in `.github/workflows/cicd.yml` and enforces strict phase separation:

```text
[ Trigger: Push / PR ]
         │
         ▼
┌─────────────────────────────────┐
│           CI STAGE              │
│ 1. Checkout repository          │
│ 2. Set up Python 3.11           │
│ 3. Install requirements.txt     │
│ 4. Run model.py (train & eval)  │
│ 5. Execute pytest test cases    │
└─────────────────────────────────┘
         │
         │  PASS (needs: ci)
         ▼
┌─────────────────────────────────┐
│           CD STAGE              │
│ 1. Checkout repository          │
│ 2. Configure GitHub Pages       │
│ 3. Upload pages artifact        │
│ 4. Deploy index.html to Pages   │
└─────────────────────────────────┘
         │
         ▼
[ Live GitHub Pages Deployment ]
```

---

## 🧪 Automated Testing (`pytest`)

Automated tests in `test_model.py` validate:
1. **`test_data_loading`**: Dataset dimensions, feature counts, and 3 classes.
2. **`test_model_training`**: Proper model fitting and 50 estimators.
3. **`test_model_predictions`**: Output dimensionality and valid class labels.
4. **`test_model_accuracy_threshold`**: Accuracy &ge; 90%.
5. **`test_index_html_metadata`**: Verification that `index.html` displays Model Name, Dataset, Accuracy, and CI/CD Status.

Run tests locally:
```bash
pytest -v
```

---

## 🚀 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train model & evaluate
python model.py

# Run test cases
pytest -v
```

---

## 🌐 GitHub Pages Deployment Configuration

To enable automated deployment via GitHub Actions:
1. Navigate to the repository **Settings** &rarr; **Pages**.
2. Under **Build and deployment** &rarr; **Source**, select **GitHub Actions**.
3. Push to `main` (or `master`) or run the workflow manually via `Actions` &rarr; `ML CI/CD Pipeline` &rarr; `Run workflow`.
4. The live dashboard will be accessible at:
   ```text
   https://231fa04109-lab.github.io/<repository-name>/
   ```
