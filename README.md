# Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether a telecom
customer is likely to **churn** (cancel their subscription), based on the
well-known **Telco Customer Churn** dataset. The project covers data
exploration, preprocessing, model training/comparison, and a deployable
**Streamlit** web app where you can enter a customer's details and get a
live churn prediction.

This follows the same dataset and general workflow popularized by
Siddhardhan's Telco Customer Churn YouTube project, restructured here as a
clean, reproducible, production-style repository.

---

## What this project does

1. **Explores** the Telco Customer Churn dataset (demographics, account info,
   services subscribed, billing details) to understand churn drivers.
2. **Cleans & preprocesses** the data — handling missing values, encoding
   categorical features, and balancing classes with SMOTE.
3. **Trains and compares three models** (Decision Tree, Random Forest,
   XGBoost) to find the best-performing churn classifier.
4. **Saves the trained model** (plus the encoders and feature order) with
   `joblib` so the exact same preprocessing can be reused elsewhere.
5. **Serves predictions** through an interactive **Streamlit** web app, where
   a user can input a customer's profile and instantly see whether they're
   predicted to churn, along with a churn probability score and the model's
   feature importances.

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   
├── notebooks/
│   └── 01_churn_prediction.ipynb              # EDA, preprocessing, model training & evaluation
├── models/
│   └── churn_model.joblib                     # created after you run the notebook
├── app/
│   └── streamlit_app.py                       # Streamlit web app for live predictions
├── requirements.txt                           # pinned Python dependencies
├── README.md
└── .gitignore
```

---

## Tech Stack

| Purpose                     | Tool / Library            |
|------------------------------|---------------------------|
| Language                    | Python 3.10 – 3.12         |
| Data manipulation            | pandas, numpy              |
| Machine learning              | scikit-learn               |
| Gradient boosting model      | xgboost                    |
| Handling class imbalance     | imbalanced-learn (SMOTE)   |
| Model persistence            | joblib                     |
| Web app / UI                 | Streamlit                  |
| Visualization (in notebook)  | matplotlib, seaborn        |
| Notebook environment         | Jupyter                    |

All versions are pinned in `requirements.txt` and have been tested together
end-to-end (notebook → saved model → app).

---

## About the Dataset

The **Telco Customer Churn** dataset contains ~7,000 customer records from a
fictional telecom company, with features such as:

- **Demographics**: gender, senior citizen status, partner, dependents
- **Account info**: tenure, contract type, payment method, billing
- **Services**: phone service, internet service, online security, tech
  support, streaming TV/movies, etc.
- **Target**: `Churn` — whether the customer left within the last month

### Downloading the dataset

1. Go to the dataset page on Kaggle:
   [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Click **Download** (you'll need a free Kaggle account, and to be signed in).
3. Unzip the download if needed, and locate the file named
   `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
4. Move that file into this project's `data/` folder, so the final path is:
   ```
   data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment (recommended)

This keeps the project's dependencies isolated from other Python projects
on your machine.

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked if you see `(venv)` at the start of your terminal
prompt.

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs pinned, tested versions of pandas, scikit-learn, xgboost,
imbalanced-learn, streamlit, joblib, matplotlib, seaborn, and Jupyter.

### 4. Add the dataset

Follow the [Downloading the dataset](#downloading-the-dataset) steps above
so that `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` exists.

### 5. Train the model (run the notebook)

Launch Jupyter:

```bash
jupyter notebook notebooks/01_churn_prediction.ipynb
```

Then, from the notebook's menu, run **all cells** (`Kernel → Restart & Run All`,
or `Run → Run All Cells` depending on your Jupyter version). The notebook will:

- Load and explore the dataset (EDA: distributions, correlations, etc.)
- Clean `TotalCharges` and encode categorical columns
- Balance the training data with SMOTE
- Train and evaluate **Decision Tree**, **Random Forest**, and **XGBoost**
- Pick the best model by F1 score
- Save everything the app needs — the model, encoders, scaler, and feature
  order — into a single file:
  ```
  models/churn_model.joblib
  ```

You only need to do this once (or again if you want to retrain on updated
data).

### 6. Run the Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

This starts a local server and should automatically open the app in your
browser at `http://localhost:8501`. If it doesn't open automatically, copy
that URL into your browser manually.

In the app, fill in a customer's details (contract type, tenure, monthly
charges, services subscribed, etc.) and click **Predict Churn** to see:

- The prediction (**Churn** or **No Churn**)
- A churn probability / confidence score
- A simple risk-level readout (low / moderate / high)
- The model's overall feature importances

To stop the app, go back to your terminal and press `Ctrl + C`.

---

## Optional: Deploy on Streamlit Community Cloud

Streamlit Community Cloud lets you host this app for free and share it with
a public link. High level steps (see
[Streamlit's official docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)
for the most up-to-date walkthrough):

1. **Push your project to GitHub.**
   Make sure the repo includes `app/streamlit_app.py`, `requirements.txt`,
   and a trained `models/churn_model.joblib` (Community Cloud can't run your
   notebook for you — train locally first and commit the resulting model
   file, or adjust `.gitignore` if you want it tracked).

2. **Create a Streamlit Community Cloud account.**
   Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.

3. **Create a new app.**
   Click **"Create app"**, choose **"Yup, I have an app"**, then select:
   - Your GitHub repository
   - The branch (e.g. `main`)
   - The main file path: `app/streamlit_app.py`

4. **(Optional) Set the Python version and secrets.**
   In "Advanced settings," you can pick a specific Python version and add
   any environment variables/secrets your app needs (this project doesn't
   require any secrets by default).

5. **Click "Deploy."**
   Streamlit Community Cloud installs your `requirements.txt` and launches
   the app — most deployments finish within a few minutes. After that,
   pushing new commits to your repo will automatically update the live app.

> **Note:** the free tier has resource limits (around 1 GB memory) and
> apps can go to sleep after periods of inactivity, waking up again on the
> next visit. That's generally fine for a project like this.

---

## Troubleshooting

| Problem | Likely fix |
|---|---|
| `ModuleNotFoundError` when running the notebook or app | Make sure your virtual environment is activated and you ran `pip install -r requirements.txt` |
| App shows "Could not find a trained model" | Run the notebook first (Step 5 above) to generate `models/churn_model.joblib` |
| `FileNotFoundError` for the CSV in the notebook | Double-check the dataset is at exactly `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| `streamlit: command not found` | Activate your virtual environment, or try `python -m streamlit run app/streamlit_app.py` |
| Prediction errors about an "unexpected value" for a field | The dropdowns in the app are built from the categories the model was trained on — re-run the notebook if you've changed the dataset |

---

## License

This project is intended for educational and portfolio purposes. The
dataset is publicly available on Kaggle under its own license terms.
