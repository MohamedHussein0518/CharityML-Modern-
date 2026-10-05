# 🎗️ Finding Donors for CharityML — Intelligent Donor Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%20REST%20API-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Reflex](https://img.shields.io/badge/Frontend-Reflex%20(Pure%20Python%20UI)-4F46E5?logo=react&logoColor=white)](https://reflex.dev/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade Machine Learning system and modern interactive intelligence dashboard designed for **CharityML** to identify high-income individuals ($> \$50,000$) who are most likely to donate, maximizing outreach return-on-investment (ROI) through precision-oriented predictive modeling.

---

## 📌 1. Project Concept (فكرة المشروع)

CharityML is a non-profit organization that relies on philanthropic donations to fund its community initiatives. Empirical research shows that individuals earning **more than \$50,000 annually** are significantly more likely to donate.

- **The Business Dilemma:** Sending direct marketing mailers to everyone in a census database is cost-prohibitive. Contacting someone who earns $\le \$50\text{K}$ results in wasted campaign budget with near-zero return.
- **The Machine Learning Solution:** Predict whether an individual earns $> \$50\text{K}$ using public demographic and census attributes (Age, Education, Marital Status, Occupation, Capital Gains/Losses, Hours worked per week).
- **Metric Priority ($F_{0.5}$ Score):** In charity fundraising, **Precision is more critical than Recall** ($\beta = 0.5$). We prioritize minimizing False Positives (wasted outreach costs) while accurately capturing true high-probability donors.

$$\beta = 0.5 \implies F_{0.5} = (1 + 0.5^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(0.5^2 \cdot \text{Precision}) + \text{Recall}}$$

---

## ⚙️ 2. What the Platform Does (بيعمل إيه)

1. **🎯 Donor Scoring Engine (Live Single Profiler):**
   - Allows campaign managers to input demographic and economic parameters for any candidate.
   - Provides 1-click test presets: *Executive Profile ($>50\text{K}$)* and *Entry-Level Profile ($\le 50\text{K}$)*.
   - Outputs an instant **Probability Gauge (%)**, **Donor Qualification Badge**, and **Tailored Fundraising Outreach Strategy**.

2. **📊 Model Benchmarking & GridSearch Tuning:**
   - Displays real-time comparative benchmarks across **5 supervised algorithms** evaluated in the project (Gradient Boosting, AdaBoost, Random Forest, Logistic Regression, Extra Trees).
   - Highlights hyperparameter tuning results (`n_estimators=150`, `learning_rate=0.1`, $5$-fold CV $F_{0.5} = 0.7494$).

3. **⚖️ Feature Importance & Feature Selection (Questions 6, 7 & 8):**
   - Interactive dual-bar visualization showing **Individual Feature Weight** and **Cumulative Predictive Weight** for the top 5 features (accounting for **90.6%** of model information).
   - In-depth trade-off analysis comparing the **Full Model (103 features)** vs. **Reduced Model (5 features)**.

4. **📈 Enterprise Dashboard UI:**
   - Calm, minimalist, modern corporate aesthetic (inspired by Stripe, Linear, and Vercel) with high-contrast typography and real-time backend health status.

---

## 🏗️ 3. How It's Built — Architecture & Engineering Decisions

```
┌────────────────────────────────────────────────────────┐
│             Client Web Browser (Port 3000)             │
│            Reflex Modern Corporate Dashboard           │
└───────────────────────────▲────────────────────────────┘
                            │
              HTTP / JSON REST API Calls (CORS)
                            │
┌───────────────────────────▼────────────────────────────┐
│              Flask REST API (Port 5000)                │
│    /api/predict | /api/metrics | /api/features | /stats │
└───────────────────────────▲────────────────────────────┘
                            │
                  Inference Pipeline
                            │
┌───────────────────────────▼────────────────────────────┐
│      Trained Scikit-Learn Model Bundle (.joblib)       │
│  - Tuned GradientBoostingClassifier                    │
│  - Log(x+1) Skew Transform on Capital Gain / Loss      │
│  - MinMaxScaler on Continuous Numeric Features         │
│  - 103-Dimension One-Hot Feature Alignment             │
└────────────────────────────────────────────────────────┘
```

### Engineering Decisions:
1. **Decoupled Architecture (Flask + Reflex):**
   - Keeping the ML inference engine inside **Flask** allows the model to serve as a standalone microservice that can be scaled independently or consumed by external APIs.
   - **Reflex** compiles pure Python code into a high-performance Next.js/React frontend without requiring manual React/TypeScript boilerplate.
2. **Deterministic Preprocessing Pipeline:**
   - Preprocessing steps (`log(x + 1)`, `MinMaxScaler`, and one-hot alignment across 103 columns) are serialized into a single `charityml_model.joblib` bundle to prevent data leakage or column-mismatch during runtime.
3. **Resilient Fallback Engine:**
   - The Reflex dashboard features a calibrated offline fallback rule engine that guarantees zero UI downtime even if the Flask backend server is restarting.

---

## 📁 4. Project Structure (هيكل المشروع)

```
P2/
├── finding_donors/
│   ├── census.csv                              # 1994 U.S. Census Dataset (45,222 records)
│   ├── finding_donors.ipynb                    # Complete project Jupyter Notebook
│   ├── visuals.py                              # Matplotlib evaluation & distribution functions
│   ├── Q8_Report.md                            # Comprehensive Question 8 technical report
│   │
│   └── charityml_platform/                     # Production Web Application
│       ├── backend/
│       │   ├── app.py                          # Flask REST API endpoints
│       │   ├── train_and_save_model.py         # Model training & joblib serialization
│       │   ├── requirements.txt                # Flask, scikit-learn, pandas dependencies
│       │   └── model/
│       │       └── charityml_model.joblib      # Serialized ML model & scaler artifacts
│       │
│       ├── frontend/
│       │   ├── rxconfig.py                     # Reflex application configuration
│       │   ├── requirements.txt                # Reflex, requests, httpx dependencies
│       │   └── charityml_dashboard/
│       │       ├── __init__.py
│       │       └── charityml_dashboard.py      # Reflex reactive frontend application
│       │
│       ├── run_backend.bat                     # 1-Click launcher for Flask Backend
│       ├── run_frontend.bat                    # 1-Click launcher for Reflex Dashboard
│       └── README.md                           # Platform quickstart guide
│
└── README.md                                   # Root project documentation
```

---

## 🚀 5. How to Run It (طريقة التشغيل)

### ⚠️ Prerequisites (متطلبات أساسية هامة جداً):
1. **Python 3.10+** (tested and verified on Python 3.12).
2. **Node.js (v22.0.0 or higher / LTS):**  
   Reflex compiles Python code into a Next.js/React frontend in the background and **strictly requires Node.js**.  
   - Download & install from [nodejs.org](https://nodejs.org) or via PowerShell:
     ```powershell
     winget install OpenJS.NodeJS.LTS
     ```
   - Ensure Node.js is recognized in your terminal:
     ```powershell
     node -v
     ```
3. **Two Active Terminals (2 Terminals):**  
   Because the system uses a decoupled client-server architecture, **you must run the Backend and Frontend concurrently in two separate terminal windows**.

---

### Step-by-Step Instructions:

#### 🖥️ Terminal 1: Launch Flask Backend
```powershell
cd c:\Users\Win\OneDrive\Desktop\DEPI\project\P2\finding_donors\charityml_platform\backend
pip install -r requirements.txt
python app.py
```
> ✅ Output confirms: `Starting CharityML Flask Backend on http://127.0.0.1:5000`

---

#### 🖥️ Terminal 2: Launch Reflex Dashboard
```powershell
cd c:\Users\Win\OneDrive\Desktop\DEPI\project\P2\finding_donors\charityml_platform\frontend
pip install -r requirements.txt
reflex run
```
> ✅ Once compilation hits 100%, open your browser at:  
> 👉 **`http://localhost:3000`**

---

### 🟢 1-Click Windows Alternative:
You can also simply navigate to `charityml_platform/` and double-click:
1. **`run_backend.bat`** (Starts Flask on port 5000)
2. **`run_frontend.bat`** (Starts Reflex on port 3000)

---

## 👥 6. Team & My Role

*(Order below is arbitrary — not ranked by importance.)*

| Name | Role | GitHub Profile |
| :--- | :--- | :--- |
| **Mohamed Hussein — Team Leader** | | [![GitHub](https://img.shields.io/badge/GitHub-MohamedHussein0518-181717?logo=github)](https://github.com/MohamedHussein0518) |
| **Anas Sayed** | | [![GitHub](https://img.shields.io/badge/GitHub-Riplinux-181717?logo=github)](https://github.com/Riplinux) |
| **Hassan Ali** | | [![GitHub](https://img.shields.io/badge/GitHub-7assan--Ali-181717?logo=github)](https://github.com/7assan-Ali) |
| **Ahmed Rabie** | | [![GitHub](https://img.shields.io/badge/GitHub-ahmedrabiem-181717?logo=github)](https://github.com/ahmedrabiem) |
| **Heba Ramadan** | | [![GitHub](https://img.shields.io/badge/GitHub-hebaramadan1-181717?logo=github)](https://github.com/hebaramadan1) |
| **Maha Khaled** | | [![GitHub](https://img.shields.io/badge/GitHub-Maha--123--dot-181717?logo=github)](https://github.com/Maha-123-dot) |

---

### 🎓 Under the Supervision of:
**George Samuel — Instructor** ([@gsamuei](https://github.com/gsamueil))  
*Digital Egypt Pioneers Initiative (DEPI) — Machine Learning Track*
