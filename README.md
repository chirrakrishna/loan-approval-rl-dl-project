🚀 Loan Decision Optimization: Deep Learning + Offline Reinforcement Learning
📌 Project Overview

This project builds an intelligent loan approval system using:

Supervised Deep Learning (risk prediction)

Offline Reinforcement Learning (profit-maximizing policy)

Dataset: LendingClub Accepted Loans (2007–2018)
Goal: Maximize profitability while reducing loan defaults

✅ Task 1 — EDA & Preprocessing

Notebook: 1_Preprocessing.ipynb

✔ What Was Done

1️⃣ Exploratory Data Analysis

Checked missing values

Studied distribution of key features

Identified target imbalance

Removed non-predictive columns (id, url, title, etc.)

2️⃣ Feature Engineering

loan_to_income = loan_amnt / annual_inc

amt_per_term = loan_amnt / term

3️⃣ Preprocessing Pipeline

SimpleImputer (median / most_frequent)

StandardScaler for numeric scaling

OrdinalEncoder for categorical encoding

Combined using ColumnTransformer

✔ Output

final_preprocessor.pkl

Cleaned dataset for model training

🤖 Task 2 — Deep Learning Model (Supervised Learning)

Notebook: 2_Deep_Learning_Model.ipynb

🎯 Target Definition

0 → Fully Paid

1 → Default / Charged Off

🧱 Model Architecture (PyTorch MLP)

148 → 256 → 128 → 64 → 1

Activations: ReLU

Regularization: Dropout

Loss: BCEWithLogitsLoss

Optimizer: Adam

📊 Results

AUC ≈ 0.99

Best F1-score → tuned threshold

🔎 DL Policy
If predicted_default_probability < threshold:
    Approve Loan
Else:
    Deny Loan

🧠 Task 3 — Offline Reinforcement Learning
Notebook: 3_RL_Environment.ipynb & 4_Offline_RL_Training.ipynb
📌 RL Setup

State (s): Preprocessed feature vector (149 values)
Action (a):

0 → Deny

1 → Approve

Reward (r):

Deny → 0

Approve + Fully Paid → loan_amnt * int_rate

Approve + Default → -loan_amnt

✔ RL Dataset Created

Saved as:

offline_rl_dataset.npz

offline_rl_dataset_fixed.npz

Contains:

states

actions

rewards

next_states

dones

🏋️ RL Training

Offline Q-learning (no environment interaction)

Q-network learns:

Approve if Q(s,1) > Q(s,0)

📈 RL Output

Learned approval policy

Estimated policy value (expected profit of RL decisions)

📊 Task 4 — Analysis & Business Insights
1️⃣ Why DL Metrics (AUC & F1)?

AUC → how well the model separates good borrowers vs risky borrowers

F1 → best balance between identifying defaults & minimizing false approvals

Helps as risk classifier

2️⃣ Why RL Metric = Policy Value?

Measures profit, not accuracy

Answers business question:
“How much money will this approval policy make?”

3️⃣ DL vs RL Decisions

DL denies high-risk applicants

RL approves some high-risk applicants if expected interest > expected loss

RL focuses on maximizing money, not accuracy

4️⃣ Future Improvements

Use advanced Offline RL algorithms (CQL, IQL, BCQ)

Add financial & behavioral data

Improve reward design

Create simulated environment for real-time RL

🧪 How to Run This Project
Install dependencies
pip install -r requirements.txt

Run notebooks in order:

1_Preprocessing.ipynb

2_Deep_Learning_Model.ipynb

3_RL_Environment.ipynb

4_Offline_RL_Training.ipynb

📝 Conclusion

This project builds two complementary systems:

Deep Learning Model → Predicts default risk with high accuracy (AUC ≈ 0.99)

Offline RL Agent → Learns approval decisions that maximize expected financial return

Together, they create a smart and profitable loan approval strategy for fintech applications.
