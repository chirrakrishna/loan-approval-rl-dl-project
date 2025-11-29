## Loan Decision Optimization using Deep Learning and Offline Reinforcement Learning

Author: Krishna Chirra



Project Overview

This project builds an intelligent loan approval system using two approaches:

Supervised Deep Learning (Classification)

Offline Reinforcement Learning (Decision Optimization)

Dataset: LendingClub Accepted Loans (2007–2018)
Objective: Improve loan approval decisions to maximize profitability and reduce default risk.

Task 1 — Exploratory Data Analysis (EDA) & Preprocessing

Notebook: 1_Preprocessing.ipynb

What was done
1. Exploratory Data Analysis

Identified missing values

Examined distributions of key numeric features

Identified categorical columns

Analyzed target imbalance

Removed non-predictive columns such as id, url, title, etc.

2. Feature Engineering

loan_to_income = loan_amnt / annual_inc

amt_per_term = loan_amnt / term

3. Preprocessing Pipeline

SimpleImputer (median for numeric, most_frequent for categorical)

StandardScaler for numeric scaling

OrdinalEncoder for categorical encoding

Combined using ColumnTransformer

Output artifacts:

final_preprocessor.pkl

Clean processed dataset for modeling

Task 2 — Deep Learning Model (Supervised Learning)

Notebook: 2_Deep_Learning_Model.ipynb

Target Definition

0 → Fully Paid

1 → Default / Charged Off

Model Architecture (MLP – PyTorch)

Input size → 256 → 128 → 64 → 1

Activation: ReLU

Regularization: Dropout

Loss Function: BCEWithLogitsLoss

Optimizer: Adam

Results

AUC Score: ~0.99

F1 Score: optimized using threshold tuning

Policy Derived from DL Model
If predicted_default_probability < threshold:
    Approve loan
Else:
    Deny loan

Task 3 — Offline Reinforcement Learning Agent

Notebooks:

3_RL_Environment.ipynb

4_Offline_RL_Training.ipynb

RL Problem Setup
State (s)

Preprocessed feature vector (149 dimensions)

Action (a)

0 = Deny Loan

1 = Approve Loan

Reward (r)

If action = 0 → reward = 0

If action = 1 and fully paid → reward = loan_amnt * int_rate

If action = 1 and defaulted → reward = -loan_amnt

Offline RL Dataset Created

Saved files:

offline_rl_dataset.npz

offline_rl_dataset_fixed.npz

Contains:

states

actions

rewards

next_states

dones

Training

Implemented Offline Q-Learning

No interaction with environment required

Q-network learns:

Approve if Q(s,1) > Q(s,0)

Outputs

Trained RL Q-network

Learned approval policy

Estimated policy value (expected return of RL decisions)

Task 4 — Final Analysis & Findings
1. Why AUC and F1 for Deep Learning

AUC measures how well the model separates risky vs safe borrowers

F1 balances false approvals and false denials

Suitable for classification-based decision systems

2. Why Estimated Policy Value for RL

Measures profitability, not accuracy

Represents the expected financial return of RL decisions

Aligns directly with business objectives

3. Comparing DL vs RL Decisions

Deep Learning denies most high-risk borrowers

RL may approve a high-risk loan if expected interest > expected loss

RL focuses on long-term profit, not accuracy

4. Future Improvements

Try advanced Offline RL algorithms (CQL, IQL, BCQ)

Expand dataset with financial behavior, banking history, credit utilization

Improve reward shaping

Build simulation environment for online RL testing

How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Run notebooks in the following order

1_Preprocessing.ipynb

2_Deep_Learning_Model.ipynb

3_RL_Environment.ipynb

4_Offline_RL_Training.ipynb

Conclusion

This project demonstrates a complete loan decision framework:

Deep Learning provides high-accuracy default prediction
Offline Reinforcement Learning produces a profit-optimized approval policy

Together, they offer a powerful strategy for financial decision-making in real-world fintech applications.
