# Thiranex Task 4 – Real-World Data Project

## Bank Marketing Dataset Analysis

This project focuses on exploratory data analysis of the Bank Marketing dataset as part of the Thiranex Internship Task 4.

### Project Objectives

- Perform data cleaning and validation
- Handle missing and unknown categorical values
- Handle the `pdays = -1` sentinel value
- Preserve legitimate negative account balances
- Identify potential data leakage
- Perform exploratory data analysis (EDA)
- Analyze customer characteristics and subscription patterns
- Generate statistical summaries and visualizations

### Dataset

The project uses the Bank Marketing dataset containing information about client demographics, financial status, previous campaign contacts, and subscription outcomes.

The target variable is:

- `y` – whether the client subscribed to a term deposit (`yes` / `no`)

### Data Cleaning

The raw dataset is preserved in:

`data/raw/bank-full.csv`

The leakage-free processed dataset is:

`data/processed/bank_cleaned_leakfree.csv`

The following transformations were performed:

- Preserved explicit `unknown` categorical values
- Created `pcontacted` to distinguish previously contacted clients
- Converted the `pdays = -1` sentinel to `0` while preserving its meaning through `pcontacted`
- Created `is_overdrawn` for negative account balances
- Preserved all observations
- Excluded `duration` because it is a potential target-leakage variable

### Exploratory Data Analysis

The EDA includes:

- Target distribution
- Numerical variable analysis
- Balance analysis
- Age analysis
- Campaign and previous-contact analysis
- Customer job and education analysis
- Housing, loan and default associations
- Contact channel and campaign month analysis
- Correlation analysis
- Statistical testing
- Data visualizations

### Project Structure

```text
Thiranex_Task4_Real_World_Data_Project/
│
├── bank+marketing/
├── data/
│   ├── raw/
│   │   └── bank-full.csv
│   └── processed/
│       └── bank_cleaned_leakfree.csv
│
├── outputs/
│   ├── figures/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_numerical_distributions.png
│   │   ├── 03_balance_distribution.png
│   │   ├── 04_age_distribution.png
│   │   ├── 05_campaign_distribution.png
│   │   ├── 06_previous_contact_analysis.png
│   │   ├── 07_pcontacted_vs_target.png
│   │   ├── 08_job_vs_subscription_rate.png
│   │   ├── 09_education_vs_subscription_rate.png
│   │   ├── 10_credit_loan_vs_subscription_rate.png
│   │   ├── 11_month_contact_poutcome_vs_target.png
│   │   ├── 12_correlation_heatmap.png
│   │   └── 13_numerical_relationships_target.png
│   ├── bank_cleaning_report.txt
│   ├── eda_report.txt
│   ├── eda_quality_control_report.txt
│   ├── eda_descriptive_statistics.csv
│   ├── eda_categorical_summary.csv
│   ├── eda_target_analysis.csv
│   └── eda_correlation_matrix.csv
│
├── src/
│   ├── data_cleaning.py
│   └── eda_analysis.py
│
├── .gitignore
└── README.md
```

### Reproducibility & Pipeline Execution

To reproduce the data cleaning and exploratory data analysis pipelines from scratch, run the following commands from the project root:

1. **Execute Data Cleaning Pipeline**:
   ```bash
   python src/data_cleaning.py
   ```
   *Verifies the raw file integrity, drops `duration` (leakage mitigation), handles sentinels and overdrafts, and writes `data/processed/bank_cleaned_leakfree.csv` along with `outputs/bank_cleaning_report.txt`.*

2. **Execute Exploratory Data Analysis Pipeline**:
   ```bash
   python src/eda_analysis.py
   ```
   *Generates descriptive statistics, contingency tables, correlation matrices, hypothesis tests, audit reports, and all 13 visualizations in `outputs/figures/`.*
