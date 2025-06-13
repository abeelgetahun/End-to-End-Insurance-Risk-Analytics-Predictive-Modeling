# Task 1 Report: EDA, Data Understanding, and Project Setup

## 1. Git & GitHub Project Setup 🗂️🐙

- **Repository Created:** The project is hosted on GitHub with a clear structure, including `README.md`, `.gitignore`, and a well-organized directory layout (`src/`, `notebooks/`, `data/`, `reports/`, etc.).
- **Branching:** A dedicated branch (e.g., `task-1`) was used for Task 1 analysis, with regular commits and descriptive messages.
- **CI/CD:** GitHub Actions workflow (`.github/workflows/ci.yml`) is configured for continuous integration, ensuring code quality and reproducibility. ⚙️✅
- **Documentation:** The `README.md` provides an overview, project goals, and directory structure. 📄

## 2. Data Understanding & Environment 🧑‍💻📦

- **Data Loading:** Raw insurance data was loaded using a custom `InsuranceDataLoader` class (`src/data/data_loader.py`), supporting robust data ingestion and conversion.
- **Environment:** The project uses a virtual environment (`.venv`), and dependencies are managed via `requirements.txt` and `setup.py`. 🐍
- **Reproducibility:** All code and notebooks are version-controlled and environment setup is documented. 🔁

## 3. Exploratory Data Analysis (EDA) 📊🔍

### 3.1 Data Summarization & Structure

- **Descriptive Statistics:** Calculated for all key numerical features (`TotalPremium`, `TotalClaims`, `SumInsured`, etc.) using `describe()` and custom summaries in `02_eda_analysis.ipynb`. 📈
- **Data Types:** Data types were reviewed and corrected (e.g., dates parsed, numerics coerced) during preprocessing. 🏷️
- **Data Quality:** Missing values were assessed and summarized; memory usage and dataset shape were reported. 🧹

### 3.2 Univariate Analysis

- **Numerical Distributions:** Histograms plotted for main financial variables (e.g., `TotalPremium`, `TotalClaims`, `SumInsured`) with outlier handling for clear visualization. 📊
- **Categorical Distributions:** Bar charts for top categories in `Province`, `Gender`, `VehicleType`, and `make` to understand portfolio composition. 🗃️

### 3.3 Bivariate & Multivariate Analysis

- **Loss Ratio Analysis:** Computed overall and segmented loss ratios (TotalClaims / TotalPremium) by `Province`, `VehicleType`, and `Gender`. Visualized with bar plots to highlight high- and low-risk segments. 🟧
- **Correlation Matrix:** Explored relationships between numerical variables using a heatmap, identifying strong correlations and potential drivers of risk. 🔥
- **Temporal Trends:** Time series plots of monthly premium, claims, policy count, and loss ratio, revealing business seasonality and trends over the 18-month period. 📆
- **Geographical & Segment Trends:** Created a risk heatmap (Province vs. VehicleType) to identify regional and product-specific risk concentrations. 🗺️

### 3.4 Outlier Detection

- **Box Plots:** Used for key numerical variables to detect and quantify outliers (e.g., in `TotalClaims`, `CustomValueEstimate`). 📦
- **Summary Table:** Outlier counts and percentages reported for transparency and further investigation. 🧾

### 3.5 Creative Visualizations

- **Premium vs. Claims Scatter Plot:** Interactive scatter plot colored by loss ratio, highlighting break-even and outlier policies. 🎯
- **Risk Heatmap:** Loss ratio heatmap by Province and Vehicle Type, visually pinpointing high-risk segments. 🌡️
- **Temporal Dashboard:** Multi-panel time series dashboard for premium, claims, policy count, and loss ratio trends. 🖥️

## 4. Key Insights & Statistical Thinking 🧠📐

- **Portfolio Loss Ratio:** Overall loss ratio calculated and compared across business segments.
- **Outlier Impact:** Identified and quantified outliers that could skew analysis, supporting robust statistical conclusions.
- **Segmented Risk:** Actionable insights into which provinces, vehicle types, and customer segments drive higher risk.
- **Temporal Patterns:** Detected changes in claim frequency/severity over time, supporting business planning.
- **Statistical Evidence:** Used distributions, box plots, and correlation analysis to support findings and recommendations.

## 5. References & Proactivity 📚🚀

- **Self-Learning:** Leveraged pandas, seaborn, matplotlib, plotly, and statistical methods for comprehensive EDA.
- **References:** [Pandas Documentation](https://pandas.pydata.org/), [Seaborn Documentation](https://seaborn.pydata.org/), [Plotly Express](https://plotly.com/python/), [Scipy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html).

---

**Conclusion:**  
All minimum requirements for Task 1 have been met, including robust Git/GitHub usage, CI/CD, thorough EDA, statistical analysis, and creative visualizations. The project is well-documented, reproducible, and provides actionable business insights for further modeling.
