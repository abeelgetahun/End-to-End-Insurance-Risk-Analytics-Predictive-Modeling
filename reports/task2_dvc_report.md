 
# Task 2: Data Version Control Implementation Report


This project demonstrates a reproducible and auditable data pipeline for insurance risk analytics using Data Version Control (DVC). In regulated industries like finance and insurance, reproducibility and auditability are essential for compliance, debugging, and transparency. DVC ensures that all data inputs are version-controlled alongside code, enabling robust and traceable machine learning workflows.

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── MachineLearningRating_v3.txt
│   └── processed/
│       └── insurance_data.csv
├── data_loader.py
├── .dvc/
├── .dvcignore
├── .git/
├── README.md
└── ...
```

---

## Getting Started

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
pip install dvc
```

### 3. Initialize DVC

```sh
dvc init
```

### 4. Configure Local DVC Remote Storage

```sh
mkdir /path/to/your/local/storage
dvc remote add -d localstorage /path/to/your/local/storage
```

### 5. Add Data to DVC

```sh
dvc add data/raw/insurance_data.csv
```

### 6. Commit Changes to Git

```sh
git add data/raw/insurance_data.csv.dvc .gitignore .dvc/config
git commit -m "Track raw insurance data with DVC"
```

### 7. Push Data to DVC Remote

```sh
dvc push
```

---

## Data Pipeline

- **Data Loading:** Uses `data_loader.py` to load, clean, and save processed data.
- **DVC Integration:** All data files are tracked and versioned with DVC for full reproducibility.
- **Branching:** Work is organized using feature branches (e.g., `task-2`) and merged via Pull Requests for auditability.

---

## Reproducibility & Auditability

- All data and code changes are tracked in Git and DVC.
- To reproduce a previous analysis, simply checkout the corresponding Git commit and run:
    ```sh
    dvc pull
    python data_loader.py
    ```

---

## Contributing

1. Create a new branch for your task (e.g., `task-2`).
2. Commit your changes with descriptive messages.
3. Open a Pull Request to merge into `main`.

---

## References

- [DVC Documentation](https://dvc.org/doc)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## License

This project is for educational and demonstration purposes.
## Overview
Implemented DVC for data versioning and pipeline management to ensure reproducibility and auditability.

## Implementation Summary
- Configured local DVC remote storage
- Tracked raw and processed datasets
- Created data pipeline with validation
- Established data versioning workflow

## Data Versions Created
1. Raw dataset: 1.2GB, 1M+ records
2. Cleaned dataset: Removed duplicates, handled missing values
3. Sample dataset: 10% sample for development
4. Filtered dataset: Specific provinces for targeted analysis

## Benefits Achieved
- Reproducible data pipeline
- Audit trail for data changes
- Efficient storage management
- Version control for large datasets