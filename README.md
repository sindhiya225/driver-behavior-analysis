# Driver Behavior & Risk Analysis System

## Project Overview
This project analyzes driving patterns using telematics data to identify:
- Risky driving behaviors
- Fuel-inefficient habits
- Safety compliance patterns
- Driver segmentation for targeted training

## Key Features
- **K-Means Clustering** for driver segmentation
- **Time-series analysis** of acceleration/deceleration patterns
- **Risk scoring algorithm** for each driver
- **Fuel efficiency metrics** calculation
- **Interactive Power BI dashboard**
- **Statistical hypothesis testing**

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Machine Learning (K-Means, PCA)
- Power BI for visualization
- Statistical Analysis (ANOVA, T-tests)
- Jupyter Notebooks for exploration

## Installation
```bash
git clone https://github.com/sindhiya225/driver-behavior-analysis.git
cd driver-behavior-analysis
pip install -r requirements.txt

# Run complete analysis pipeline
python main.py --config config/config.yaml

# Generate clusters
python -m src.clustering --data data/data_cleaned.csv --k 5

# Create Power BI dataset

python -m src.data_processor --output powerbi/dataset.csv
