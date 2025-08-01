# Disease Classification System

A comprehensive machine learning system for disease prediction based on symptoms, featuring data processing pipelines, multiple ML models, publication-quality visualizations, and a web interface.

## Project Structure

```
disease-classification-system/
├── README.md                           # Project documentation
├── requirements.txt                    # Main requirements
├── requirements_build.txt              # Build-specific requirements
├── requirements_deploy.txt             # Deployment requirements
├── .gitignore                          # Git ignore file
│
├── data/
│   ├── raw/                           # Original datasets
│   │   ├── dataset.csv
│   │   ├── symptom_Description.csv
│   │   ├── symptom_precaution.csv
│   │   └── Symptom-severity.csv
│   └── processed/                     # Combined datasets
│       ├── combined_disease_dataset.csv
│       └── data_summary_report.txt
│
├── src/                               # Source code
│   ├── data_combiner.py               # Data combination pipeline
│   ├── ml_pipeline.py                 # ML training pipeline
│   └── run_pipelines.py               # Main pipeline runner
│
├── web_app/                           # Web application
│   ├── app.py                         # Flask application
│   ├── app_data/                      # JSON data for web app
│   │   ├── symptoms.json              # Symptom database
│   │   ├── symptom_batches.json       # Symptom batches
│   │   ├── treatments.json            # Treatment information
│   │   └── condition_patterns.json    # Rule-based patterns
│   ├── templates/                     # HTML templates
│   │   └── index.html                 # Main web interface
│   └── static/                        # CSS, JS, images
│       ├── css/
│       ├── js/
│       └── images/
│
├── models/                            # Model artifacts and analysis
│   ├── saved_models/
│   │   └── disease_classification_model.pkl
│   ├── analysis/
│   │   ├── model_comparison_plots.png
│   │   └── feature_importance_analysis.csv
│   └── visualizations/                # Publication quality graphs
│       ├── graph_1_model_performance.png
│       ├── graph_2_cv_stability.png
│       ├── graph_3_roc_comparison.png
│       ├── graph_4_feature_importance.png
│       └── graph_5_rf_superiority.png
│
└── reports/                           # Execution reports and logs
    └── pipeline_execution_report.txt
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv disease_ml_env

# Activate environment
# Windows:
disease_ml_env\Scripts\activate
# macOS/Linux:
source disease_ml_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV files in the `data/raw/` directory:
- `dataset.csv` - Main symptom dataset
- `symptom_Description.csv` - Disease descriptions
- `symptom_precaution.csv` - Disease precautions
- `Symptom-severity.csv` - Symptom severity weights

### 3. Run Complete Pipeline

```bash
cd src/
python run_pipelines.py
```

This will:
- Combine all CSV files into a clean dataset
- Train multiple ML models
- Generate publication-quality visualizations
- Save the best model for deployment

### 4. Run Web Application

```bash
cd web_app/
python app.py
```

Access the web interface at `http://localhost:5000`

## Pipeline Components

### Data Combination Pipeline (`src/data_combiner.py`)
- Combines multiple CSV files
- Validates data quality
- Creates symptom severity features
- Generates comprehensive statistics

### ML Training Pipeline (`src/ml_pipeline.py`)
- Trains multiple models (Random Forest, XGBoost, etc.)
- Performs hyperparameter tuning
- Creates publication-quality visualizations
- Saves best model for deployment

### Web Application (`web_app/app.py`)
- Flask-based web interface
- ML model integration
- Rule-based fallback system
- RESTful API endpoints

## Model Performance

The system typically achieves:
- **Accuracy**: 85%+
- **F1-Score**: 80%+
- **ROC-AUC**: 99%+

Random Forest consistently performs best across metrics.

## API Endpoints

- `POST /api/analyze` - Analyze symptoms and predict diseases
- `GET /api/symptoms` - Get all available symptoms
- `GET /api/suggest-symptoms/<query>` - Get symptom suggestions
- `GET /api/treatment/<condition>` - Get treatment information
- `GET /api/debug` - System status and debug info

## Output Files

### Data Processing
- `data/processed/combined_disease_dataset.csv` - Clean, ML-ready dataset
- `data/processed/data_summary_report.txt` - Data quality report

### Model Training
- `models/saved_models/disease_classification_model.pkl` - Trained model
- `models/analysis/model_comparison_plots.png` - Performance charts
- `models/analysis/feature_importance_analysis.csv` - Feature analysis

### Visualizations
- `models/visualizations/graph_1_model_performance.png` - Performance dashboard
- `models/visualizations/graph_2_cv_stability.png` - Cross-validation analysis
- `models/visualizations/graph_3_roc_comparison.png` - ROC curve comparison
- `models/visualizations/graph_4_feature_importance.png` - Feature importance
- `models/visualizations/graph_5_rf_superiority.png` - Model superiority analysis

## Development

### Adding New Models
1. Add model to `src/ml_pipeline.py` in the `initialize_models()` method
2. Add hyperparameters to `hyperparameter_tuning()` if needed
3. Re-run the training pipeline

### Adding New Features
1. Modify feature engineering in `src/ml_pipeline.py`
2. Update the data combination pipeline if needed
3. Re-train models

### Web Interface Customization
1. Modify templates in `web_app/templates/`
2. Update static assets in `web_app/static/`
3. Add new API endpoints in `web_app/app.py`

## Requirements

- Python 3.12+
- 4GB+ RAM for training
- 1GB+ disk space for outputs
- CSV dataset files

## License

This project is for educational and research purposes.

## Data Source

Expected CSV format from Kaggle disease-symptom datasets:
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

## Support

For issues or questions:
1. Check the execution reports in `reports/`
2. Review console output for detailed error messages
3. Ensure all required CSV files are in `data/raw/`
4. Verify Python environment and package versions