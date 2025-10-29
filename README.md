# Water Quality Classification System

This project implements a machine learning pipeline for classifying water quality based on various parameters. The classification follows the Central Pollution Control Board (CPCB) standards for Water Quality Index (WQI).

## Project Structure

- `water_quality_classification.ipynb`: Jupyter notebook containing the complete ML pipeline
- `app.py`: Streamlit application for water quality prediction
- `water_quality_with_final_wqi.csv`: Original dataset
- `best_water_quality_model.pkl`: Saved best model
- `scaler.pkl`: Saved feature scaler
- `label_mapping.pkl`: Mapping between numeric and text labels
- `feature_names.pkl`: List of feature names
- `demo_test_cases.csv`: Sample cases for demonstration

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Hyperparameter tuning
- Interactive Streamlit web application
- Demo cases for quick testing

## CPCB Water Quality Classification

The water quality labels are based on the CPCB standards for WQI:

- **Moderate (51-100)**: Water is suitable for drinking after conventional treatment
- **Poor (26-50)**: Water can be used for wildlife and fisheries
- **Very Poor (0-25)**: Water is suitable for controlled waste disposal

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Himanshu-joshi986/Water-Quality-Classification
   cd water-quality-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `water_quality_classification.ipynb` and run all cells to:
   - Load and preprocess the data
   - Perform EDA
   - Train and evaluate models
   - Save the best model and components

### Running the Streamlit Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the application to:
   - Enter water quality parameters manually
   - Use demo cases for quick testing
   - View predictions and explanations

## Deployment on Streamlit Cloud

1. Create a GitHub repository and push your code:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/Himanshu-joshi986/Water-Quality-Classification
   git push -u origin main
   ```

2. Create a `requirements.txt` file with all dependencies:
   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   xgboost
   imbalanced-learn
   shap
   streamlit
   ```

3. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account

4. Click on "New app" and select your repository, branch, and the main file (`app.py`)

5. Click "Deploy" and wait for the deployment to complete

6. Your app will be available at a URL like: `https://yourusername-water-quality-classification-app-xyz.streamlit.app`

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost (optional)
- imbalanced-learn
- shap (optional)
- streamlit
- pickle

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Central Pollution Control Board (CPCB) for the water quality standards
- The dataset providers for making the water quality data available