# Water Pump RUL Prediction Project

## Overview
This project aims to predict the **Remaining Useful Life (RUL)** of a water pump using sensor data. The dataset (`rul_hrs.csv`) contains time-series sensor readings and RUL values, which are used to train a machine learning model to estimate how much time remains before the pump fails. The project employs **Principal Component Analysis (PCA)** for dimensionality reduction and a **Random Forest Regressor** for prediction.

## Dataset
- **File**: `rul_hrs.csv`
- **Description**: Contains 166,441 rows with 53 columns:
  - `timestamp`: Date and time of sensor readings.
  - 51 sensor columns (e.g., `sensor_00` to `sensor_51`): Various sensor measurements.
  - `rul`: Target variable representing the remaining useful life in hours.
- **Preprocessing**:
  - The `Unnamed: 0` column is dropped.
  - No missing values were found in the dataset.
  - Features are separated (`X`: sensor data, `y`: RUL).
  - PCA is applied to reduce the dimensionality of sensor data to 50 components.

## Methodology
1. **Data Loading**:
   - The dataset is loaded using `pandas`.
2. **Feature Engineering**:
   - Features (`X`) are extracted, excluding the `rul` column.
   - PCA is applied to transform the sensor data into 50 components (`X_scaled_pca`).
3. **Data Splitting**:
   - The dataset is split into 80% training and 20% testing sets using `train_test_split` with `random_state=42`.
4. **Model Training**:
   - A `RandomForestRegressor` is trained on the PCA-transformed training data.
   - Training time: Approximately 914 seconds.
5. **Evaluation**:
   - The model is evaluated on the test set using the following metrics:
     - **R² Score**: 99.09% (indicating excellent fit).
     - **Mean Absolute Error (MAE)**: 0.94%.
     - **Mean Squared Error (MSE)**: 0.07%.
     - **Root Mean Squared Error (RMSE)**: 2.56%.
6. **Model Saving**:
   - The trained model is saved as `CPU_Model.pkl` using `joblib`.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `numpy`
  - `joblib`
  - `time`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required packages:
   ```bash
   pip install pandas scikit-learn numpy joblib `

## Usage
1. Ensure the dataset (`rul_hrs.csv`) is in the correct directory.
2. Run the Jupyter notebook (`CPU.ipynb`) to:
   - Load and preprocess the data.
   - Train the Random Forest model.
   - Evaluate performance and save the model.
3. To use the saved model for predictions:
   ```python
   import joblib
   model = joblib.load('CPU_Model.pkl')
   predictions = model.predict(X_new_data)  # Replace with new PCA-transformed data
   ```

## Results
- The Random Forest model achieves a high R² score of 99.09%, indicating strong predictive performance.
- Low error metrics (MAE: 0.94%, MSE: 0.07%, RMSE: 2.56%) suggest accurate RUL predictions.
- The model is saved as `CPU_Model.pkl` for future use.

## Notes
- The notebook assumes the data is pre-scaled (e.g., `X_scaled` and `y_scaled`), but the scaling step is not explicitly shown. Ensure proper scaling if replicating.
- The dataset is large (166,441 rows), so training may take significant time (~15 minutes).
- For deployment, ensure new data is preprocessed (PCA-transformed) consistently with the training data.

## Future Improvements
- Include explicit data scaling (e.g., `StandardScaler`) before PCA.
- Experiment with other models (e.g., XGBoost, Neural Networks) for comparison.
- Perform hyperparameter tuning for the Random Forest model to potentially improve performance.
- Add cross-validation to ensure robustness.

## License
This project is licensed under the MIT License.
