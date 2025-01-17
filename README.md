# Comprehensive Anomaly Detection in Time Series Data

## Project Overview

This repository provides a comprehensive approach to detecting anomalies in time series data using various methods, including Long Short-Term Memory (LSTM) networks and the Facebook Prophet model. The project demonstrates the application of these models to identify anomalies in sequential data and compare their performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [LSTM Model](#lstm-model)
  - [Prophet Model](#prophet-model)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is available at [Kaggle - Walmart Sales Dataset](https://www.kaggle.com/datasets/mikhail1681/walmart-sales). The dataset contains the following file:

- **`Walmart_Sales.csv`**: This file contains historical sales data including information on sales, store_id, item_id, date, and other relevant features.


## Project Structure

```plaintext
Comprehensive-Anomaly-Detection/
│
├── data/
│   ├── walmart_sales.csv
│   └── ...
├── notebooks/
│   ├── lstm/
│   │   ├── data_preprocessing.ipynb
│   │   ├── model_training.ipynb
│   │   ├── anomaly_detection.ipynb
│   ├── prophet/
│   │   ├── data_preprocessing.ipynb
│   │   ├── model_training.ipynb
│   │   ├── anomaly_detection.ipynb
│   └── ...
├── src/
│   ├── lstm/
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── detect_anomalies.py
│   ├── prophet/
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── detect_anomalies.py
│   └── utils.py
├── images/
│   ├── lstm_anomalies_plot.png
│   ├── prophet_anomalies_plot.png
│   └── ...
├── README.md
├── requirements.txt
└── LICENSE
```

- **data/**: Contains datasets used for training and testing.
- **notebooks/**: Jupyter notebooks for each method (LSTM, Prophet), including data preprocessing, model training, and anomaly detection.
- **src/**: Source code organized by method (LSTM, Prophet) for data preprocessing, model definition, and anomaly detection.
- **images/**: Folder for visualizations such as plots of detected anomalies.
- **README.md**: Project documentation.
- **requirements.txt**: List of dependencies required to run the project.
- **LICENSE**: License for the project.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Comprehensive-Anomaly-Detection-Hub.git
cd Comprehensive-Anomaly-Detection-Hub
pip install -r requirements.txt
```

## Usage

### LSTM Model

1. **Data Preprocessing**:
   - Preprocess the time series data by normalizing and creating sequences.
   - Example:
   ```bash
   python src/lstm/data_preprocessing.py
   ```

2. **Model Training**:
   - Train the LSTM model on the preprocessed data.
   - Example:
   ```bash
   python src/lstm/model.py
   ```

3. **Anomaly Detection**:
   - Use the trained model to detect anomalies in the time series data.
   - Example:
   ```bash
   python src/lstm/detect_anomalies.py
   ```

### Prophet Model

1. **Data Preprocessing**:
   - Preprocess the time series data to prepare it for the Prophet model.
   - Example:
   ```bash
   python src/prophet/data_preprocessing.py
   ```

2. **Model Training**:
   - Train the Prophet model on the preprocessed data.
   - Example:
   ```bash
   python src/prophet/model.py
   ```

3. **Anomaly Detection**:
   - Use the trained Prophet model to detect anomalies in the time series data.
   - Example:
   ```bash
   python src/prophet/detect_anomalies.py
   ```

## Model Architectures

### LSTM Model

- **Input Layer**: Sequences of time series data.
- **LSTM Layers**: Capture temporal dependencies in the data.
- **Dense Layer**: Outputs a single value predicting the next point in the sequence.
- **Loss Function**: Mean Squared Error (MSE).

### Prophet Model

- **Prophet Components**:
  - **Trend**: Captures the long-term trend in the data.
  - **Seasonality**: Models periodic changes (daily, weekly, yearly).
  - **Holidays**: Accounts for special events or holidays affecting the data.

## Results

The project compares the performance of different models (LSTM, Prophet) in detecting anomalies in the NAB dataset. Detected anomalies are visualized in time series plots.

- Example plot of LSTM detected anomalies:

![LSTM Anomalies Plot](images/lstm_anomalies_plot.png)

- Example plot of Prophet detected anomalies:

![Prophet Anomalies Plot](images/prophet_anomalies_plot.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The [NAB dataset](https://github.com/numenta/NAB) provided by Numenta, Inc.
- TensorFlow and Keras for providing tools to build the LSTM model.
- Facebook Prophet for the time series forecasting model.
