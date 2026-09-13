# Datasets Used

The training pipeline combines agricultural production, climate, soil, fertilizer, drought, and market-price datasets. Restore these files locally before running `python train_model.py`.

## Required Folder Structure

Restore datasets locally using this structure before running `python train_model.py`:

```text
Dataset/
|-- Agricultural Crop Yield in Indian States Dataset/
|   `-- crop_yield.csv
|-- Crop Recommendation Dataset/
|   `-- Crop_recommendation.csv
|-- Crop Production Statistics - India/
|   `-- APY.csv
|-- Fertilizers by Product FAO/
|   `-- FertilizersProduct.csv
|-- Global Food Price Inflation/
|   |-- WLD_RTFP_country_2023-10-02.csv
|   |-- WLD_RTP_details_2023-10-02.csv
|   `-- ddi-documentation-*.pdf
|-- Indian Agriculture and Climate Dataset/
|   `-- data/
|       |-- rainfall.csv
|       |-- temperature.csv
|       `-- Crops/
|-- Predict Droughts using Weather & Soil Data/
|   |-- soil_data.csv
|   |-- train_timeseries/
|   |   `-- train_timeseries.csv
|   |-- validation_timeseries/
|   |   `-- validation_timeseries.csv
|   `-- test_timeseries/
|       `-- test_timeseries.csv
`-- Rainfall in Pakistan/
    `-- Rainfall_1901_2016_PAK.csv
```

## Download Sources

| Dataset | Purpose | Source |
| --- | --- | --- |
| Agricultural Crop Yield in Indian States Dataset | State-level crop yield, production, and area data from 1997-2020 | https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset |
| Crop Recommendation Dataset | NPK, temperature, humidity, pH, rainfall, and crop labels | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| Indian Agriculture and Climate Dataset (1961-2018) | Long-run India crop, rainfall, and temperature features | https://www.kaggle.com/datasets/swarooprangle/indian-agriculture-and-climate-dataset-1961-2018 |
| Predict Droughts using Weather & Soil Data | Soil indicators and large drought time-series data | https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data |
| Global Food Price Inflation | Country-level monthly food price inflation estimates | https://microdata.worldbank.org/index.php/catalog/4509 |
| FAOSTAT fertilizer data | Fertilizer usage and product indicators | https://www.fao.org/faostat/ |
| Pakistan rainfall data | Historical rainfall data for Pakistan | https://opendata.com.pk/dataset/rainfall-in-pakistan |
| India Area, Production, Yield (APY) | Crop-wise area, production, and yield statistics | https://indiadataportal.com/ |

## Dataset Preparation

1. Download each dataset from the source links above.
2. Place each file in the matching folder shown in the required structure.
3. Keep filenames exactly as expected by `train_model.py`.
4. Validate CSV encodings before training; FAO fertilizer files may need `latin-1`.
5. Run `python train_model.py` to generate a fresh model and training report.

## Model Artifact

The web app uses:

```text
Trained_models/latest_model.pkl
```

Raw datasets are required for retraining only. Prediction and dashboard usage work from the trained model artifact.
