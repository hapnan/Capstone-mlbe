# Bank Lead Scoring API

A FastAPI-based machine learning service that predicts the probability of bank customers making a deposit. This API uses a LightGBM model to provide lead scoring predictions for targeted marketing campaigns.

## Features

- **Single and Batch Predictions**: Support for predicting individual customers or multiple customers at once
- **Optional Customer ID Tracking**: Include customer IDs in requests to track predictions
- **RESTful API**: Built with FastAPI for high performance and automatic documentation
- **Docker Support**: Containerized deployment ready
- **Input Validation**: Comprehensive input validation using Pydantic models

## Project Structure

```
Capstone-mlbe/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   └── utils.py             # Custom transformers and utilities
├── model/
│   └── model_lead_scoring_final_deployment.joblib  # Trained model
├── best_lgbm_params.json    # Model hyperparameters
├── Dockerfile               # Docker configuration
├── requirements.txt         # Development dependencies
├── requirements-deploy.txt  # Production dependencies
└── README.md
```

## Installation

### Local Development

1. Clone the repository:

```bash
git clone https://github.com/hapnan/Capstone-mlbe.git
cd Capstone-mlbe
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Docker Deployment

Build and run using Docker:

```bash
docker build -t bank-lead-scoring-api .
docker run -p 8000:8000 bank-lead-scoring-api
```

## Running the API

### Local

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the API is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### `GET /`

Health check endpoint.

**Response:**

```json
{
  "status": "API is running successfully"
}
```

### `POST /predict`

Predict deposit probability for one or more customers.

**Request Body (Single Prediction):**

```json
{
  "data": {
    "id": 1,
    "age": 56,
    "job": "housemaid",
    "marital": "married",
    "education": "basic.4y",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "telephone",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
  }
}
```

**Request Body (Batch Prediction):**

```json
{
  "data": [
    {
      "id": 1,
      "age": 56,
      "job": "housemaid",
      ...
    },
    {
      "id": 2,
      "age": 45,
      "job": "technician",
      ...
    }
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {
      "id": 1,
      "prediction_label": "Deposit",
      "lead_score_probability": 0.8534
    }
  ],
  "count": 1,
  "status": "success"
}
```

## Input Features

| Field            | Type    | Description                  | Constraints                                    |
| ---------------- | ------- | ---------------------------- | ---------------------------------------------- |
| `id`             | int/str | Optional customer identifier | Optional                                       |
| `age`            | int     | Customer age                 | 18-100                                         |
| `job`            | str     | Job type                     | e.g., admin, blue-collar, entrepreneur         |
| `marital`        | str     | Marital status               | married, single, divorced                      |
| `education`      | str     | Education level              | e.g., basic.4y, high.school, university.degree |
| `default`        | str     | Has credit in default?       | yes, no, unknown                               |
| `housing`        | str     | Has housing loan?            | yes, no, unknown                               |
| `loan`           | str     | Has personal loan?           | yes, no, unknown                               |
| `contact`        | str     | Contact type                 | cellular, telephone                            |
| `month`          | str     | Last contact month           | jan, feb, mar, ...                             |
| `day_of_week`    | str     | Last contact day             | mon, tue, wed, thu, fri                        |
| `campaign`       | int     | Contacts during campaign     | ≥1                                             |
| `pdays`          | int     | Days since last contact      | ≥0 (999 = never)                               |
| `previous`       | int     | Previous contacts            | ≥0                                             |
| `poutcome`       | str     | Previous campaign outcome    | failure, nonexistent, success                  |
| `emp_var_rate`   | float   | Employment variation rate    | -                                              |
| `cons_price_idx` | float   | Consumer price index         | -                                              |
| `cons_conf_idx`  | float   | Consumer confidence index    | -                                              |
| `euribor3m`      | float   | Euribor 3 month rate         | -                                              |
| `nr_employed`    | float   | Number of employees          | -                                              |

## Output Format

The API returns predictions in a consistent format:

- `predictions`: Array of prediction objects
  - `id`: Customer ID (if provided in request)
  - `prediction_label`: "Deposit" or "No Deposit"
  - `lead_score_probability`: Probability score (0-1)
- `count`: Number of predictions
- `status`: "success" or error message

## Example Usage

### Using cURL

**Single Prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "age": 56,
      "job": "housemaid",
      "marital": "married",
      "education": "basic.4y",
      "default": "no",
      "housing": "no",
      "loan": "no",
      "contact": "telephone",
      "month": "may",
      "day_of_week": "mon",
      "campaign": 1,
      "pdays": 999,
      "previous": 0,
      "poutcome": "nonexistent",
      "emp_var_rate": 1.1,
      "cons_price_idx": 93.994,
      "cons_conf_idx": -36.4,
      "euribor3m": 4.857,
      "nr_employed": 5191.0
    }
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"

# Single prediction
data = {
    "data": {
        "id": 1,
        "age": 56,
        "job": "housemaid",
        "marital": "married",
        "education": "basic.4y",
        "default": "no",
        "housing": "no",
        "loan": "no",
        "contact": "telephone",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0
    }
}

response = requests.post(url, json=data)
print(response.json())
```

## Model Information

The API uses a LightGBM (Light Gradient Boosting Machine) model trained on bank marketing campaign data. The model predicts whether a customer will make a deposit based on:

- Customer demographics
- Campaign interaction history
- Economic indicators

## Dependencies

### Production

- FastAPI 0.124.0
- Uvicorn 0.23.2
- Pandas 2.3.3
- Scikit-learn 1.7.2
- LightGBM 4.6.0
- Joblib 1.5.2
- Pydantic 2.12.3

See `requirements-deploy.txt` for complete production dependencies and `requirements.txt` for development dependencies.

## License

[Add your license information here]

## Contact

Repository: [https://github.com/hapnan/Capstone-mlbe](https://github.com/hapnan/Capstone-mlbe)

## Contributing

[Add contributing guidelines if applicable]
