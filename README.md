# Wine Quality Classification - MLOps Pipeline

A complete end-to-end MLOps project that predicts wine quality (good/bad) based on physicochemical properties using machine learning.

## Project Overview

This project demonstrates a full MLOps pipeline including:
- Data preprocessing and feature engineering
- Model training with MLflow experiment tracking
- FastAPI REST API for predictions
- Streamlit UI for interactive predictions
- Docker containerization
- CI/CD with GitHub Actions
- AWS deployment (ECR + EC2)

## Dataset

The [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UCI Machine Learning Repository contains physicochemical properties of red wine samples with quality ratings.

**Features:**
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free/Total sulfur dioxide
- Density, pH, Sulphates, Alcohol

**Target:** Binary classification (good/bad quality based on threshold)

## Project Structure

```
wine_quality/
├── data/
│   └── wine_quality.csv          # Dataset
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Data preprocessing
│   ├── train.py                  # Model training with MLflow
│   └── predict.py                # Prediction utilities
├── api/
│   ├── __init__.py
│   └── main.py                   # FastAPI application
├── streamlit_app/
│   └── app.py                    # Streamlit UI
├── models/
│   ├── model.pkl                 # Trained model
│   └── scaler.pkl                # Feature scaler
├── tests/
│   ├── test_train.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── cicd.yml              # GitHub Actions CI/CD
├── mlruns/                       # MLflow tracking
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-service orchestration
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
mkdir -p data
curl -o data/wine_quality.csv "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```

### 3. Train the Model

```bash
python -m src.train
```

This will:
- Load and preprocess the wine quality dataset
- Train a Random Forest classifier
- Log metrics and parameters to MLflow
- Save model and scaler artifacts

### 4. View MLflow Experiments

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 to view experiment tracking.

### 5. Start the API Server

**Option 1: FastAPI dev mode (with hot reload)**
```bash
fastapi dev api/main.py --host 0.0.0.0 --port 8000
```

**Option 2: Uvicorn**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 7. Start Streamlit UI (Optional)

```bash
streamlit run streamlit_app/app.py --server.port 8501
```

Open http://localhost:8501 for the interactive UI.

## Docker

### Build and Run with Docker

```bash
# Build image
docker build -t wine-quality-api:latest .

# Run container
docker run -d -p 8000:8000 --name wine-api wine-quality-api:latest
```

### Run with Docker Compose

```bash
# Start API and Streamlit services
docker-compose up -d

# Include MLflow UI (dev profile)
docker-compose --profile dev up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8000 | http://localhost:8000 |
| Streamlit | 8501 | http://localhost:8501 |
| MLflow (dev) | 5000 | http://localhost:5000 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single wine quality prediction |
| `/predict_batch` | POST | Batch predictions |
| `/features` | GET | Feature information |
| `/sample` | GET | Sample wine features |

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "quality_label": "bad",
  "probability_bad": 0.85,
  "probability_good": 0.15,
  "confidence": 0.85
}
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_train.py -v
pytest tests/test_api.py -v
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/cicd.yml`) includes:

1. **Test Job**: Runs on every push/PR
   - Set up Python environment
   - Install dependencies
   - Download dataset
   - Train model
   - Run pytest

2. **Build Job**: Runs on push to main
   - Build Docker image
   - Cache layers for faster builds

3. **Push to ECR**: Runs on push to main
   - Push Docker image to AWS ECR

4. **Deploy to EC2**: Runs on push to main
   - SSH into EC2 instance
   - Pull latest image
   - Restart container

### Required GitHub Secrets

For AWS deployment, add these secrets in your GitHub repository:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `EC2_HOST` | EC2 public IP/hostname |
| `EC2_USERNAME` | EC2 SSH username (e.g., `ec2-user`) |
| `EC2_SSH_KEY` | EC2 private SSH key |

## Configuration

Edit `config.yaml` to customize:
- Model type (random_forest, logistic_regression)
- Model hyperparameters
- Data split ratio
- Quality threshold
- MLflow settings

## Model Performance

After training, typical metrics:
- Accuracy: ~80%
- Precision: ~82%
- Recall: ~80%
- F1 Score: ~81%

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Developer     │────▶│    GitHub       │────▶│  GitHub Actions │
│   Push Code     │     │   Repository    │     │    CI/CD        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                ▼                                │
                        │  ┌─────────────────┐     ┌─────────────────┐                   │
                        │  │   Run Tests     │────▶│  Build Docker   │                   │
                        │  │   (pytest)      │     │     Image       │                   │
                        │  └─────────────────┘     └────────┬────────┘                   │
                        │                                   │                             │
                        │                                   ▼                             │
                        │                          ┌─────────────────┐                   │
                        │                          │   Push to ECR   │                   │
                        │                          └────────┬────────┘                   │
                        │                                   │                             │
                        │                                   ▼                             │
                        │                          ┌─────────────────┐                   │
                        │                          │  Deploy to EC2  │                   │
                        │                          └────────┬────────┘                   │
                        │                                   │                             │
                        └───────────────────────────────────┼─────────────────────────────┘
                                                            │
                                                            ▼
                        ┌─────────────────────────────────────────────────────────────────┐
                        │                         AWS EC2                                  │
                        │  ┌─────────────────┐     ┌─────────────────┐                   │
                        │  │   FastAPI       │◀───▶│   Streamlit     │                   │
                        │  │   :8000         │     │   :8501         │                   │
                        │  └─────────────────┘     └─────────────────┘                   │
                        └─────────────────────────────────────────────────────────────────┘
```

## Next Steps

- [x] Phase 1: Local Model Development
- [x] Phase 2: FastAPI Deployment
- [x] Phase 3: Docker Containerization
- [x] Phase 4: CI/CD with GitHub Actions
- [ ] Phase 5: AWS Deployment (ECR + EC2)
- [ ] Phase 6: Full Integration Testing
