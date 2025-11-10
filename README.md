# Delhi House Rent Prediction: An End-to-End MLOps Project

This project demonstrates a complete, end-to-end MLOps pipeline for a machine learning model. The goal is not just to build a model, but to deploy it in a scalable, reliable, and production-grade environment using a modern MLOps stack.

The system predicts house rent prices in Delhi based on features like location, size, and property type. It's deployed as a microservices architecture with a FastAPI backend and a Streamlit frontend, all orchestrated by Kubernetes and monitored in real-time.

---

## 🏛️ System Architecture

This diagram outlines the full MLOps workflow, from continuous integration and experiment tracking to a production deployment with autoscaling and monitoring.

![MLOps Architecture Diagram](/deployment/diagram.png)

---

## ✨ Key Features

- **End-to-End ML Pipeline:** From data preprocessing and feature engineering to model training.
- **CI/CD Automation:** **GitHub Actions** pipeline for automated testing, validation, and training.
- **Experiment Tracking:** **MLflow** for logging experiments, saving model parameters, and versioning.
- **Microservices Architecture:**
  - **FastAPI** backend to serve the model as a high-performance REST API.
  - **Streamlit** frontend for a user-friendly, interactive web interface.
- **Containerization:** **Docker** for packaging the API and frontend into portable, isolated images.
- **Orchestration:** **Kubernetes (K8s)** to deploy, manage, and scale the containerized services.
- **Real-time Monitoring:** **Prometheus** for scraping custom metrics (like API latency) and **Grafana** for rich, queryable dashboards.
- **Autoscaling:** **KEDA** (Kubernetes Event-driven Autoscaling) to automatically scale the FastAPI pods based on custom Prometheus metrics.

---

## 🛠️ Tech Stack

- **Programming & Frameworks:** Python, FastAPI, Streamlit
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **MLOps & DevOps:** Kubernetes (K8s), Docker, Prometheus, Grafana, MLflow, KEDA, Helm, GitHub Actions
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions

---

## 🚀 How to Run

There are two ways to run this project:

1.  **Locally with Docker Compose:** (Recommended for a quick demo)
2.  **On Kubernetes:** (For the full MLOps deployment)

### 1. Local Setup with Docker Compose

This method starts the FastAPI and Streamlit services on your local machine.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/delhi-house-rent-prediction.git](https://github.com/your-username/delhi-house-rent-prediction.git)
    cd delhi-house-rent-prediction
    ```

2.  **Build and run the containers:**

    ```bash
    docker compose build
    docker compose up -d
    ```

3.  **Access the applications:**
    - **Streamlit App:** `http://localhost:8501`
    - **FastAPI Docs:** `http://localhost:8000/docs`

### 2. Full Deployment on Kubernetes (Kind)

This method deploys the entire MLOps stack, including Prometheus, Grafana, and KEDA, onto a local Kubernetes cluster using `kind`.

#### **Prerequisites:**

- Docker Desktop
- `kind`
- `kubectl`
- `helm`

#### **Step 1: Create the Kubernetes Cluster**

(Using the provided `k8s-code` helper repository)

```bash
git clone [https://github.com/initcron/k8s-code.git](https://github.com/initcron/k8s-code.git)
cd k8s-code/helper/kind/
kind create cluster --config kind-three-node-cluster.yaml
cd ../../.. # Go back to your project's root
```

#### **Step 2: Deploy the Monitoring Stack (Prometheus & Grafana)**

```bash
# Add the Prometheus Helm repository
helm repo add prometheus-community [https://prometheus-community.github.io/helm-charts](https://prometheus-community.github.io/helm-charts)
helm repo update

# Install the stack in a new 'monitoring' namespace
helm upgrade --install prom   -n monitoring   --create-namespace   prometheus-community/kube-prometheus-stack   --set grafana.service.type=NodePort   --set grafana.service.nodePort=30200   --set prometheus.service.type=NodePort   --set prometheus.service.nodePort=30300
```

#### **Step 3: Deploy Your Applications**

Apply all the Kubernetes manifests for your services and deployments.

```bash
# From your project's root directory
kubectl apply -f deployment/kubernetes/
```

#### **Step 4: Deploy Monitoring & Autoscaling**

Apply the configurations for KEDA and the ServiceMonitor.

```bash
# From your project's root directory
kubectl apply -f deployment/monitoring/
```

#### **Step 5: Access the Services**

Everything is now running on your cluster. You can access the services via their NodePort:

- **Streamlit App:** `http://localhost:30000`
- **FastAPI Docs:** `http://localhost:30100/docs`
- **Grafana Dashboard:** `http://localhost:30200`
- **Prometheus UI:** `http://localhost:30300`

---

## 🔁 CI/CD Workflow (GitHub Actions)

This project is configured with a CI pipeline (`.github/workflows/main.yml`) that automates the MLOps lifecycle:

1. **Triggers:** Runs on every push or pull_request to the main branch.
2. **Tests:** Installs dependencies and runs the data processing and feature engineering scripts (`engineer.py`) as a test.
3. **Validation:** Spins up a temporary MLflow Docker container inside the CI job.
4. **Training:** Executes `train_model.py`, which trains the model and logs all metrics, parameters, and artifacts to the temporary MLflow server to ensure the model trains successfully.
5. **Build & Push (Next Step):** The pipeline can be extended to build and push the FastAPI and Streamlit Docker images to a container registry like Docker Hub or Azure CR.

---

## 📁 Project Structure

```
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── configs/                # Model configuration files (e.g., best_model_config.yaml)
├── data/
│   ├── raw/                # Raw, immutable data
│   ├── interim/            # Cleaned data
│   └── processed/          # Final, featured data for training
├── deployment/
│   ├── kubernetes/         # K8s manifests (Deployment, Service)
│   ├── monitoring/         # K8s monitoring (ServiceMonitor, KEDA ScaledObject)
│   └── mlflow/             # Docker compose for MLflow
├── models/
│   └── trained/            # Final trained model (.pkl) and preprocessor
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── api/                # FastAPI source code (main.py, inference.py)
│   ├── data/               # Data processing script
│   ├── features/           # Feature engineering script (engineer.py)
│   ├── models/             # Model training script (train_model.py)
│   └── streamlit_app/      # Streamlit frontend source code (app.py)
├── Dockerfile              # Dockerfile for the FastAPI service
├── docker-compose.yaml     # Docker compose for local dev
└── requirements.txt        # Python dependencies for the ML pipeline
```
