# KNN Studio

A small FastAPI app for exploring the K-nearest-neighbors model from `01-K Nearest Neighbors with Python.ipynb`. It uses the existing `Classified Data` file and presents a dataset page, a prediction form, and an evaluation page.

## Folder map

```text
project-folder/
|-- main.py                  FastAPI routes and form/API validation
|-- KNN/
|   |-- __init__.py          Marks KNN as a Python package
|   `-- classifieddata1.py   Dataset checks, model pipeline, predictions, evaluation
|-- templates/               HTML pages rendered by Jinja
|   |-- base.html            Shared header, navigation, and page frame
|   |-- home.html            Dataset summary and class balance
|   |-- predict.html         Feature form and prediction result
|   `-- evaluation.html      Test score, confusion matrix, and K comparison
|-- static/styles.css        Page styling
|-- tests/test_app.py        Model and route checks
|-- Classified Data          Source CSV used by the model
|-- requirements.txt         Python libraries needed to run the app
`-- railway.json             Railway start and health-check settings
```

The notebook and the other course files are separate learning materials. The app uses only `Classified Data`.

## How the pieces work

FastAPI maps a URL and HTTP method to a Python function called a route. For example, `GET /predict` returns the prediction page, and `POST /predict` receives its submitted form. That route checks each value, calls `predict_one()` in `KNN/classifieddata1.py`, then gives the result to Jinja to render as HTML.

The `/api/predict` route accepts the same ten feature names as JSON instead of a browser form. FastAPI validates the JSON body and provides interactive request documentation at `/docs`. `/health` returns a small JSON response for deployment health checks.

`classifieddata1.py` is the model layer: it loads the CSV, checks the expected columns, and keeps the scaler and KNN estimator together in a scikit-learn `Pipeline`. The scaler is fit on training rows during evaluation, preventing the test rows from influencing preprocessing. The UI prediction pipeline is fit on the full dataset after the evaluation process is defined.

The notebook chose K=23, so that is the prediction model's starting setting. The evaluation page reports K=23 on a held-out test split and compares K=1 through K=39 with five-fold cross-validation on training rows. The class and feature names in this dataset are anonymized; class 0/1 are not assigned real-world meanings.

This version uses HTML templates rather than Streamlit so there is one server and one deployment to understand. Streamlit can make Python-first interfaces quickly, but combining it with FastAPI would mean running and connecting two applications. You can add that architecture later if you specifically need Streamlit widgets.

## Run locally

Use Python 3.10 or newer. Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/`. The prediction UI is at `/predict`, evaluation is at `/evaluation`, automatic API docs are at `/docs`, and the health check is at `/health`.

To run the tests:

```powershell
python -m pytest
```

## Push to GitHub

This workspace was not a Git repository when the app work began. Create an empty GitHub repository first, then from this folder initialize Git and add only the app files and its source dataset. For example:

```powershell
git init
git add .gitignore README.md main.py requirements.txt railway.json KNN templates static tests "Classified Data"
git status
git commit -m "Build KNN FastAPI learning app"
git branch -M main
git remote add origin https://github.com/YOUR-NAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository URL. Review `git status` before committing; do not add notebooks or the unrelated advertising/Titanic datasets unless you intend to publish those too. GitHub may prompt you to authenticate using its configured credential flow.

## Deploy on Railway

1. Sign in to Railway and create a new project from the GitHub repository.
2. Select the repository and deploy the `main` branch. Railway reads `railway.json`, installs dependencies from `requirements.txt`, and runs Uvicorn on the injected `$PORT`.
3. In the Railway service settings, generate a public domain.
4. Open the service URL and check `/health`, `/`, `/predict`, and `/evaluation`.

Keep credentials and `.env` files out of Git. Railway deployment and GitHub publishing require access to your accounts; this project only prepares the local code and configuration.