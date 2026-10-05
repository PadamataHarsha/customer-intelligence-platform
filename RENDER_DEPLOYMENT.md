# Deployment to Render

This guide explains how to deploy the KNN Studio FastAPI application to Render.

## Prerequisites

1. A Render account (https://render.com)
2. This GitHub repository connected to your Render account
3. Python 3.10+ for local testing

## Deployment Steps

### Option 1: Deploy using render.yaml (Recommended)

1. Go to https://dashboard.render.com/new/web
2. Connect your GitHub repository
3. Render will automatically detect `render.yaml` in the root directory
4. Configure the following settings:
   - **Name**: `knn-studio` (or your preferred name)
   - **Runtime**: Python
   - **Build Command**: `pip install -r knn-app/requirements.txt`
   - **Start Command**: `cd knn-app && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click "Deploy"

### Option 2: Manual Deployment

1. Go to https://dashboard.render.com/new/web
2. Connect your GitHub repository
3. Fill in the deployment settings:
   - **Name**: `knn-studio`
   - **Runtime**: Python
   - **Runtime Version**: 3.10
   - **Build Command**: `pip install -r knn-app/requirements.txt`
   - **Start Command**: `cd knn-app && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Leave other settings at defaults
5. Click "Create Web Service"

## Important Configuration

### Environment Variables
No environment variables are required for basic deployment. The app reads from the local `Classified Data` file.

### Health Check
The deployment includes a health check endpoint at `/health`. Render will use this to verify the service is running.

### Port
The application automatically uses the `$PORT` environment variable provided by Render, so no configuration is needed.

## After Deployment

Once deployed, your application will be available at:
- **Home page**: `https://your-service-name.onrender.com/`
- **Prediction UI**: `https://your-service-name.onrender.com/predict`
- **Evaluation**: `https://your-service-name.onrender.com/evaluation`
- **API Docs**: `https://your-service-name.onrender.com/docs`
- **Health Check**: `https://your-service-name.onrender.com/health`

## Troubleshooting

### Build Fails
- Check that `requirements.txt` is in the `knn-app/` directory
- Ensure Python 3.10+ is specified
- Check Render deployment logs for specific errors

### Application Won't Start
- Verify the start command includes `--host 0.0.0.0` and uses `$PORT`
- Check logs in Render dashboard for startup errors
- Ensure all dependencies are correctly specified in `requirements.txt`

### Missing Dataset
The `Classified Data` file must be committed to the repository. It should be located at:
```
customer-intelligence-platform/knn-app/Classified Data
```

If the dataset is missing, the application will fail at startup with a FileNotFoundError.

## Local Testing

Before deploying, test locally:

```bash
cd knn-app
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://localhost:8000/` in your browser.

## Resources

- Render Documentation: https://render.com/docs
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Uvicorn Documentation: https://www.uvicorn.org/
