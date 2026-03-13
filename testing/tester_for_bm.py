import joblib

# Load the model that was saved right before the crash
model_path = 'models/best_tuned_pipeline.pkl'
try:
    model = joblib.load(model_path)
    print("Success! The model is loaded and ready for use.")
except Exception as e:
    print(f"Error loading model: {e}")