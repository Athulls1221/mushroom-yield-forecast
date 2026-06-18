import joblib

best_model = joblib.load("models/random_forest.joblib")

joblib.dump(best_model, "models/champion.joblib")

print("Champion model saved.")