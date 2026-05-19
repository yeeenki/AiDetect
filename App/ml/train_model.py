from App.ml.text_detect import load_dataset, run_statistical_analysis, train_ml_model
import joblib

df = load_dataset("App\ml\detection_data.json")
feat_df = run_statistical_analysis(df)

model, tfidf, feat_cols, _ = train_ml_model(df, feat_df)

joblib.dump(model, "app/ml/model.pkl")
joblib.dump(tfidf, "app/ml/tfidf.pkl")
joblib.dump(feat_cols, "app/ml/feat_cols.pkl")