# Restaurant Rating Prediction System (Streamlit dashboard)

Interactive dashboard built on the Statistical Inference lab project (Zomato restaurant data).

## Project structure
```
restaurant-rating-dashboard/
    app.py                  # Streamlit app (5 pages)
    utils.py                # data loading, model loading, metrics, prediction
    requirements.txt
    README.md
    data/
        enhanced_zomato_dataset_clean.csv
    backend/
        SI_PROJECT.ipynb
        best_restaurant_model.pkl        # Ridge Regression (best model in notebook)
        best_restaurant_model_name.pkl   # "Ridge Regression"
        restaurant_scaler.pkl            # StandardScaler fitted on the 5 features
```

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## What the project files contain
- **Target:** `Avg_Rating_Restaurant`
- **Features (exact order):** `Dining_Rating, Delivery_Rating, Dining_Votes, Delivery_Votes, Avg_Price_Restaurant`
- **Data:** the notebook drops duplicate rows, then keeps one row per (`Restaurant_Name`, `City`) → 907 restaurants. The dashboard repeats these steps without changing the CSV.
- **Saved model:** only the best model (Ridge Regression, alpha=0.1) was saved. Linear, Lasso, Decision Tree and Random Forest were **not** saved as `.pkl`, so their metrics are shown from the notebook's printed output and are not recalculated or retrained.
- **Preprocessing:** Ridge uses the saved `StandardScaler`; tree-based models would use raw values (as in the notebook).

## Notes for viva
- Saved vs recalculated metrics are labelled separately on the Model Performance page. The recalculated Ridge metrics (R² 0.9501, MAE 0.0244, MSE 0.0027, RMSE 0.0522) match the notebook's output.
- `Avg_Rating_Restaurant` equals the average of `Dining_Rating` and `Delivery_Rating` for about 87% of restaurants. Since both are model inputs, the high R² mostly reflects that relationship. The dashboard states this limitation.
- The `.pkl` files were saved with scikit-learn 1.6.1. Newer versions load them but print a version notice; `pip install scikit-learn==1.6.1` removes it.
- Original notebook, CSV and `.pkl` files are untouched; the app only reads them.
