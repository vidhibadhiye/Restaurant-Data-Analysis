"""Restaurant Rating Prediction System - Streamlit dashboard.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import utils as u

st.set_page_config(page_title="Restaurant Rating Prediction System", page_icon="🍽️", layout="wide")

ACCENT = "#E23744"  # Zomato-style red
st.markdown(
    f"""
    <style>
      div[data-testid="stMetric"] {{
        background: #fff7f7; border: 1px solid #f3d3d6; border-left: 6px solid {ACCENT};
        padding: 12px 16px; border-radius: 10px;
      }}
      div[data-testid="stMetric"] * {{ color: #2b2b2b !important; }}
      h1, h2, h3 {{ color: {ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)
PLOT_COLORS = [ACCENT, "#2B6CB0", "#38A169", "#D69E2E", "#805AD5", "#718096"]


# ------------------------------------------------------------------ cached loading
@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    raw = u.load_raw_data()
    return raw, u.build_restaurant_df(raw)


@st.cache_resource(show_spinner="Loading saved model...")
def get_artifacts():
    return u.load_artifacts()


try:
    raw_df, rdf = get_data()
except FileNotFoundError as exc:
    st.error(f"{exc}. Place the CSV inside the `data/` folder and restart.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"The dataset could not be read: {exc}")
    st.stop()

artifacts, artifact_errors = get_artifacts()
model_ready = not artifact_errors and all(k in artifacts for k in ("model", "scaler", "name"))

# ------------------------------------------------------------------------ sidebar
st.sidebar.title("🍽️ Restaurant Analytics")
page = st.sidebar.radio("Navigate", ["Dashboard", "Data Analysis", "Model Performance", "Prediction", "Insights"])
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(raw_df):,} menu-item rows → **{len(rdf):,} restaurants** "
    "(one per restaurant + city, as in the notebook)."
)
if model_ready:
    st.sidebar.success(f"Saved model loaded: {artifacts['name']}")
else:
    st.sidebar.error("Saved model not available")


def fig_style(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    return fig


# ---------------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("Restaurant Rating Prediction System")
    st.caption("Built from the Statistical Inference lab project: Zomato restaurant data, regression models "
               f"and the saved model for predicting `{u.TARGET}`.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total restaurants", f"{len(rdf):,}")
    c2.metric("Average rating", f"{rdf[u.TARGET].mean():.2f} / 5")
    c3.metric("Average price", f"₹{rdf['Avg_Price_Restaurant'].mean():,.0f}")
    c4.metric("Median rating", f"{rdf[u.TARGET].median():.2f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Cities", rdf["City"].nunique())
    c6.metric("Cuisines", rdf["Cuisine"].nunique())
    c7.metric("Avg dining votes", f"{rdf['Dining_Votes'].mean():,.0f}")
    c8.metric("Rated 4.0 or above", f"{(rdf[u.TARGET] >= 4).mean():.0%}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(rdf, x=u.TARGET, nbins=25, title="Distribution of restaurant ratings",
                           color_discrete_sequence=[ACCENT])
        fig.update_layout(xaxis_title="Average restaurant rating", yaxis_title="Restaurants")
        st.plotly_chart(fig_style(fig))
    with right:
        top = rdf["Cuisine"].value_counts().head(8).reset_index()
        top.columns = ["Cuisine", "Restaurants"]
        fig = px.bar(top, x="Restaurants", y="Cuisine", orientation="h", title="Most common cuisines",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_style(fig))
    st.info("Use the sidebar to explore the data, see model performance, or predict a restaurant's rating.")

# ----------------------------------------------------------------- Data Analysis
elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")

    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        rmin, rmax = float(rdf[u.TARGET].min()), float(rdf[u.TARGET].max())
        pmin, pmax = float(rdf["Avg_Price_Restaurant"].min()), float(rdf["Avg_Price_Restaurant"].max())
        rating_range = f1.slider("Restaurant rating", rmin, rmax, (rmin, rmax), step=0.05)
        price_range = f2.slider("Average price (₹)", float(np.floor(pmin)), float(np.ceil(pmax)),
                                (float(np.floor(pmin)), float(np.ceil(pmax))))
        f3, f4 = st.columns(2)
        cities = f3.multiselect("City", sorted(rdf["City"].unique()))
        cuisines = f4.multiselect("Cuisine", sorted(rdf["Cuisine"].unique()))

    view = rdf[rdf[u.TARGET].between(*rating_range) & rdf["Avg_Price_Restaurant"].between(*price_range)]
    if cities:
        view = view[view["City"].isin(cities)]
    if cuisines:
        view = view[view["Cuisine"].isin(cuisines)]
    st.write(f"Showing **{len(view):,}** of {len(rdf):,} restaurants.")
    if view.empty:
        st.warning("No restaurants match these filters. Widen the filters to see charts.")
        st.stop()

    t1, t2, t3 = st.tabs(["Data quality", "Distributions", "Relationships"])

    with t1:
        show = ["Restaurant_Name", "City", "Cuisine", "Place_Name", "Dining_Rating", "Delivery_Rating",
                "Dining_Votes", "Delivery_Votes", "Avg_Price_Restaurant", u.TARGET]
        st.subheader("Dataset preview (filtered restaurants)")
        st.dataframe(view[show].head(100))
        st.subheader("Missing values and duplicates")
        a, b, c = st.columns(3)
        a.metric("Missing values (raw CSV)", int(raw_df.isna().sum().sum()))
        b.metric("Duplicate rows (raw CSV)", f"{int(raw_df.duplicated().sum()):,}")
        c.metric("Rows after de-duplication", f"{len(raw_df.drop_duplicates()):,}")
        miss = raw_df.isna().sum().reset_index()
        miss.columns = ["Column", "Missing values"]
        st.dataframe(miss, height=250)
        st.caption("The notebook drops duplicate rows, then keeps one row per restaurant and city "
                   f"({len(rdf):,} restaurants). All charts here use that restaurant-level table.")
        st.subheader("Summary statistics")
        st.dataframe(view[u.FEATURES + [u.TARGET]].describe().T.round(3))

    with t2:
        a, b = st.columns(2)
        fig = px.histogram(view, x=u.TARGET, nbins=25, marginal="box", title="Rating distribution",
                           color_discrete_sequence=[ACCENT])
        a.plotly_chart(fig_style(fig))
        fig = px.histogram(view, x="Avg_Price_Restaurant", nbins=40, marginal="box", title="Price distribution",
                           color_discrete_sequence=["#2B6CB0"])
        b.plotly_chart(fig_style(fig))
        st.caption("Ratings cluster around 3.8–4.0 and prices are skewed right: a few expensive "
                   "restaurants pull the average above the median.")

    with t3:
        a, b = st.columns(2)
        fig = px.scatter(view, x="Avg_Price_Restaurant", y=u.TARGET, color="City", opacity=0.7,
                         hover_data=["Restaurant_Name", "Cuisine"], title="Price vs rating",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        a.plotly_chart(fig_style(fig))
        fig = px.scatter(view, x="Dining_Rating", y=u.TARGET, opacity=0.6, color="Delivery_Rating",
                         hover_data=["Restaurant_Name"], title="Dining rating vs restaurant rating",
                         color_continuous_scale="Reds")
        b.plotly_chart(fig_style(fig))
        top_c = view["Cuisine"].value_counts().head(10).index
        fig = px.box(view[view["Cuisine"].isin(top_c)], x="Cuisine", y=u.TARGET,
                     title="Rating by the 10 most common cuisines (in this selection)",
                     color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig_style(fig))
        corr = view[u.FEATURES + [u.TARGET]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation matrix (model features + target)")
        st.plotly_chart(fig_style(fig))
        st.caption("Red = positive correlation, blue = negative. Dining and delivery ratings move "
                   "closely with the restaurant rating; price and vote counts show much weaker links.")

# ------------------------------------------------------------- Model Performance
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        f"**Target:** `{u.TARGET}` &nbsp;|&nbsp; **Features (in order):** "
        + ", ".join(f"`{f}`" for f in u.FEATURES)
        + " &nbsp;|&nbsp; **Split:** 80% train / 20% test, `random_state=42` (from the notebook)."
    )

    st.subheader("1. Results saved in the notebook (all five models)")
    nb_metrics = u.notebook_saved_metrics()
    if nb_metrics is None:
        st.warning("Could not find the model-comparison output inside `backend/SI_PROJECT.ipynb`, "
                   "so no saved metrics can be shown.")
    else:
        st.dataframe(nb_metrics.style.format({"R2": "{:.4f}", "MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}),
                     hide_index=True)
        st.caption("Source: printed output of the notebook cell 'Model Comparison' (test set, 182 restaurants). "
                   "These numbers are read from your notebook, not recalculated.")
        fig = px.bar(nb_metrics, x="Model", y="R2", title="R² by model (notebook results)",
                     color_discrete_sequence=[ACCENT], text_auto=".4f")
        fig.update_yaxes(range=[0.9, 0.96])
        st.plotly_chart(fig_style(fig))
        st.caption("The y-axis starts at 0.90 to make small differences visible; all models score similarly.")

    st.subheader("2. Newly calculated with the saved model")
    if not model_ready:
        st.error("The saved model could not be loaded, so nothing can be recalculated:\n\n- "
                 + "\n- ".join(artifact_errors or ["Unknown problem."]))
    else:
        try:
            m, n_test, y_true, y_pred = u.recompute_saved_model_metrics(artifacts, rdf)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not evaluate the saved model: {exc}")
        else:
            st.caption(f"Only **{artifacts['name']}** has a `.pkl` file, so only it can be re-evaluated. "
                       "The other four models were not saved; they are not retrained here. "
                       f"Evaluation uses the notebook's test split ({n_test} restaurants).")
            cols = st.columns(4)
            for col, key in zip(cols, ["R2", "MAE", "MSE", "RMSE"]):
                col.metric(key if key != "R2" else "R²", f"{m[key]:.4f}")
            if nb_metrics is not None:
                row = nb_metrics[nb_metrics["Model"] == artifacts["name"]]
                if not row.empty:
                    r = row.iloc[0]
                    same = all(abs(round(m[k], 4) - r[k]) < 1e-4 for k in ["R2", "MAE", "MSE", "RMSE"])
                    (st.success if same else st.warning)(
                        "Recalculated values match the notebook output." if same else
                        "Recalculated values differ from the notebook output; check the file versions.")
            fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual rating", "y": "Predicted rating"},
                             title="Actual vs predicted (test set)", color_discrete_sequence=[ACCENT])
            lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_style(fig))

            if hasattr(artifacts["model"], "coef_"):
                coef = pd.DataFrame({"Feature": u.FEATURES, "Coefficient (standardised inputs)": artifacts["model"].coef_})
                st.markdown("**Ridge coefficients** (inputs are standardised, so sizes are comparable)")
                st.dataframe(coef.style.format({"Coefficient (standardised inputs)": "{:.4f}"}),
                             hide_index=True)

    st.subheader("How to read these results")
    st.warning(
        "The target `Avg_Rating_Restaurant` equals the simple average of `Dining_Rating` and "
        f"`Delivery_Rating` for {np.isclose((rdf.Dining_Rating + rdf.Delivery_Rating) / 2, rdf[u.TARGET], atol=1e-6).mean():.0%} "
        "of restaurants. Because those two ratings are model inputs, the high R² mostly reflects this "
        "built-in relationship, not an ability to predict ratings from price or votes. Results are from "
        "one train/test split on this dataset only; no external validation was done."
    )
    if artifacts.get("version_warnings"):
        with st.expander("scikit-learn version note"):
            st.write("\n".join(f"- {w}" for w in artifacts["version_warnings"]))
            st.caption("The .pkl files were saved with scikit-learn 1.6.1. Installing that version "
                       "(`pip install scikit-learn==1.6.1`) removes this notice.")

# ---------------------------------------------------------------------- Prediction
elif page == "Prediction":
    st.title("Restaurant Rating Prediction")
    if not model_ready:
        st.error("Prediction is unavailable because required model files are missing or incompatible:\n\n- "
                 + "\n- ".join(artifact_errors or ["Unknown problem."]))
        st.stop()

    st.write(f"Model: **{artifacts['name']}** (loaded from `backend/best_restaurant_model.pkl`). "
             "It uses these five inputs, in the same order as the notebook.")
    with st.form("predict_form"):
        a, b = st.columns(2)
        dining_rating = a.number_input("Dining rating (0–5)", 0.0, 5.0, 4.0, 0.1)
        delivery_rating = b.number_input("Delivery rating (0–5)", 0.0, 5.0, 4.1, 0.1)
        c, d = st.columns(2)
        dining_votes = c.number_input("Dining votes", min_value=0, value=100, step=10)
        delivery_votes = d.number_input("Delivery votes", min_value=0, value=80, step=10)
        price = st.number_input("Average restaurant price (₹)", min_value=0.0, value=500.0, step=25.0)
        submitted = st.form_submit_button("Predict rating", type="primary")

    if submitted:
        values = {"Dining_Rating": dining_rating, "Delivery_Rating": delivery_rating,
                  "Dining_Votes": dining_votes, "Delivery_Votes": delivery_votes,
                  "Avg_Price_Restaurant": price}
        errs, warns = u.validate_inputs(values, rdf)
        if errs:
            for e in errs:
                st.error(e)
        else:
            try:
                pred = u.predict_rating(artifacts, values)
            except Exception as exc:  # noqa: BLE001
                st.error(f"The model could not make a prediction: {exc}")
            else:
                for w in warns:
                    st.warning(w)
                shown = float(np.clip(pred, 0, 5))
                st.metric("Predicted restaurant rating", f"{shown:.2f} / 5")
                if shown != pred:
                    st.caption(f"The raw model output ({pred:.2f}) was outside 0–5 and is shown capped to that range.")
                avg = rdf[u.TARGET].mean()
                st.caption(f"Dataset average: {avg:.2f}. The prediction is {abs(shown - avg):.2f} "
                           f"{'above' if shown >= avg else 'below'} average.")
                st.caption(f"For reference, the plain average of the two ratings entered is "
                           f"{(dining_rating + delivery_rating) / 2:.2f}. See the note on the Model Performance page.")
    else:
        st.info("Enter the restaurant details and press **Predict rating**.")

# ------------------------------------------------------------------------ Insights
elif page == "Insights":
    st.title("Insights")
    st.caption("All statements below are calculated from the dataset when the page loads.")

    corr = rdf[u.FEATURES + [u.TARGET]].corr()[u.TARGET].drop(u.TARGET).sort_values(ascending=False)
    cdf = corr.reset_index()
    cdf.columns = ["Feature", "Correlation with rating"]
    fig = px.bar(cdf, x="Correlation with rating", y="Feature", orientation="h",
                 title="Correlation of each model feature with the restaurant rating",
                 color_discrete_sequence=[ACCENT])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_style(fig))
    st.markdown(
        f"- **Dining rating** (r = {corr['Dining_Rating']:.2f}) and **delivery rating** "
        f"(r = {corr['Delivery_Rating']:.2f}) have the strongest links with the restaurant rating.\n"
        f"- **Price** (r = {corr['Avg_Price_Restaurant']:.2f}) and **vote counts** "
        f"(dining {corr['Dining_Votes']:.2f}, delivery {corr['Delivery_Votes']:.2f}) show weak links.\n"
        "- Correlation shows association, not cause."
    )

    city = (rdf.groupby("City")[u.TARGET].agg(["mean", "count"]).query("count >= 10")
            .sort_values("mean", ascending=False).reset_index())
    fig = px.bar(city, x="City", y="mean", title="Average rating by city (cities with 10+ restaurants)",
                 color_discrete_sequence=[ACCENT], hover_data=["count"])
    fig.update_yaxes(title="Average rating", range=[3.5, 4.2])
    st.plotly_chart(fig_style(fig))
    st.markdown(f"- Highest: **{city.iloc[0]['City']}** ({city.iloc[0]['mean']:.2f}); lowest: "
                f"**{city.iloc[-1]['City']}** ({city.iloc[-1]['mean']:.2f}). Differences between cities are small. "
                "The y-axis is zoomed in.")

    cu = (rdf.groupby("Cuisine")[u.TARGET].agg(["mean", "count"]).query("count >= 10")
          .sort_values("mean", ascending=False).reset_index())
    fig = px.bar(cu.head(10), x="mean", y="Cuisine", orientation="h", hover_data=["count"],
                 title="Highest-rated cuisines (10+ restaurants)", color_discrete_sequence=[ACCENT])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(title="Average rating", range=[3.5, 4.2])
    st.plotly_chart(fig_style(fig))

    bands = pd.cut(rdf["Avg_Price_Restaurant"], [0, 150, 250, 400, np.inf],
                   labels=["Under ₹150", "₹150–250", "₹250–400", "Over ₹400"])
    pb = rdf.groupby(bands, observed=True)[u.TARGET].mean().reset_index()
    pb.columns = ["Price band", "Average rating"]
    fig = px.bar(pb, x="Price band", y="Average rating", title="Average rating by price band",
                 color_discrete_sequence=[ACCENT], text_auto=".2f")
    fig.update_yaxes(range=[3.5, 4.2])
    st.plotly_chart(fig_style(fig))
    st.markdown(f"- Moving from the cheapest to the most expensive band changes the average rating by "
                f"{pb.iloc[-1, 1] - pb.iloc[0, 1]:+.2f}, which is small compared with the effect of the dining and delivery ratings.")

    match = np.isclose((rdf.Dining_Rating + rdf.Delivery_Rating) / 2, rdf[u.TARGET], atol=1e-6).mean()
    st.warning(f"**Important limitation:** for {match:.0%} of restaurants the target equals the average of "
               "dining and delivery rating. The model is therefore largely re-creating that average, "
               "so its high R² should not be read as proof that price or popularity predict ratings.")