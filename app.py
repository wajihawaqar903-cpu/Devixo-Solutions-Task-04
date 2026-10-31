from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

CURRENCY = "$"  # change if your dataset uses another currency
BASE_DIR = Path(__file__).parent  # the folder where app.py is saved


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "best_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_cols = joblib.load(BASE_DIR / "feature_columns.pkl")
    meta = joblib.load(BASE_DIR / "metadata.pkl")
    return model, scaler, feature_cols, meta


def age_group(age):
    if age <= 10:
        return "New"
    if age <= 30:
        return "Modern"
    if age <= 60:
        return "Old"
    return "Very Old"


def engineer_features(inp, ref_year):
    """Same formulas as the notebook's feature engineering step."""
    house_age = ref_year - inp["YearBuilt"]
    return {
        "Area": inp["Area"],
        "Bedrooms": inp["Bedrooms"],
        "Bathrooms": inp["Bathrooms"],
        "Floors": inp["Floors"],
        "HouseAge": house_age,
        "TotalRooms": inp["Bedrooms"] + inp["Bathrooms"],
        "AreaPerBedroom": inp["Area"] / (inp["Bedrooms"] + 1),
        "BathBedRatio": inp["Bathrooms"] / (inp["Bedrooms"] + 1),
        "AreaPerFloor": inp["Area"] / max(inp["Floors"], 1),
        "IsNew": int(house_age <= 10),
        "AgeGroup": age_group(house_age),
    }


def build_model_input(inp, feats, feature_cols, scaler):
    """Encode and scale the input exactly like the training data."""
    X = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    # numeric features
    for col, val in feats.items():
        if col in X.columns:
            X.at[0, col] = val

    # one-hot columns (training used drop_first, so a missing column means the base category)
    categorical = {
        "Location": inp["Location"],
        "Condition": inp["Condition"],
        "Garage": inp["Garage"],
        "AgeGroup": feats["AgeGroup"],
    }
    for col, val in categorical.items():
        dummy = f"{col}_{val}"
        if dummy in X.columns:
            X.at[0, dummy] = 1.0

    # scale the same columns the scaler was fitted on
    scale_cols = list(scaler.feature_names_in_)
    X[scale_cols] = scaler.transform(X[scale_cols])
    return X


model, scaler, feature_cols, meta = load_artifacts()
r = meta["ranges"]
cats = meta["categories"]
stats = meta["price_stats"]

# ---------- Header ----------
st.title("🏠 House Price Predictor")
st.write("Enter the details of a house and click **Predict Price** to get an estimated price.")

# ---------- Inputs ----------
with st.form("house_form"):
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area", min_value=float(r["Area"][0]), max_value=float(r["Area"][1] * 1.5),
                               value=float(r["Area"][2]), step=50.0)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=max(r["Bedrooms"][1], 10),
                                   value=min(max(3, r["Bedrooms"][0]), r["Bedrooms"][1]), step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=max(r["Bathrooms"][1], 10),
                                    value=min(max(2, r["Bathrooms"][0]), r["Bathrooms"][1]), step=1)
        floors = st.number_input("Floors", min_value=1, max_value=max(r["Floors"][1], 5),
                                 value=min(max(1, r["Floors"][0]), r["Floors"][1]), step=1)
    with c2:
        year_built = st.number_input("Year Built", min_value=r["YearBuilt"][0],
                                     max_value=meta["reference_year"],
                                     value=min(2010, meta["reference_year"]), step=1)
        location = st.selectbox("Location", cats["Location"])
        condition = st.selectbox("Condition", cats["Condition"])
        garage = st.selectbox("Garage", cats["Garage"])

    submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    inp = {
        "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Floors": floors,
        "YearBuilt": year_built, "Location": location, "Condition": condition, "Garage": garage,
    }
    feats = engineer_features(inp, meta["reference_year"])
    X_input = build_model_input(inp, feats, feature_cols, scaler)
    price = float(model.predict(X_input)[0])

    st.divider()
    st.subheader("Prediction")
    st.metric("Estimated Price", f"{CURRENCY}{price:,.0f}")

    mae = meta["test_metrics"]["MAE"]
    st.caption(f"Typical error of this model: about ±{CURRENCY}{mae:,.0f} "
               f"(range {CURRENCY}{max(price - mae, 0):,.0f} to {CURRENCY}{price + mae:,.0f})")

    # ---------- Relevant information ----------
    st.subheader("About this prediction")

    diff_pct = (price - stats["mean"]) / stats["mean"] * 100
    pct_rank = int(np.clip(np.searchsorted(meta["price_percentiles"], price), 0, 100))

    k1, k2, k3 = st.columns(3)
    k1.metric("Dataset average price", f"{CURRENCY}{stats['mean']:,.0f}", f"{diff_pct:+.1f}% vs average")
    k2.metric("Price percentile", f"{pct_rank}th", help="Share of houses in the dataset priced below this estimate")
    k3.metric("Price per unit area", f"{CURRENCY}{price / area:,.2f}")

    if price < stats["min"] or price > stats["max"]:
        st.warning("This estimate is outside the price range seen in the training data, so treat it with extra caution.")

    with st.expander("Your input and the features the model used", expanded=True):
        shown = pd.DataFrame({
            "Feature": ["Area", "Bedrooms", "Bathrooms", "Floors", "Year Built", "Location", "Condition", "Garage",
                        "House Age (years)", "Total Rooms", "Area per Bedroom", "Bath/Bed Ratio",
                        "Area per Floor", "Age Group"],
            "Value": [area, bedrooms, bathrooms, floors, year_built, location, condition, garage,
                      feats["HouseAge"], feats["TotalRooms"], round(feats["AreaPerBedroom"], 2),
                      round(feats["BathBedRatio"], 2), round(feats["AreaPerFloor"], 2), feats["AgeGroup"]],
        })
        shown["Value"] = shown["Value"].astype(str)
        st.dataframe(shown, hide_index=True, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        with st.expander("What influences the price most?"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(8)
            st.bar_chart(imp)
            st.caption("Top features by importance in the trained model.")

    with st.expander("Model information"):
        m = meta["test_metrics"]
        st.write(f"**Model:** {meta['model_name']}")
        st.write(f"**R² (test):** {m['R2']:.4f}  |  **MAE:** {m['MAE']:,.2f}  |  **RMSE:** {m['RMSE']:,.2f}")
        st.write("Predictions are estimates based on historical data and are not a professional valuation.")