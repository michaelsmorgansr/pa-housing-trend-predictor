import streamlit as st
import pandas as pd
import joblib
import numpy as np

# PAGE CONFIG
st.set_page_config(page_title="PA Housing Trend Predictor", layout="wide")

st.title("Pennsylvania ZIP Housing Trend Predictor")
st.write(
    "This app predicts 12-month % change in single-family Zillow Home Value Index (ZHVI) "
    "for Pennsylvania ZIP codes using housing, socioeconomic, environmental, and flood-related features."
)

# LOAD DATA + MODEL
@st.cache_data
def load_data():
    df = pd.read_csv("pa_modeling_w_acs_tri_flood.csv", dtype={"zcta": str})
    df["zcta"] = df["zcta"].astype(str).str.zfill(5)
    return df

@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model_with_tri_fema.joblib")

df = load_data()
model = load_model()

# FEATURES USED BY FINAL MODEL
feature_cols = [
    "zhvi_2026_03",
    "zhvi_2025_03",
    "pct_chg_1m",
    "pct_chg_3m",
    "pct_chg_6m",
    "vol_12m",
    "median_household_income",
    "pct_bachelor_or_higher",
    "pct_owner_occupied",
    "population_total",
    "tri_facility_count_county",
    "tri_total_releases_county",
    "flood_zone_count"
]

target_col = "y_pct_chg_12m"

# USER-FRIENDLY FEATURE LABELS
feature_labels = {
    "zhvi_2026_03": "Average home value (Mar 2026)",
    "zhvi_2025_03": "Average home value (Mar 2025)",
    "pct_chg_1m": "1-month home value change",
    "pct_chg_3m": "3-month home value change",
    "pct_chg_6m": "6-month home value change",
    "vol_12m": "12-month volatility",
    "median_household_income": "Median household income",
    "pct_bachelor_or_higher": "Bachelor's degree or higher",
    "pct_owner_occupied": "Owner-occupied housing",
    "population_total": "Population",
    "tri_facility_count_county": "Industrial facilities reporting (county)",
    "tri_total_releases_county": "Industrial chemical releases (county)",
    "flood_zone_count": "Mapped flood hazard zones (county)"
}

# FORMAT FEATURE VALUES
def format_feature_value(feature, value):
    if pd.isna(value):
        return "Not available"

    dollar_features = {
        "zhvi_2026_03",
        "zhvi_2025_03",
        "median_household_income"
    }

    percent_features = {
        "pct_chg_1m",
        "pct_chg_3m",
        "pct_chg_6m",
        "pct_bachelor_or_higher",
        "pct_owner_occupied",
        "vol_12m"
    }

    integer_features = {
        "population_total",
        "tri_facility_count_county",
        "flood_zone_count"
    }

    if feature in dollar_features:
        return f"${value:,.0f}"
    elif feature in percent_features:
        return f"{value:.2f}%"
    elif feature == "tri_total_releases_county":
        return f"{value:,.0f} lbs"
    elif feature in integer_features:
        return f"{int(value):,}"
    else:
        return f"{value:,.2f}"

# ZIP SELECTOR
zip_list = sorted(df["zcta"].dropna().unique().tolist())

default_index = 0
if "18109" in zip_list:
    default_index = zip_list.index("18109")

selected_zip = st.selectbox("Select a Pennsylvania ZIP code", zip_list, index=default_index)

# GET ZIP ROW
zip_df = df[df["zcta"] == selected_zip].copy()

if zip_df.empty:
    st.error(f"No data found for ZIP {selected_zip}")
else:
    row = zip_df.iloc[0]

    X_input = zip_df[feature_cols]
    prediction = model.predict(X_input)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")
        st.metric("Predicted 12-month % change", f"{prediction:.2f}%")

        if target_col in zip_df.columns and pd.notna(row[target_col]):
            st.metric("Actual 12-month % change", f"{row[target_col]:.2f}%")

        st.write("#### ZIP Details")
        st.write(f"**ZIP:** {row['zcta']}")
        st.write(f"**County:** {row['CountyName']}")
        st.write(f"**Metro:** {row['Metro']}")

        st.write("### Environmental Context")

        release_val = row["tri_total_releases_county"]
        facility_count = row["tri_facility_count_county"]

        if pd.notna(release_val):
            release_millions = release_val / 1_000_000

            if release_val >= 1_000_000:
                exposure_score = "High"
            elif release_val >= 250_000:
                exposure_score = "Moderate"
            else:
                exposure_score = "Low"

            st.write(f"**Exposure score (county):** {exposure_score}")
            st.write(f"**Industrial facilities reporting (county):** {int(facility_count)}")
            st.write(
                f"**Industrial chemical releases (county, annual):** "
                f"~{release_millions:.2f} million pounds"
            )
        else:
            st.write("**Exposure score (county):** Not available")

        if pd.notna(row["flood_zone_count"]):
            st.write(f"**Mapped flood hazard zones (county):** {int(row['flood_zone_count'])}")
        else:
            st.write("**Mapped flood hazard zones (county):** Not available")

        st.write(
            "This app uses county-level EPA TRI and FEMA-derived flood context as simplified "
            "environmental and hazard indicators. These values provide context rather than direct "
            "property-level risk estimates."
        )

    with col2:
        st.subheader("Feature Values")

        feature_display = pd.DataFrame({
            "Feature": [feature_labels.get(col, col) for col in feature_cols],
            "Value": [format_feature_value(col, row[col]) for col in feature_cols]
        })

        st.dataframe(feature_display, use_container_width=True, hide_index=True)

    st.write("### Summary")
    st.write(
        f"For ZIP **{row['zcta']}**, the model predicts a **{prediction:.2f}%** "
        f"change in single-family ZHVI over 12 months."
    )