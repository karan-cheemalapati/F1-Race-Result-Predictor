import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train import FEATURE_COLS

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Result Predictor",
    layout="wide"
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E85D24;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .predict-result {
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost.pkl')

@st.cache_data
def load_data():
    df           = pd.read_csv('data/f1_features.csv')
    return df

model = load_model()
df    = load_data()

# ── helper: build input row ───────────────────────────────────────────────────
def build_input(df, driver, constructor, circuit, year, grid,
                quali_gap, driver_standing, constructor_standing):
    driver_data = df[df['driverRef'] == driver]
    cons_data   = df[df['constructorRef'] == constructor]
    circ_data   = df[(df['driverRef'] == driver) & (df['circuitRef'] == circuit)]

    def safe_mean(series, default=0):
        return series.mean() if len(series) > 0 else default

    row = {
        'grid':                       grid,
        'driver_age':                 safe_mean(driver_data['driver_age'], 28),
        'driver_experience':          safe_mean(driver_data['driver_experience'], 50),
        'driver_standing_pts':        safe_mean(driver_data[driver_data['driver_standing_pos'] == driver_standing]['driver_standing_pts'], 100),
        'driver_standing_pos':        driver_standing,
        'driver_wins_so_far':         safe_mean(driver_data['driver_wins_so_far'], 0),
        'constructor_standing_pts':   safe_mean(cons_data[cons_data['constructor_standing_pos'] == constructor_standing]['constructor_standing_pts'], 100),
        'constructor_standing_pos':   constructor_standing,
        'constructor_wins_so_far':    safe_mean(cons_data['constructor_wins_so_far'], 0),
        'quali_gap_to_pole':          quali_gap,
        'quali_position':             grid,
        'circuit_win_rate':           safe_mean(circ_data['circuit_win_rate'], 0),
        'circuit_podium_rate':        safe_mean(circ_data['circuit_podium_rate'], 0),
        'circuit_avg_finish':         safe_mean(circ_data['circuit_avg_finish'], 10),
        'circuit_races_count':        len(circ_data),
        'driver_avg_finish_last3':    safe_mean(driver_data['driver_avg_finish_last3'], 10),
        'driver_avg_finish_last5':    safe_mean(driver_data['driver_avg_finish_last5'], 10),
        'constructor_avg_pts_last3':  safe_mean(cons_data['constructor_avg_pts_last3'], 10),
        'constructor_avg_pts_last5':  safe_mean(cons_data['constructor_avg_pts_last5'], 10),
        'grid_vs_quali':              0,
        'season_progress':            0.5,
        'pts_gap_to_leader':          safe_mean(driver_data[driver_data['driver_standing_pos'] == driver_standing]['pts_gap_to_leader'], 50),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Race Parameters")
st.sidebar.markdown("Configure the race and click Predict.")

driver_list      = sorted(df['driverRef'].dropna().unique().tolist())
constructor_list = sorted(df['constructorRef'].dropna().unique().tolist())
circuit_list     = sorted(df['circuitRef'].dropna().unique().tolist())

selected_driver      = st.sidebar.selectbox("Driver", driver_list, index=driver_list.index('hamilton') if 'hamilton' in driver_list else 0)
selected_constructor = st.sidebar.selectbox("Constructor", constructor_list, index=constructor_list.index('mercedes') if 'mercedes' in constructor_list else 0)
selected_circuit     = st.sidebar.selectbox("Circuit", circuit_list, index=circuit_list.index('monza') if 'monza' in circuit_list else 0)
selected_year        = st.sidebar.slider("Season", min_value=1994, max_value=2024, value=2023)
grid_pos             = st.sidebar.slider("Grid Position", min_value=1, max_value=20, value=1)
quali_gap            = st.sidebar.slider("Qualifying Gap to Pole (s)", min_value=0.0, max_value=3.0, value=0.0, step=0.01)
driver_standing      = st.sidebar.slider("Driver Championship Standing", min_value=1, max_value=20, value=1)
constructor_standing = st.sidebar.slider("Constructor Championship Standing", min_value=1, max_value=10, value=1)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Predict Podium", use_container_width=True)

# ── main header ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">F1 Race Result Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicting podium finishes using machine learning on 70+ years of F1 data (1950–2024)</p>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Races",   f"{df['raceId'].nunique():,}")
col2.metric("Total Drivers", f"{df['driverId'].nunique():,}")
col3.metric("Seasons",       f"{df['year'].nunique()}")
col4.metric("Model",         "XGBoost  |  AUC 0.940")
st.markdown("---")

# ── prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    input_df   = build_input(df, selected_driver, selected_constructor, selected_circuit,
                              selected_year, grid_pos, quali_gap, driver_standing, constructor_standing)
    prob       = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    # ── result banner ──
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction == 1:
            st.success("### PODIUM FINISH PREDICTED")
        else:
            st.error("### NO PODIUM PREDICTED")
    with col2:
        st.metric("Podium Probability", f"{prob:.1%}")
    with col3:
        st.metric("Grid Position", f"P{grid_pos}")

    # probability bar
    prob_color = "#1D9E75" if prob >= 0.5 else "#E85D24"
    st.markdown(
        f"""
        <div style="background:#e0e0e0;border-radius:10px;height:30px;width:100%;margin-bottom:1rem;">
            <div style="background:{prob_color};width:{prob*100:.1f}%;height:30px;
                        border-radius:10px;text-align:center;line-height:30px;
                        color:white;font-weight:bold;">{prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    # ── chart 1: driver circuit history ──
    st.subheader(f"{selected_driver}'s Circuit History")

    driver_circuit = df[(df['driverRef'] == selected_driver) & (df['circuitRef'] == selected_circuit)]
    driver_all     = df[df['driverRef'] == selected_driver]

    circuit_podium_rate = driver_circuit['podium'].mean() if len(driver_circuit) > 0 else 0
    overall_podium_rate = driver_all['podium'].mean()     if len(driver_all) > 0 else 0
    circuit_races       = len(driver_circuit)
    circuit_wins        = (driver_circuit['positionOrder'] == 1).sum() if len(driver_circuit) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Races at Circuit",       circuit_races)
    col2.metric("Wins at Circuit",        circuit_wins)
    col3.metric("Podium Rate at Circuit", f"{circuit_podium_rate:.1%}")
    col4.metric("Overall Podium Rate",    f"{overall_podium_rate:.1%}")

    if len(driver_circuit) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # finish positions at this circuit over years
        axes[0].bar(driver_circuit['year'], driver_circuit['positionOrder'],
                    color=['#E85D24' if p <= 3 else '#378ADD' for p in driver_circuit['positionOrder']],
                    edgecolor='white')
        axes[0].axhline(y=3.5, color='gold', linestyle='--', linewidth=1.5, label='Podium line')
        axes[0].invert_yaxis()
        axes[0].set_title(f"{selected_driver} — Finish position at {selected_circuit}", fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Finish Position')
        axes[0].legend()

        # podium rate: this circuit vs overall
        categories = [f'At {selected_circuit}', 'All circuits']
        values     = [circuit_podium_rate, overall_podium_rate]
        colors     = ['#E85D24', '#378ADD']
        bars = axes[1].bar(categories, values, color=colors, edgecolor='white', width=0.4)
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"{selected_driver} — Podium rate comparison", fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Podium rate')
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.02, f'{val:.1%}',
                         ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info(f"{selected_driver} has no recorded races at {selected_circuit} in the dataset.")

    st.markdown("---")

    # ── chart 2: head-to-head comparison ──
    st.subheader(f"Head-to-Head: Top Drivers at Same Conditions")
    st.markdown("Comparing podium probability for top drivers at the same race configuration.")

    top_drivers = (
        df.groupby('driverRef')['podium']
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    if selected_driver not in top_drivers:
        top_drivers = [selected_driver] + top_drivers[:7]

    comparison_probs = []
    for drv in top_drivers:
        inp  = build_input(df, drv, selected_constructor, selected_circuit,
                           selected_year, grid_pos, quali_gap,
                           driver_standing, constructor_standing)
        prob_drv = model.predict_proba(inp)[0][1]
        comparison_probs.append({'Driver': drv, 'Podium Probability': prob_drv})

    comp_df = pd.DataFrame(comparison_probs).sort_values('Podium Probability', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ['#E85D24' if d == selected_driver else '#378ADD' for d in comp_df['Driver']]
    bars    = ax.barh(comp_df['Driver'], comp_df['Podium Probability'],
                      color=colors, edgecolor='white')
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(f"Podium probability — Grid P{grid_pos} at {selected_circuit}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Podium Probability')
    for bar, val in zip(bars, comp_df['Podium Probability']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=10)

    orange_patch = mpatches.Patch(color='#E85D24', label='Selected driver')
    blue_patch   = mpatches.Patch(color='#378ADD', label='Other drivers')
    ax.legend(handles=[orange_patch, blue_patch], loc='lower right')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── chart 3: SHAP feature contribution ──
    st.subheader("Why did the model predict this?")
    st.markdown("Top features that pushed the prediction up ↑ or down ↓")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    shap_df = pd.DataFrame({
        'Feature':    FEATURE_COLS,
        'SHAP Value': shap_values[0]
    }).reindex(pd.Series(np.abs(shap_values[0])).sort_values(ascending=False).index)
    shap_df = shap_df.head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['#1D9E75' if v > 0 else '#E85D24' for v in shap_df['SHAP Value']]
    bars    = ax.barh(shap_df['Feature'], shap_df['SHAP Value'],
                      color=colors, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title('Feature Contribution to Prediction (SHAP values)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('SHAP Value  (green = increases podium probability, red = decreases)')

    green_patch = mpatches.Patch(color='#1D9E75', label='Increases podium probability')
    red_patch   = mpatches.Patch(color='#E85D24', label='Decreases podium probability')
    ax.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # SHAP waterfall
    with st.expander("Show detailed SHAP waterfall plot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=FEATURE_COLS
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── race config summary ──
    st.subheader("Race Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.info(f"**Driver**\n\n{selected_driver}")
    col2.info(f"**Constructor**\n\n{selected_constructor}")
    col3.info(f"**Circuit**\n\n{selected_circuit}")
    col4.info(f"**Season**\n\n{selected_year}")

# ── landing page ──────────────────────────────────────────────────────────────
else:
    st.subheader("Configure a race in the sidebar and click Predict")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Data**
        - 11 datasets merged
        - 1950–2024 coverage
        - 26,000+ race entries
        - 700+ drivers
        """)
    with col2:
        st.markdown("""
        **Features**
        - Grid & qualifying gap
        - Championship standing
        - Circuit history
        - Driver & team form
        """)
    with col3:
        st.markdown("""
        **Model**
        - XGBoost classifier
        - ROC-AUC: 0.940
        - F1 Score: 0.707
        - SHAP explainability
        """)

    st.markdown("---")
    st.markdown("### Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        top_drivers = (
            df.groupby('driverRef')['podium']
            .sum().sort_values(ascending=False)
            .head(10).reset_index()
        )
        st.markdown("**Top 10 Drivers by Podiums**")
        st.dataframe(top_drivers.rename(columns={'driverRef': 'Driver', 'podium': 'Podiums'}),
                     hide_index=True, use_container_width=True)

    with col2:
        top_teams = (
            df[df['positionOrder'] == 1]
            .groupby('constructorRef').size()
            .sort_values(ascending=False)
            .head(10).reset_index(name='Wins')
        )
        st.markdown("**Top 10 Constructors by Wins**")
        st.dataframe(top_teams.rename(columns={'constructorRef': 'Constructor'}),
                     hide_index=True, use_container_width=True)