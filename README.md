# Mortgage Risk Classification and Approval Strategy Simulation

This project uses machine learning to predict the risk level of mortgage applicants and simulate how different loan approval strategies affect overall portfolio risk. It combines domain-based lending logic with predictive modeling and an interactive Streamlit app to support better decision-making across loan operations.

---

## Project Highlights

- Merged borrower and loan datasets for full-feature modeling
- Created a rule-based risk label aligned with industry practices
- Trained and compared Logistic Regression, Random Forest, and XGBoost models
- Validated with 5-fold stratified cross-validation (mean F1 ≈ 0.987)
- Simulated approval strategies based on adjustable risk thresholds
- Deployed a live app for real-time predictions using Streamlit

---

## Folder Structure
mortgage-risk-model/
│
├── mortgage_risk_app.py # Streamlit app
├── xgb_pipeline.pkl # Trained model pipeline
├── requirements.txt # Package dependencies
├── README.md # This file
├── notebooks/
│ └── Moskova_StJulien_Risk_Modeling.ipynb # Full EDA + modeling notebook
└── outputs/
└── figures/ # Visualizations (feature importance, simulation)


---

## How to Run Locally

1. Clone this repo:

    ```bash
    git clone https://github.com/kingisbarlo/mortgage-risk-model.git
    cd mortgage-risk-model
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Streamlit app:

    ```bash
    streamlit run mortgage_risk_app.py
    ```

---

## Live App (Hosted on Streamlit Cloud)

[ Click here to launch the app](https://mortgage-risk-model-erou2zbg3hs5wvtl26mjy7.streamlit.app/)

---

## Future Work

- Use real-world loan outcomes (e.g., default data) to replace synthetic labels
- Add SHAP-based explanations for individual loan risk predictions
- Build out a dashboard view for executive strategy scenarios
- Integrate logging for analyzing user behavior

---

## License

MIT License — feel free to fork, adapt, or contribute!

