# SmartFab — Streamlit Demo & Edge-ML Repo

Modular IoT & Edge‑AI Energy Optimization prototype for MSMEs.
This repo contains scripts to generate synthetic sensor data, train an edge-ready model, and run a Streamlit dashboard demo.

Structure:
- data/ : synthetic data and pilot results
- firmware/edge_ai_code : training script and saved model artifacts
- software/streamlit_dashboard : Streamlit app, assets, requirements
- docs/ : placeholders for architecture/flow diagrams
- LICENSE, .gitignore, CONTRIBUTING.md

Run steps:
1. python data/generate_synthetic_data.py --machines 6 --hours 72 --freq_min 1
2. python firmware/edge_ai_code/train_model.py
3. cd software/streamlit_dashboard && streamlit run app.py
