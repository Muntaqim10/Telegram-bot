# RallyHunter Workspace Rules

## Alerting Mandate
- **Unmuted Alerts**: Dispatch Telegram alerts for Daily Breakouts/Breakdowns, Pullback/Retests, and Intraday ORB setups.
- **Fakeout Filter Enforced**: All breakout and ORB alerts must be verified by the XGBoost Classifier (Win Probability >= 75%) and Sentinel AI checks ("CONCORDANT") before dispatching to prevent fakes.
- **UOA Alerts Enforce Fakeout Filter**: Unusual Options Activity (UOA) alerts are enabled provided all setups pass the XGBoost Classifier (Win Probability >= 75%) and Sentinel AI checks ("CONCORDANT") before dispatching to prevent fakeouts.
