## 🔍 Sélection du meilleur modèle

Parmi les 4 combinaisons testées sur les hyperparamètres `max_depth` et `learning_rate`, le modèle ayant obtenu la meilleure performance est :

- `max_depth = 6`
- `learning_rate = 0.1`
- **Accuracy = 0.96812**

Cette configuration a été identifiée à l'aide de l'outil de visualisation de MLflow (graphique "Parallel Coordinates") :

![Comparaison des modèles](./docs/mlflow_parcoords.png)

Ce modèle sera utilisé pour la suite : prédiction avec wrapper, et déploiement via FastAPI.