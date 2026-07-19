from model_training.config import TrainingConfig
from model_training.train import train

GRID = [
    {"model_type": "linear_regression", "model_params": {}},
    {"model_type": "ridge", "model_params": {"alpha": 0.5}},
    {"model_type": "ridge", "model_params": {"alpha": 5.0}},
    {"model_type": "random_forest", "model_params": {"n_estimators": 200, "max_depth": 10}},
    {
        "model_type": "gradient_boosting",
        "model_params": {"learning_rate": 0.05, "n_estimators": 300},
    },
]


def main():
    for overrides in GRID:
        config = TrainingConfig(**overrides)
        result = train(config)
        print(config.model_type, overrides.get("model_params"), result["metrics"]["test"])


if __name__ == "__main__":
    main()
