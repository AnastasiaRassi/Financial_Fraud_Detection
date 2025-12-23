"""
Inspired by paper: https://arxiv.org/pdf/2208.11900

Deviation from paper:
- I no longer implement the MLP in PyTorch here.
- Instead, we construct classical ML models (e.g. KNN, RF, LR) as
  specified under 'model:' in 'config.yaml'.

All changes are documented to keep the deviation explicit and
reproducible.
"""
from typing import Dict, Optional, Any, List
import importlib
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _resolve_sklearn_class(class_name: str) -> Any:
    """
    Resolve a sklearn classifier class by name.

    This assumes the class lives in 'sklearn.<module>' where '<module>' is
    inferred from common classifier locations. If the mapping is ambiguous,
    this function will raise an explicit error so the config can be fixed.
    """
    # Minimal mapping to avoid guessing submodules too much.
    # If you extend the list of models, update this mapping explicitly.
    module_map = {
        "KNeighborsClassifier": "neighbors",
        "RandomForestClassifier": "ensemble",
        "LinearSVC": "svm",
        "DecisionTreeClassifier": "tree",
        "Perceptron": "linear_model",
        "PassiveAggressiveClassifier": "linear_model",
        "AdaBoostClassifier": "ensemble",
        "GradientBoostingClassifier": "ensemble",
        "LogisticRegression": "linear_model",
        "SGDClassifier": "linear_model",
        "RidgeClassifier": "linear_model",
        "GaussianNB": "naive_bayes",
        "QuadraticDiscriminantAnalysis": "discriminant_analysis",
        "DummyClassifier": "dummy",
    }

    if class_name not in module_map:
        raise ValueError(
            f"Unsupported sklearn_class '{class_name}'. "
            "Please add it explicitly to 'module_map' in 'src/model.py'."
        )

    module_name = module_map[class_name]
    module = importlib.import_module(f"sklearn.{module_name}")
    return getattr(module, class_name)


def build_models(config: Dict) -> List[Any]:
    """
    Build all sklearn models specified in the configuration.

    Args:
        config: Configuration dictionary. It must contain a 'model' key
                with a list of model specs, each having at least:
                - name: human-readable model name
                - sklearn_class: sklearn classifier class name

    Returns:
        List of instantiated sklearn model objects.

    Notes:
        - Any extra keys in each model spec (e.g. 'algorithm' for
          'AdaBoostClassifier') are passed directly as kwargs to the
          sklearn constructor.
        - No architecture / hyperparameter guessing is performed here;
          everything must come from the config for reproducibility.
    """
    model_specs = config.get("model", [])
    if not isinstance(model_specs, list) or len(model_specs) == 0:
        raise ValueError(
            "config['model'] must be a non-empty list of model definitions "
            "(see 'general_utils/config.yaml')."
        )

    models: List[Any] = []
    for spec in model_specs:
        if not isinstance(spec, dict):
            raise ValueError("Each model definition must be a dict.")

        name = spec.get("name")
        sklearn_class_name = spec.get("sklearn_class")
        if not name or not sklearn_class_name:
            raise ValueError(
                "Each model definition must contain 'name' and 'sklearn_class'."
            )

        cls = _resolve_sklearn_class(sklearn_class_name)

        # Pass any remaining entries as keyword arguments
        kwargs = {k: v for k, v in spec.items() if k not in {"name", "sklearn_class"}}
        model = cls(**kwargs)
        # Attach a readable name for logging / MLflow
        setattr(model, "_fd_name", name)
        models.append(model)

    return models

