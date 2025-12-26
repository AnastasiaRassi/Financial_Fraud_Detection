"""
Inspired by paper: https://arxiv.org/pdf/2208.11900

Deviation from paper:
I no longer implement the MLP in PyTorch here.
Instead, I construct classical ML models """

from typing import Dict, Optional, Any, List
import importlib
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _resolve_sklearn_class(class_name: str) -> Any:
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
        raise ValueError(f"Unsupported sklearn_class '{class_name}'. add it explicitly to 'module_map' in 'src/model.py'.")
 
    module_name = module_map[class_name]
    module = importlib.import_module(f"sklearn.{module_name}")
    return getattr(module, class_name)


def build_models(config: Dict) -> List[Any]:
    model_specs = config.get("model", [])
    if not isinstance(model_specs, list) or len(model_specs) == 0:
        raise ValueError("config['model'] must be a non-empty list.")

    models= []
    for spec in model_specs:
        if not isinstance(spec, dict):
            raise ValueError("Each model definition must be a dictionary.")

        name = spec.get("name")
        sklearn_class_name = spec.get("sklearn_class")
        if not name or not sklearn_class_name:
            raise ValueError( "Each model definition must contain 'name' and 'sklearn_class'.")

        cls = _resolve_sklearn_class(sklearn_class_name)
        kwargs = {k: v for k, v in spec.items() if k not in {"name", "sklearn_class"}}
        model = cls(**kwargs)

        setattr(model, "_fd_name", name)
        models.append(model)

    return models

