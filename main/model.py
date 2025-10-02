import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

def get_model(model_type="rf"):
    """Return the model based on the type selected."""
    logger.info(f"Creating {model_type} model")

    if model_type == "xgb":
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42
        )

    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    elif model_type == "dt":
        return DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42
        )

    elif model_type == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )

    elif model_type == "lr":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42
        )

    elif model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
