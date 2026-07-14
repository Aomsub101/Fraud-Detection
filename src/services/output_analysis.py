from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import logging
import shap
import json
import os

logger = logging.getLogger(__name__)

OUTPUT_PATH = "output/"


def save_shap(model, X_test):
    logger.info("Saving SHAP summary plot for %s", model.__name__)
    feature_names = X_test.columns

    # shap
    shap_values = model.shap(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"shap_plot[{model.__name__}].png", dpi=300, bbox_inches="tight")


def save_json(model, metrics):
    logger.info("Saving metrics for %s: %s", model.__name__, metrics)
    with open(OUTPUT_PATH + f"metrics[{model.__name__}].json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(metrics)


def save_pr_auc(model, y_test, y_score):
    logger.info("Saving PR curve for %s", model.__name__)
    disp = PrecisionRecallDisplay.from_predictions(y_test, y_score)
    disp.ax_.set_title(f"PR Curve (Estimator) [{model.__name__}]")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"pr_curve[{model.__name__}].png", dpi=300, bbox_inches="tight")


def output_analysis(model, metrics, X_test, y_test, y_score):
    logger.info("Running output analysis for %s", model.__name__)
    os.makedirs("output", exist_ok=True)
    save_shap(model, X_test)
    save_json(model, metrics)
    save_pr_auc(model, y_test, y_score)
    logger.info("Output analysis complete: files written to %s", OUTPUT_PATH)
