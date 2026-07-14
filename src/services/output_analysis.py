"""
Task
1. Save SHAP plot to output/shap_plot.png
2. Save metrics to output/results.json (and print as well)
3. plot pr_auc_curve
"""
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import shap
import json


OUTPUT_PATH = "output/"


def save_shap(model, X_test):
    feature_names = X_test.columns

    # shap
    shap_values = model.shap
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"shap_plot[{model.__name__}].png", dpi="300", bbox_inches="tight")
    plt.clf()


def save_json(metrics):
    # json
    with open(OUTPUT_PATH + f"metrics[{model.__name__}].json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(metrics)


def save_pr_auc(model, X_test, y_test):
    # pr_auc curve
    disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    disp.ax_.set_title(f"PR Curve (Estimator) [{model.__name__}]")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"pr_curve[{model.__name__}].png", dpi="300", bbox_inches="tight")
    plt.clf()


def output_analysis(model, metrics, X_test, y_test):
    save_shap(model, X_test)
    save_json(metrics)
    save_pr_auc(model, X_test, y_test)
