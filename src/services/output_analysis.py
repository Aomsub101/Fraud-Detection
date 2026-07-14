from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import shap
import json
import os


OUTPUT_PATH = "output/"


def save_shap(model, X_test):
    feature_names = X_test.columns

    # shap
    shap_values = model.shap(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"shap_plot[{model.__name__}].png", dpi=300, bbox_inches="tight")
    plt.clf()


def save_json(model, metrics):
    # json
    with open(OUTPUT_PATH + f"metrics[{model.__name__}].json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(metrics)


def save_pr_auc(model, y_test, y_score):
    # pr_auc curve
    disp = PrecisionRecallDisplay.from_predictions(y_test, y_score)
    disp.ax_.set_title(f"PR Curve (Estimator) [{model.__name__}]")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"pr_curve[{model.__name__}].png", dpi=300, bbox_inches="tight")
    plt.clf()


def output_analysis(model, metrics, X_test, y_test, y_score):
    os.makedirs("output", exist_ok=True)
    save_shap(model, X_test)
    save_json(model, metrics)
    save_pr_auc(model, y_test, y_score)
