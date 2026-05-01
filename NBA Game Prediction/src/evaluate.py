from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

def evaluate_models(models, X_test, y_test):
    results = {}

    os.makedirs("outputs", exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = (acc, prec, rec, f1)

        print(f"\n{name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1: {f1:.4f}")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        plt.imshow(cm)
        plt.title(f"{name} Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(f"outputs/{name.replace(' ', '_')}_cm.png")
        plt.close()

    return results


def plot_results(results):
    os.makedirs("outputs", exist_ok=True)

    names = list(results.keys())
    accuracies = [results[m][0] for m in names]

    plt.figure()
    plt.bar(names, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=20)

    plt.savefig("outputs/model_accuracy.png")
    plt.close()
