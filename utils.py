from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit


def validation_curve_model(X, Y, model, param_name, parameters, log=True):

    train_scores, test_scores = validation_curve(model, X, Y, param_name=param_name, param_range=parameters,
                                                 cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0), scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation curve")
    plt.fill_between(parameters, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(parameters, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

    if log==True:
        plt.semilogx(parameters, train_scores_mean, 'o-', color="r", label="Training score")
        plt.semilogx(parameters, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    else:
        plt.plot(parameters, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(parameters, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.ylabel('Score')
    plt.xlabel('Parameter C')
    plt.legend(loc="best")
    plt.show()


def calculate_mean_error(model, validation_x, validation_y):
    val_predictions = model.predict(validation_x)
    mean_absolute_error(validation_y, val_predictions)
    return val_predictions
