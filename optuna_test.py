import optuna
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)

# Surrogate function
fx = x ** 2 + y

plt.plot(x, fx, 'r')
plt.show()


def objective(trial):
    x = trial.suggest_int('x', -1, 10)
    y = trial.suggest_int('y', -1, 10)

    res = x ** 2 + y
    return res


# Begin optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

trial = study.best_trial

print('result: {}'.format(trial.value))
print(type(trial.value))
print("Best hyperparameters: {}".format(trial.params))
