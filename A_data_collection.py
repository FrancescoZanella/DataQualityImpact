from sklearn.datasets import make_regression






# DEFAULT PARAMETERS FOR REGRESSION
#X, y = make_dataset_for_regression(n_samples=1000, n_features=3, n_informative=3, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, seed=2023)

def make_dataset_for_regression(n_samples, n_features, n_informative, n_targets, bias, effective_rank, tail_strength, noise, seed):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets,
                           bias=bias, effective_rank=effective_rank, tail_strength=tail_strength, noise=noise, random_state=seed)
    return X, y




