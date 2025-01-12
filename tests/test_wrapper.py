import pandas as pd
from sklearn.datasets import fetch_openml

from mamut.wrapper import Mamut

pd.set_option("display.max_columns", None)

credit_g = fetch_openml(data_id=31)
X = credit_g.data
y = credit_g.target

print(type(X), type(y))

mamut = Mamut(n_iterations=2, exclude_models=["SVC"])
model = mamut.fit(X, y)

print(mamut.results_df_)
assert mamut.results_df_ is not None
assert mamut.best_score_ is not None
assert mamut.best_model_ is not None
assert mamut.fitted_models_ is not None
