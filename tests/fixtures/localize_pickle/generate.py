import json
import pathlib
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn

assert sklearn.__version__ == "1.5.2"
p = pathlib.Path(__file__).resolve().parent
p.mkdir(exist_ok=True)
x = np.array(
    [[i / 20, j / 4] for i in range(-20, 21) for j in range(-4, 5)], dtype=np.float64
)
y = ((x[:, 0] ** 2 + x[:, 1]) > 0.3).astype(int)
x = np.vstack((x, [[np.nan, 0.8], [0.5, np.nan]]))
y = np.r_[y, [1, 0]]
model = HistGradientBoostingClassifier(
    max_iter=3, max_leaf_nodes=3, min_samples_leaf=3, random_state=12
).fit(x, y)
xcheck = [[0.2, -0.9], [-0.8, 0.6], [0.5, 0.2], [None, 0.8], [0.5, None]]
(p / "sklearn152-binomial.pkl").write_bytes(pickle.dumps(model, protocol=2))
(p / "sklearn152-binomial.json").write_text(
    json.dumps(
        {
            "sklearn_version": sklearn.__version__,
            "inputs": xcheck,
            "probabilities": model.predict_proba(
                np.array(xcheck, dtype=float)
            ).tolist(),
        },
        indent=2,
    )
    + "\n"
)
print((p / "sklearn152-binomial.pkl").stat().st_size)
