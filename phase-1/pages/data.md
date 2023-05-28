# Data

This benchmark features 8 different hidden tabular datasets:

|Phase|Nickname|Task|
|---|---|---|
| Feedback | dataset1 | Classification |
| Feedback | dataset2 | Classification |
| Feedback | dataset3 | Classification |
| Feedback | dataset4 | Classification |
| Final | dataset1 | Classification |
| Final | dataset2 | Classification |
| Final | dataset3 | Classification |
| Final | dataset4 | Classification |

Each dataset is sent independently to the candidate model as:
- `X`: a `np.ndarray` of shape `(num_samples, num_features)`,
- `y`: a `np.ndarray` of shape `(num_samples)`, representing the labels in dense format.

The sources of the data are hidden for the purpose of showing what an AutoML Benchmark looks like.
You can learn more here: [Mini-AutoML Benchmark Bundle](https://github.com/codalab/competition-examples/tree/master/codabench/mini-automl).
