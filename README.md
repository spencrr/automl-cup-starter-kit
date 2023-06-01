# AutoML Cup Starter Kit

Welcome to the AutoML Cup Competition!

To get started:

1. Create an account on [Codabench](https://www.codabench.org/)
1. Register with the AutoML Cup Competition
1. Edit `submission/model.py`
1. Upload to Codabench under the `My Submissions` tab.

## Submission

The entrypoint `model.py` and `requirements.txt` can be added to a zip as follows:

```sh
cd submission/
zip ../submission.zip *
```

## Testing

To run the training and scoring programs locally (e.g., for the `splice` dataset):

```sh
export DATASET=splice
cd ingestion_program

python3 ingestion.py \
   --dataset_dir=../data/input_data/$DATASET \
   --output_dir=../output/ \
   --ingestion_program_dir=../ingestion_program_dir/ \
   --code_dir=../submission/ \
   --score_dir=../output/ \
   --temp_dir=../output/
```

```sh
export DATASET=splice
cd scoring_program

python score.py \
   --prediction_dir=../output \
   --dataset_dir=../data/input_data/$DATASET \
   --output_dir=../output/score/
```

There is also a Docker image provided that the true competition utilizes located in `docker/Dockerfile`. To have an equivalent environment, you may use this for your testing.

## Datasets

### Splice

Splice is provided via HuggingFace and should automatically download!

### ListOps

Put `basic_train.tsv` and `basic_val.tsv` in `data/input_data/listops/listops/` before trying to run the ingestion program.

## Reference

You can refer to the source code at

-   Ingestion Program: `ingestion/ingestion.py`
-   Scoring Program: `scoring/score.py`
