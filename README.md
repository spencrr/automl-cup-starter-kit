# AutoML Cup Starter Kit

You can refer to the source code at

-   Ingestion Program: `ingestion/ingestion.py`
-   Scoring Program: `scoring/score.py`

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

## Submission

The entrypoint `model.py` and `requirements.txt` can be added to a zip as follows:

```sh
cd submission/
zip ../submission.zip *
```

## ListOps

Put `basic_train.tsv` and `basic_val.tsv` in `data/input_data/listops/listops/` before trying to run the ingestion program.
