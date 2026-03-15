.PHONY: install data features train evaluate experiment tune all clean test

install:
	pip install -r requirements.txt

data:
	python scripts/make_dataset.py

features:
	python scripts/build_features.py

tune:
	python scripts/tune_hyperparams.py

train:
	python scripts/train.py --model all

train-classical:
	python scripts/train.py --model classical

train-deep:
	python scripts/train.py --model deep --epochs 3

evaluate:
	python scripts/evaluate.py

experiment:
	python scripts/experiment.py

all:
	python setup.py --step all

# quick test with small sample
test:
	python setup.py --max_samples 2000 --step all

clean:
	rm -rf data/processed/* data/outputs/* models/*

demo:
	python main.py demo
