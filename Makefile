N_JOBS ?= 12
MODEL  ?= pca_mlp
# Everything runs through the uv-managed environment; `uv run` syncs it first.
PY     ?= uv run python
PY_CNN ?= uv run --extra cnn python

.PHONY: all setup prepare explore benchmark benchmark-groups preprocessing tune submission cnn cnn-submit tokens token-cnn token-submit cnn-s100 token-cnn-s100 transformer lint clean-cache

all: prepare explore benchmark

setup:
	uv sync --all-groups

setup-cnn:
	uv sync --extra cnn --all-groups

prepare:
	$(PY) scripts/01_prepare_data.py --n-jobs $(N_JOBS)

explore:
	$(PY) scripts/02_explore_data.py --n-jobs $(N_JOBS)

benchmark:
	$(PY) scripts/03_benchmark_models.py --n-jobs $(N_JOBS)

benchmark-groups:
	$(PY) scripts/03_benchmark_models.py --n-jobs $(N_JOBS) --n-groups 4

preprocessing:
	$(PY) scripts/04_compare_preprocessing.py --n-jobs $(N_JOBS)

tune:
	$(PY) scripts/05_tune_model.py --model $(MODEL) --n-jobs $(N_JOBS)

submission:
	$(PY) scripts/06_predict_submission.py --model $(MODEL) \
		--params results/models/best_params_$(MODEL).json --n-jobs $(N_JOBS)

cnn:
	$(PY_CNN) scripts/09_benchmark_cnn.py --device cuda --n-jobs $(N_JOBS)

cnn-tune:
	$(PY_CNN) scripts/11_tune_cnn.py --device cuda --n-jobs $(N_JOBS)

cnn-submit:
	$(PY_CNN) scripts/10_predict_cnn.py --device cuda --n-jobs $(N_JOBS)

tokens:
	$(PY) scripts/12_build_tokens.py --n-jobs $(N_JOBS)

token-cnn:
	$(PY_CNN) scripts/13_benchmark_token_cnn.py --device cuda --n-jobs $(N_JOBS)

token-submit:
	$(PY_CNN) scripts/14_predict_token_cnn.py --device cuda --n-jobs $(N_JOBS)

# Last-chance neural: first 100 shots. Pixel CNN keeps the spectrum; token CNN
# keeps only Voigt R^2 and centroid shift (no amplitude, no dictionary static).
cnn-s100:
	$(PY_CNN) scripts/09_benchmark_cnn.py --device cuda --n-shots 100 --shot-bin 1 \
		--tag cnn2d_s100 --n-jobs $(N_JOBS)

token-cnn-s100:
	$(PY_CNN) scripts/13_benchmark_token_cnn.py --device cuda --n-shots 100 --shot-bin 1 \
		--sample-channels r2,delta_lambda --no-static --tag token_cnn_s100_r2dl \
		--n-jobs $(N_JOBS)

# Multi-scale 1-D conv tokeniser (kernels 3/7/15 -> 128-d per shot) followed by
# a transformer over the depth sequence. Defaults are the no_reg settings from
# scripts/16_tune_transformer.py (heavy dropout collapses this model at N=120).
transformer:
	$(PY_CNN) scripts/15_benchmark_transformer.py --device cuda --n-jobs $(N_JOBS)

transformer-tune:
	$(PY_CNN) scripts/16_tune_transformer.py --device cuda --n-jobs $(N_JOBS)

lint:
	uv run ruff check src scripts

clean-cache:
	rm -rf cache/features cache/images cache/tokens cache/lines
