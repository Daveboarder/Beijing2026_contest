N_JOBS ?= 12
MODEL  ?= pca_mlp
# Everything runs through the uv-managed environment; `uv run` syncs it first.
PY     ?= uv run python
PY_CNN ?= uv run --extra cnn python

.PHONY: all setup prepare explore benchmark benchmark-groups preprocessing tune submission cnn cnn-submit tokens token-cnn token-submit lint clean-cache

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

lint:
	uv run ruff check src scripts

.PHONY: depth-transformer test-depth-transformer autotransformer test-autotransformer ae-pca-mlp test-ae-pca-mlp
depth-transformer:
	$(PY_CNN) scripts/15_depth_transformer.py benchmark --device cuda --tag initial

test-depth-transformer:
	$(PY_CNN) -m unittest discover -s tests -v

autotransformer:
	$(PY_CNN) scripts/16_autotransformer.py benchmark --device cuda --tag initial

test-autotransformer:
	$(PY_CNN) -m unittest tests.test_autotransformer -v

ae-pca-mlp:
	$(PY_CNN) scripts/17_ae_pca_mlp.py benchmark --device cuda --tag g4_broadcast

test-ae-pca-mlp:
	$(PY_CNN) -m unittest tests.test_ae_pca_mlp -v

clean-cache:
	rm -rf cache/features cache/images cache/tokens cache/lines
