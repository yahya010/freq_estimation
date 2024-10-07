LANGUAGE := english
DATASET := natural_stories
MODEL := pythia-70m
DATA_DIR := ./corpora/rt/
CHECKPOINT_DIR := ./checkpoints/rt/
RESULTS_DIR := ./results/rt/

DATASET_BASE_NAME := $(if $(filter-out $(DATASET), provo_skip2zero),$(DATASET),provo)
DATASET_BASE_NAME := $(if $(filter-out $(DATASET), dundee_skip2zero),$(DATASET_BASE_NAME),dundee)

SKIP_IN_DATASET := $(if $(filter-out $(DATASET), provo_skip2zero),False,True)
SKIP_IN_DATASET := $(if $(filter-out $(DATASET), dundee_skip2zero),$(SKIP_IN_DATASET),True)


COLA_URL := https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
PROVO_URL1 := https://osf.io/a32be/download
PROVO_URL2 := https://osf.io/e4a2m/download
UCL_URL := https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip
BNC_URL := https://gu-clasp.github.io/914a288ca1e127a7f1547412d9a7e056/bnc.csv

NS_URL1 := https://raw.githubusercontent.com/languageMIT/naturalstories/master/probs/all_stories_gpt3.csv
NS_URL2 := https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv

NS_DIR := $(DATA_DIR)/natural_stories/
NS_FILE1 := $(NS_DIR)/all_stories_gpt3.csv
NS_FILE2 := $(NS_DIR)/processed_RTs.tsv

COLA_DIR_BASE := $(DATA_DIR)/cola/
COLA_FILE_RAW := $(COLA_DIR_BASE)/cola.zip
COLA_DIR := $(COLA_DIR_BASE)/cola_public/

PROVO_DIR := $(DATA_DIR)/provo/
PROVO_FILE1 := $(PROVO_DIR)/provo.csv
PROVO_FILE2 := $(PROVO_DIR)/provo_norms.csv

UCL_DIR := $(DATA_DIR)/ucl/
UCL_FILE_RAW := $(UCL_DIR)/ucl.zip
UCL_FILE := $(UCL_DIR)/stimuli_pos.txt

BROWN_DIR := $(DATA_DIR)/brown/
# BROWN_FILE_RAW := $(BROWN_DIR)/data.zip
BROWN_FILE := $(BROWN_DIR)/brown_spr.csv

BNC_DIR := $(DATA_DIR)/bnc/
BNC_FILE := $(BROWN_DIR)/bnc.csv

DUNDEE_DIR := $(DATA_DIR)/dundee/
DUNDEE_FILE_RAW := $(DATA_DIR)/dundee.zip
DUNDEE_FILE := $(DUNDEE_DIR)/eye-tracking/sa01ma1p.dat


DATASET_DIR := $(DATA_DIR)/$(DATASET_BASE_NAME)

TEXT_RT_DIR := $(CHECKPOINT_DIR)/text_rt_data/
SURPRISALS_DIR := $(CHECKPOINT_DIR)/surprisals_rt_data/
PREPROCESSED_RT_DIR := $(CHECKPOINT_DIR)/preprocessed_rt_data/
MERGED_DATA_DIR := $(CHECKPOINT_DIR)/merged_data/
PARAMS_DIR := $(CHECKPOINT_DIR)/params/
DELTA_LLH_DIR := $(CHECKPOINT_DIR)/delta_llh/

TEXT_RT_FILE := $(TEXT_RT_DIR)/$(DATASET).txt
SURPRISALS_FILE := $(SURPRISALS_DIR)/suprisal-$(DATASET)-$(MODEL).tsv
PREPROCESSED_RT_FILE := $(PREPROCESSED_RT_DIR)/$(DATASET).tsv
MERGED_DATA_FILE := $(MERGED_DATA_DIR)/$(DATASET)-$(MODEL).tsv
PROCESSED_FILE := $(RT_ENTROPY_DIR)/rt_vs_entropy-$(DATASET)-$(MODEL).tsv

ANALYSIS_PARAMS_FNAME_BASE := $(PARAMS_DIR)/params-$(DATASET)-$(MODEL)

LLH_FILE := $(DELTA_LLH_DIR)/llh-$(DATASET)-$(MODEL).tsv

WORD_LENGTHS_DIR := $(CHECKPOINT_DIR)/wiki40b_probs/
WORD_PROBS_FILE := $(WORD_LENGTHS_DIR)/finished_probs/surprisals_wiki_en_$(MODEL).tsv
LENGTH_PREDICTIONS_FILE := $(WORD_LENGTHS_DIR)/lengths/lengths_wiki_en_$(MODEL).tsv


all: get_data process_data get_llh

# get_length_predictions: $(LENGTH_PREDICTIONS_FILE)

get_llh: $(LLH_FILE)

process_data: $(TEXT_RT_FILE) $(SURPRISALS_FILE) $(PREPROCESSED_RT_FILE) $(MERGED_DATA_FILE)

get_data: $(COLA_DIR) $(PROVO_FILE2) $(UCL_FILE) $(NS_FILE2) $(DUNDEE_FILE) $(BROWN_FILE)
#  $(BNC_FILE)

# plot_results:
# 	mkdir -p results/plots
# 	python src/h03_paper/plot_effects.py
# 	python src/h03_paper/plot_entropy_vs_surprisal.py
# 	python src/h03_paper/plot_renyi_llh.py
# 	python src/other/renyi_analysis_script.py

print_table_1:
	python src/h03_paper/print_table_1_surprisal.py

plot_rt:
	mkdir -p $(RESULTS_DIR)
	python src/h03_paper/plot_rt.py --dataset natural_stories dundee provo --input-path $(DELTA_LLH_DIR) --output-path $(RESULTS_DIR)

# print_table_2:
# 	python src/h03_paper/print_table_2_fixed.py --model $(MODEL)

# plot_wordlengths:
# 	python src/h03_paper/plot_model_wordlength.py

# $(LENGTH_PREDICTIONS_FILE):
# 	python src/h01_data/get_length_predictions.py --model $(MODEL) --dataset $(DATASET) --input-path $(WORD_LENGTHS_DIR) --output-fname $(LENGTH_PREDICTIONS_FILE)

$(LLH_FILE):
	mkdir -p $(PARAMS_DIR)
	mkdir -p $(DELTA_LLH_DIR)
	Rscript src/h02_rt_model/rt_vs_surprisal.R $(MERGED_DATA_FILE) $(LLH_FILE) $(ANALYSIS_PARAMS_FNAME_BASE) --merge-workers --is-linear

# Preprocess rt data
$(MERGED_DATA_FILE):
	echo "Process rt data in " $(DATASET)
	mkdir -p $(MERGED_DATA_DIR)
	python src/h01_data/get_rt_with_surprisal_dataset.py --rt-fname $(PREPROCESSED_RT_FILE) --surprisal-fname $(SURPRISALS_FILE) --output-fname $(MERGED_DATA_FILE)

# Preprocess rt data
$(PREPROCESSED_RT_FILE):
	echo "Process rt data in " $(DATASET)
	mkdir -p $(PREPROCESSED_RT_DIR)
	python src/h01_data/preprocess_rt_dataset.py --dataset $(DATASET) --input-path $(DATASET_DIR) --output-fname $(PREPROCESSED_RT_FILE)

# Get surprisals
$(SURPRISALS_FILE):
	echo "Process rt data in " $(DATASET)
	mkdir -p $(SURPRISALS_DIR)
	wordsprobability --model $(MODEL) --input $(TEXT_RT_FILE) --output $(SURPRISALS_FILE) --return-buggy-surprisals

# Preprocess rt data
$(TEXT_RT_FILE):
	echo "Process rt data in " $(DATASET)
	mkdir -p $(TEXT_RT_DIR)
	python src/h01_data/extract_rt_text.py --dataset $(DATASET) --input-path $(DATASET_DIR) --output-fname $(TEXT_RT_FILE)

# Get natural stories data
$(NS_FILE2):
	mkdir -p $(NS_DIR)
	wget -O $(NS_FILE1) $(NS_URL1)
	wget -O $(NS_FILE2) $(NS_URL2)

# # Get BNC data
# $(BNC_FILE):
# 	mkdir -p $(BNC_DIR)
# 	wget -O $(BNC_FILE) $(BNC_URL)

# Get brown data
$(BROWN_FILE):
	mkdir -p $(BROWN_DIR)
	gdown -O $(BROWN_DIR)/brown_spr.csv https://drive.google.com/file/d/1cxBysSPldAj6nRqywVe4EstQoKtk67Qr

# Get UCL data
$(UCL_FILE):
	mkdir -p $(UCL_DIR)
	wget -O $(UCL_FILE_RAW) $(UCL_URL)
	unzip $(UCL_FILE_RAW) -d $(UCL_DIR)

# Get PROVO data
$(PROVO_FILE2):
	mkdir -p $(PROVO_DIR)
	wget -O $(PROVO_FILE1) $(PROVO_URL1)
	wget -O $(PROVO_FILE2) $(PROVO_URL2)

# Get dundee data
$(DUNDEE_FILE):
	unzip $(DUNDEE_FILE_RAW) -d $(DATA_DIR)

# Get COLA data
$(COLA_DIR):
	mkdir -p $(COLA_DIR_BASE)
	wget -O $(COLA_FILE_RAW) $(COLA_URL)
	unzip $(COLA_FILE_RAW) -d $(COLA_DIR_BASE)
