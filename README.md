# NLP2023Fall

## Dataset Overview: Generate mutilple choice dataset with COT by GPT
### Dataset1: TruthfulQA

*Information about TruthfulQA dataset goes here.*

### Dataset2: McTest

We use a middle-sized dataset sourced from Hugging Face, originally consisting of 2,000 entries. Details and composition are as follows:

- **Training Set:** 240 entries
- **Validation Set:** 120 entries
- **Test Set:** 240 entries

Find the orginal dataset here: [McTest Dataset on Hugging Face](https://huggingface.co/datasets/sagnikrayc/mctest).

#### Processing Notebook

For dataset generation, preprocessing, and model training/testing processes, please refer to:

- [McTest.ipynb](gpt-multiple-choice-cot-dataset-generator/McTest/McTest.ipynb)

#### Generated Dataset Repository

The resulting dataset is hosted on Hugging Face at the following repository, which facilitates easy access and integration into machine learning workflows:

- **Hugging Face Repository:** `McTest640COT`

### Dataset3: RACE

The RACE dataset is a middle-sized collection sourced from Hugging Face with a total of 28,300 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:

- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

Find the orginal dataset here: [RACE Dataset on Hugging Face](https://huggingface.co/datasets/race/viewer/middle).

#### Subset Selection Method

The subset was formulated by excluding stories with more than 800 characters to ensure uniformity. We then performed random sampling to obtain 600 training entries, and 200 entries each for validation and test sets.

#### Processing Notebook

For a detailed explanation of dataset preparation and preprocessing steps, see the accompanying notebook:

- [RACE.ipynb](gpt-multiple-choice-cot-dataset-generator/RACE/RACE.ipynb)

#### Generated Dataset Repository

The resulting dataset is hosted on Hugging Face at the following repository, which facilitates easy access and integration into machine learning workflows:

- **Hugging Face Repository:** `BENBENBENb/RACE1000COT`

### Dataset4: ARC
In this project, we have worked with the ARC dataset, which consists of two distinct sets: ARC-Easy and ARC-Challenge. The original ARC-Easy dataset contains approximately 5.2k entries, while the ARC-Challenge includes around 2.59k entries. Due to computational constraints, it was necessary to create smaller, more manageable subsets of the data, as detailed below:

- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

Find the dataset here: [ARC Dataset on Hugging Face](https://huggingface.co/datasets/ai2_arc/viewer/ARC-Easy).

#### Subset Selection Method
- Approach: Equal parts from ARC-Easy and ARC-Challenge were combined after random shuffling to ensure variety and reproducibility.
- Subset Sizes: 300 entries for each category in the training set and 100 entries for each in the validation and test sets, totaling 600 for training and 200 for both validation and test sets.
- Modification on the Key: In the original dataset, the answerKey could be either numeric ('1', '2', '3', '4') or alphabetic ('A', 'B', 'C', 'D'). We converts numeric answer keys to their corresponding alphabetic labels to achieve a uniform format across the dataset.

#### Processing Notebook

For a detailed explanation of dataset preparation and preprocessing steps, see the accompanying notebook:

- [ARC.ipynb](gpt-multiple-choice-cot-dataset-generator/ARC/ARC.ipynb)

#### Generated Dataset Repository

The resulting dataset is hosted on Hugging Face at the following repository, which facilitates easy access and integration into machine learning workflows:

- **Hugging Face Repository:** `BENBENBENb/ARC1000COT`

### Dataset5: CommonsenseQA

The CommonsenseQA dataset is a middle-sized collection sourced from Hugging Face with a total of 12,102 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:

- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

Find the orginal dataset here: [CommonsenseQA Dataset on Hugging Face](https://huggingface.co/datasets/commonsense_qa).

#### Subset Selection Method

We performed random sampling to obtain 600 training entries, and 200 entries each for validation and test sets.

#### Processing Notebook

For a detailed explanation of dataset preparation and preprocessing steps, see the accompanying notebook:

- [CommonsenseQA.ipynb](gpt-multiple-choice-cot-dataset-generator/CommonsenseQA/CommonsenseQA.ipynb)

#### Generated Dataset Repository

The resulting dataset is hosted on Hugging Face at the following repository, which facilitates easy access and integration into machine learning workflows:

- **Hugging Face Repository:** `BENBENBENb/CommonsenseQA1000COT`