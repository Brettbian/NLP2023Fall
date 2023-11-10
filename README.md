# NLP2023Fall

## Datasets Overview
This section documents the multiple-choice datasets used in the NLP2023Fall project, focusing on the generation of multiple-choice datasets with Chain of Thought (COT) reasoning facilitated by GPT models.

### 1. TruthfulQA
*Details about the TruthfulQA dataset will be added here.*

### 2. McTest
#### Overview
McTest is a medium-sized dataset sourced from [Hugging Face](https://huggingface.co/datasets/sagnikrayc/mctest), comprising 2,000 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 240 entries
- **Validation Set:** 120 entries
- **Test Set:** 240 entries

#### Processing and Access
- **Notebook:** [McTest.ipynb](gpt-multiple-choice-cot-dataset-generator/McTest/McTest.ipynb)
- **Generated Dataset:** [Hugging Face Repository](https://huggingface.co/datasets/BENBENBENb/McTest640COT)

### 3. RACE
#### Overview
The RACE dataset, a medium-sized collection from [Hugging Face](https://huggingface.co/datasets/race/viewer/middle), includes 28,300 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

#### Subset Selection
Selected by excluding stories over 800 characters and randomly sampling to achieve the subset size.

#### Processing and Access
- **Notebook:** [RACE.ipynb](gpt-multiple-choice-cot-dataset-generator/RACE/RACE.ipynb)
- **Generated Dataset:** [BENBENBENb/RACE1000COT](https://huggingface.co/datasets/BENBENBENb/RACE1000COT)

### 4. ARC
#### Overview
[ARC dataset](https://huggingface.co/datasets/ai2_arc/viewer/ARC-Easy)contains ARC-Easy and ARC-Challenge sets, which consists of two distinct sets: ARC-Easy and ARC-Challenge. The original ARC-Easy dataset contains approximately 5.2k entries, while the ARC-Challenge includes around 2.59k entries. Given computational constraints, we have processed a subset for practical handling, as detailed below::
- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

#### Subset Selection
Combined equal parts from ARC-Easy and ARC-Challenge, ensuring variety.

#### Processing and Access
- **Notebook:** [ARC.ipynb](gpt-multiple-choice-cot-dataset-generator/ARC/ARC.ipynb)
- **Generated Dataset:** [BENBENBENb/ARC1000COT](https://huggingface.co/datasets/BENBENBENb/ARC1000COT)

### 5. CommonsenseQA
#### Overview
CommonsenseQA is sourced from [Hugging Face](https://huggingface.co/datasets/commonsense_qa), totaling 12,102 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

#### Processing and Access
- **Notebook:** [CommonsenseQA.ipynb](gpt-multiple-choice-cot-dataset-generator/CommonsenseQA/CommonsenseQA.ipynb)
- **Generated Dataset:** [BENBENBENb/CommonsenseQA1000COT](https://huggingface.co/datasets/BENBENBENb/CommonsenseQA1000COT)

---

This revised format combines the introduction of each dataset with its corresponding link, making the README clearer and more streamlined.