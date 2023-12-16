## Datasets Overview
This section documents the datasets, focusing on the generation of sentiment-analysis datasets with Chain of Thought (COT) reasoning facilitated by GPT models.

### Task 1: Multiple-choice Question

### 1. Truthfulqa
#### Overview
[Truthfulqa](https://huggingface.co/datasets/truthful_qa) is a dataset that focuses on assessing the truthfulness of answers, where models are evaluated based on their ability to provide accurate and factually correct responses to questions. We have processed a subset for practical handling, as detailed below:
- **Training Set:** 656 entries
- **Test Set:** 161 entries

#### Processing and Access
- **Notebook:** [Truthfulqa.ipynb](cot_generator_multiple-choice-question/Truthfulqa.ipynb)
- **Generated Dataset:** [Hugging Face Repository](https://huggingface.co/datasets/BENBENBENb/McTest640COT)

### 2. McTest
#### Overview
[McTest](https://huggingface.co/datasets/sagnikrayc/mctest) is a medium-sized dataset sourced from Hugging Face, comprising 2,000 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 240 entries
- **Validation Set:** 120 entries
- **Test Set:** 240 entries

#### Processing and Access
- **Notebook:** [McTest.ipynb](cot_generator_multiple-choice-question/McTest.ipynb)
- **Generated Dataset:** [Hugging Face Repository](https://huggingface.co/datasets/BENBENBENb/McTest640COT)

### 3. RACE
#### Overview
The [RACE dataset](https://huggingface.co/datasets/race/viewer/middle), a medium-sized collection from Hugging Face, includes 28,300 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 600 entries
- **Validation Set:** 200 entries
- **Test Set:** 200 entries

#### Subset Selection
Selected by excluding stories over 800 characters and randomly sampling to achieve the subset size.

#### Processing and Access
- **Notebook:** [RACE.ipynb](cot_generator_multiple-choice-question/RACE.ipynb)
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
- **Notebook:** [ARC.ipynb](cot_generator_multiple-choice-question/ARC.ipynb)
- **Generated Dataset:** [BENBENBENb/ARC1000COT](https://huggingface.co/datasets/BENBENBENb/ARC1000COT)

### Task 2: Sentiment Analysis

### 1. SST2
#### Overview
[SST2](https://huggingface.co/datasets/sst2) is a medium-sized dataset sourced from Hugging Face, comprising 70,042 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 673 entries
- **Test Set:** 869 entries

#### Processing and Access
- **Notebook:** [SST2.ipynb](cot_generator_sentiment/SST2.ipynb)
- **Generated Dataset:** [Mariaaaaa/SST2_with_COT](https://huggingface.co/datasets/Mariaaaaa/SST2_with_COT)

### 2. Twitter Sentiment Analysis
#### Overview
[Twitter Sentiment Analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis) is a large-sized dataset sourced from Hugging Face, comprising 211,983 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 650 entries
- **Test Set:** 319 entries

#### Processing and Access
- **Notebook:** [Twitter.ipynb](cot_generator_sentiment/Twitter.ipynb)
- **Generated Dataset:** [Mariaaaaa/Twitter_with_COT](https://huggingface.co/datasets/Mariaaaaa/Twitter_with_COT)

### 3. Financial Phrasebank
#### Overview
[Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank), a medium-sized collection from Hugging Face, includes 14,780 entries. The sentiment is divided into four groups, namely sentences_50agree, sentences_66agree, sentences_75agree, sentences_allagree. Given computational constraints, we have processed a subset of "sentences_allgress" for practical handling, as detailed below:
- **Training Set:** 754 entries
- **Test Set:** 302 entries

#### Processing and Access
- **Notebook:** [Finance.ipynb](cot_generator_sentiment/Finance.ipynb)
- **Generated Dataset:** [Mariaaaaa/Finance_COT_GPT4](https://huggingface.co/datasets/Mariaaaaa/Finance_COT_GPT4)

### 4. Google Play
[GooglePlay](https://huggingface.co/datasets/Mariaaaaa/Googleplay_sentiment) is a self-built dataset containing players’ reviews on games, labeled as positive or negative, including 108837 entries.
- **Training Set:** 650 entries
- **Test Set:** 306 entries

#### Processing and Access
- **Notebook:** [Googleplay.ipynb](cot_generator_sentiment/Googleplay.ipynb)
- **Generated Dataset:** [Mariaaaaa/Google_with_COT](https://huggingface.co/datasets/Mariaaaaa/Google_with_COT)
  
## Fine-tune Model
All the scripts including training and inference can be found in [fine_tune_multiple_choice](fine_tune_multiple_choice) and [fine_tune_sentiment_analysis](fine_tune_sentiment_analysis)

