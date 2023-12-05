# NLP2023Fall

## Datasets Overview
This section documents the sentiment-analysis datasets used in the NLP2023Fall project, focusing on the generation of sentiment-analysis datasets with Chain of Thought (COT) reasoning facilitated by GPT models.

### 1. SST2
#### Overview
[SST2](https://huggingface.co/datasets/sst2) is a medium-sized dataset sourced from Hugging Face, comprising 70,042 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 673 entries
- **Test Set:** 931 entries

#### Processing and Access
- **Notebook:** [SST2.ipynb](gpt-multiple-choice-cot-dataset-generator/McTest/McTest.ipynb)
- **Generated Dataset:** [Hugging Face Repository](https://huggingface.co/datasets/BENBENBENb/McTest640COT)

### 2. Twitter Sentiment Analysis
#### Overview
[Twitter Sentiment Analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis) is a large-sized dataset sourced from Hugging Face, comprising 211,983 entries. Given computational constraints, we have processed a subset for practical handling, as detailed below:
- **Training Set:** 650 entries
- **Test Set:** 319 entries

#### Processing and Access
- **Notebook:** [Twitter.ipynb](gpt-multiple-choice-cot-dataset-generator/McTest/McTest.ipynb)
- **Generated Dataset:** [Hugging Face Repository](https://huggingface.co/datasets/BENBENBENb/McTest640COT)

### 3. Financial Phrasebank
#### Overview
[Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank), a medium-sized collection from Hugging Face, includes 14,780 entries. The sentiment is divided into four groups, namely sentences_50agree, sentences_66agree, sentences_75agree, sentences_allagree. Given computational constraints, we have processed a subset of "sentences_allgress" for practical handling, as detailed below:
- **Training Set:** 754 entries
- **Test Set:** 302 entries

#### Processing and Access
- **Notebook:** [Finance.ipynb](gpt-multiple-choice-cot-dataset-generator/RACE/RACE.ipynb)
- **Generated Dataset:** [BENBENBENb/RACE1000COT](https://huggingface.co/datasets/BENBENBENb/RACE1000COT)

### 4. Google Play
