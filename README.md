# Temporal Bias Analysis in Large Language Models

## Overview

This repository contains the code and resources for analyzing temporal biases in large language models (LLMs). The study focuses on how different tokenization strategies affect the models' ability to handle diverse date formats and perform temporal reasoning tasks. The primary components of this repository include the `DateLogicQA` dataset, the `Semantic Integrity Metric`, and various scripts for evaluating and visualizing model performance.

## Table of Contents

1. Introduction
2. [DateLogicQA Dataset](#datelogicqa-dataset)
3. Methodology
    - [Semantic Integrity](#semantic-integrity)
    - [Human-Led Temporal Bias Assessment](#human-led-temporal-bias-assessment)
    - [Understanding Temporal Bias](#understanding-temporal-bias)
4. Usage

## Introduction

Temporal reasoning poses significant challenges for LLMs due to inherent biases and the complexity of date formats. This study introduces the `DateLogicQA` dataset and the `Semantic Integrity Metric` to evaluate how different tokenization strategies impact the models' ability to handle temporal data. By analyzing various models, we aim to understand the interplay between tokenization and temporal task performance.

## DateLogicQA Dataset

The `DateLogicQA` dataset is designed to explore how LLMs handle dates in various formats and contexts. It consists of 190 questions divided into four categories: commonsense, factual, conceptual, and numerical. Each question features one of seven date formats across three temporal contexts: past, present, and future. This systematic variation allows for an in-depth analysis of LLMs' performance with temporal information.

### Examples

- **Numerical**: What is the time 7 years and 9 months after 27101446?
- **Factual**: Which of the people died on 23041616? A) Shah Jahan B) Miguel de Cervantes C) Princess Diana D) William Shakespeare
- **Conceptual**: The first iPhone was released on 29062007. How many years has it been since its release?
- **Commonsense**: John was born on 15-03-1985. He graduated from college on 01-05-2007. Was John older than 18 when he graduated?

### Date Formats

- **DDMMYYYY**: 23041616
- **MMDDYYYY**: 04231616
- **DDMonYYYY**: 23April1616
- **DD-MM-YY**: 23-04-16
- **YYYY, Mon DD**: 1616, April 23
- **DD/YYYY (Julian calendar)**: 113/1616
- **YYYY/DD (Julian calendar)**: 1616/113

## Methodology

### Semantic Integrity

Semantic integrity assesses how well the tokenized date output preserves its intended meaning and structure. The score ranges from 0 to 1, with higher scores indicating better preservation. The formula for calculating semantic integrity is:

$$ SI = \max(0, \min(1, 1 - P - S - T - R)) $$

- **Unnecessary Splitting of Components (P)**: Penalty for incorrect tokenization.
- **Preservation of Separators (S)**: Penalty for losing separators.
- **Token Count (T)**: Penalty for excessive token splits.
- **Similarity with Baseline (R)**: Penalty based on cosine similarity with a baseline reference.

Run the following script to calculate semantic integrity scores for all models:
```sh
python src/semantic_int.py # Generate results for all models
```    

### Generating Results for Models

To generate results for all models using the `DateLogicQA` dataset, follow these steps:

Run the `gen_results.py` script to generate results for all models:

```sh
python src/gen_results.py # Generate results for all models
```

This script processes the `DateLogicQA` dataset and evaluates the performance of various models on temporal reasoning tasks. The results are saved for further analysis.

### Understanding Temporal Bias

We investigate potential biases in the internal embedding space and softmax computations of LLMs when processing texts with different temporal references. Temporal biases are quantified using cosine similarity and KL divergence.

Run the `embedding_chg.py` script to analyze embedding changes:

```sh
python src/embedding_chg.py # Analyze embedding changes
```

## Usage

To use the code and resources in this repository, follow these steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/gagan3012/EAIS-Temporal-Bias.git
    cd EAIS-Temporal-Bias
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Run the analysis scripts:

    ```sh
    python src/gen_results.py # Generate results for all models
    python src/semantic_int.py # Calculate semantic integrity scores 
    python src/embedding_chg.py # Analyze embedding changes
    ```

