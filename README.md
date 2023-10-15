## Task Description

The task of merging themes using an API call to an LLM (Large Language Model). The goal of this task is to automatically merge themes in a given JSON file, `themes.json`, but it should be adaptable to any set of themes.



## JSON Schema
The JSON file (`themes.json`) has the following schema:
```json
{
  "titles": {
    "<theme id code>": "<title of theme>"
  },
  "theme_attributes": {
    "<theme id code>": {
      "<key value pairs containing attributes of themes>"
    }
  }
}
```
# 

## Proposed Solution
This project is a Python-based text clustering and evaluation tool that leverages the OpenAI API for text embedding and offers two clustering methods: K-Means and hierarchical clustering. It is designed to group similar texts together and provide evaluation metrics to assess the quality of the clustering results.

## Features
- Utilizes OpenAI API for text embedding.
- Supports two clustering methods: K-Means and hierarchical clustering.
- Calculates evaluation scores, such as Adjusted Rand Score and Fowlkes-Mallows Score, to assess clustering quality.
- Provides an easy-to-use command-line interface (CLI) for clustering text data.

## Installation
To set up this project locally, follow these steps:

1. Clone the repository from GitHub.
2. Install the required Python packages listed in `requirements.txt`  and provide openai.api_key.

## Run Hierarchical clustering
```bash
python cli.py --input "your_input.json" --method 'hierarchical'
```
You can also input the following arguments for hierarchical clustering:
    --linkage_method, default='ward'
    --distance_threshold", default=0.35
    --engine", default="text-embedding-ada-002"
## Run K-Means clustering
```bash
python cli.py --input "your_input.json" --method 'kmeans'
```
## Evaluation

We placed all functions in (`functions.py`) , and (`cli.py`) serves as a CLI-friendly script.

In (`eval.ipynb`), K-Means and hierarchical clustering are compared based on the Adjusted Rand Score and Fowlkes-Mallows Score etc. for GPT-generated synonym words (test sets  are organized in ` gpt_generated_test_sets.py`)

In the data folder, there are JSON format files  for testing. (`input_ex.json`) (`output_ex.json`)(`themes.json`)

Optionally, one can use (`clean.py`)  to clean JSON-formatted input if it contains repeated words, misspelled words, or words that are not present in the NLTK word corpus.
