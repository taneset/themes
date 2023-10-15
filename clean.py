
#%%
import json
from nltk.corpus import words

def is_meaningless(word):
    """
    Check if a word is meaningless by comparing it to a list of common English words.

    Args:
        word (str): The word to check for meaning.

    Returns:
        bool: True if the word is meaningless, False if it's meaningful.
    """
    word = word.lower()
    return word not in words.words()

def clean_json(input_file, output_clean_file=None, output_meaningless_file=None):
    """
    Clean a JSON file containing theme titles by removing meaningless words and updating the data.

    Args:
        input_file (str): The path to the input JSON file.
        output_clean_file (str, optional): The path to the output JSON file with cleaned data. Defaults to 'clean_output.json'.
        output_meaningless_file (str, optional): The path to the output JSON file with meaningless titles. Defaults to 'meaningless_output.json'.

    Returns:
        str: The path to the clean JSON file.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    clean_titles = {}
    meaningless_titles = {}

    for theme, title in data['titles'].items():
        words_in_title = title.split()
        clean_words = [word for word in words_in_title if not is_meaningless(word)]
        
        if clean_words:
            clean_titles[theme] = ' '.join(clean_words)
        else:
            meaningless_titles[theme] = title

    for theme in list(data['theme_attributes'].keys()):
        if theme not in clean_titles:
            del data['theme_attributes'][theme]

    data['titles'] = clean_titles

    if output_clean_file is None:
        output_clean_file = 'clean_output.json'
    if output_meaningless_file is None:
        output_meaningless_file = 'meaningless_output.json'

    with open(output_clean_file, 'w') as f:
        json.dump(data, f, indent=4)

    with open(output_meaningless_file, 'w') as f:
        json.dump(meaningless_titles, f, indent=4)

    return output_clean_file

# Example usage
input_file = r"C:\Users\neset\OneDrive\Desktop\Thematic\themes.json"
cleaned_file = clean_json(input_file)

