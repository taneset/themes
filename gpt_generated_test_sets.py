#%%#%% GPT-generated test set and gold clusters
Synonyms = [
    # Synonyms grouped together
       "Beautiful", "Attractive", "Stunning",
    "Begin", "Start", "Commence",
    "Courage", "Bravery", "Fortitude",
    "Delicious", "Tasty", "Flavorful",
    "Endless", "Infinite", "Boundless",
    "Fast", "Quick", "Rapid",
    "Generous", "Charitable", "Benevolent",
    "Happy", "Joyful", "Ecstatic",
    "Important", "Significant", "Crucial",
    "Knowledge", "Wisdom", "Insight",
    "Large", "Big", "Enormous",
    "Motivate", "Inspire", "Encourage",
    "Negative", "Adverse", "Unfavorable",
    "Opportunity", "Chance", "Possibility",
    "Peaceful", "Serene", "Tranquil",
    "Reliable", "Trustworthy", "Dependable",
    "Strong", "Powerful", "Mighty",
    "Trust", "Confidence", "Faith",
    "Unite", "Join", "Connect",
    "Victory", "Triumph", "Conquest",
    "Young", "Youthful", "Juvenile",
    "Zest", "Enthusiasm", "Zeal"
]
Non_synonyms=[    # Non-synonyms
    "apple", "house", "lamp", "ocean",
    "dog", "moon", "tree", "rock", "cloud",
    "computer",  "table","glove", "piano", 
    "beach", "bus",  "movie", 
    "diamond"
]

word_list_one=Synonyms+Non_synonyms

gold_clusters_one = []


start_value = 1
repeat_count = 3


for i in range(1, len(Synonyms)+1):
    value = start_value + (i - 1) // repeat_count
    gold_clusters_one.append(value)


for i in range(len(Non_synonyms)):  
    gold_clusters_one.append(gold_clusters_one[-1] + 1)


word_list_two=Synonyms

gold_clusters_two = []


start_value = 1
repeat_count = 3


for i in range(1, len(Synonyms)+1):
    value = start_value + (i - 1) // repeat_count
    gold_clusters_two.append(value)

