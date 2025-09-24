# CS 590 NLP - Homework 2
# Language Models with N-grams (manual implementation)
# Author: [Your Name]

import random
import re
from collections import defaultdict, Counter

# Function: tokenize
# Tokenizes text into lowercase words using regex
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Input: path_to_train_file (string) - text file to train on
# Output: trigram language model as a dict of dicts
#
# Steps:
# 1. Read file and tokenize
# 2. Count unigrams, bigrams, trigrams
# 3. Build probability distributions for trigrams

def train_LM(path_to_train_file):
    with open(path_to_train_file, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)
    tokens = ["<s>", "<s>"] + tokens + ["</s>"]  # padding for start/end

    # Count n-grams
    #finiding unigrams count using counter similar to finding frequency
    unigram_counts = Counter(tokens)
     #finiding bigrams count
    bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))
     #finiding trigrams count
    trigram_counts = Counter(zip(tokens[:-2], tokens[1:-1], tokens[2:]))

    # Build trigram probability model
    model = defaultdict(dict)
    for (w1, w2, w3), count in trigram_counts.items():
        context = (w1, w2)
        model[context][w3] = count / bigram_counts[(w1, w2)]

    print(f"LM has finished training on {path_to_train_file} ...")
    return model


# Function: generate_word
# Selects next word based on trigram probabilities

def generate_word(model, context):
    if context not in model:
        # stop if unseen context
        return "</s>"  

    words, probs = zip(*model[context].items())
    return random.choices(words, probs)[0]

# Function: LM_generate
# Input: model (dict), prompt (string)
# Output: Generated text (up to 15 tokens)

def LM_generate(model, prompt):
    prompt_tokens = tokenize(prompt)
    generated_tokens = prompt_tokens[:]

    # Ensure at least 2 words for trigram context
    if len(generated_tokens) < 2:
        generated_tokens = ["<s>"] * (2 - len(generated_tokens)) + generated_tokens

    for _ in range(15):
        context = tuple(generated_tokens[-2:])
        next_word = generate_word(model, context)

        if next_word == "</s>":
            break
        generated_tokens.append(next_word)

    # Remove padding
    result = " ".join([w for w in generated_tokens if w not in ["<s>", "</s>"]])
    result = result[0].upper() + result[1:]
    print(result)
    return result

#Sample Usage
if __name__ == "__main__":
    # Train on Leonardo da Vinci
    DaVinciLM = train_LM("leonardodavinci.txt")

    # Train on Edgar Allan Poe
    PoeLM = train_LM("edgarallanpoe.txt")


 # Example generations for Da Vinci
    print("\n--- Da Vinci LM Examples ---")
    LM_generate(DaVinciLM, "quoth the raven")
    LM_generate(DaVinciLM, "experience shows us that")
    LM_generate(DaVinciLM, "you are my")
    LM_generate(DaVinciLM, "the art of painting")
    LM_generate(DaVinciLM, "human knowledge arises from")
    LM_generate(DaVinciLM, "nature is full of")

    # Example generations for Poe
    print("\n--- Poe LM Examples ---")
    LM_generate(PoeLM, "quoth the raven")
    LM_generate(PoeLM, "experience shows us that")
    LM_generate(PoeLM, "you are my")
    LM_generate(PoeLM, "once upon a midnight")
    LM_generate(PoeLM, "deep into that darkness")
    LM_generate(PoeLM, "the beating of my")