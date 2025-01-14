# www.youtube.com/@PythonCodeCampOrg

""" Subscribe to PYTHON CODE CAMP or I'll eat all your cookies... """

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as NNF
from sklearn.metrics.pairwise import cosine_similarity

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define some sentences we want to compare
sentences = ["I want to open a account.",
             "I want a credit card.",
             "I need to update my address.",
             "I want to apply for a loan."]

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Define function to compute similarity scores
def compute_similarity(input_sentence):
    # Tokenize the input sentence
    encoded_input = tokenizer(input_sentence, padding=True, truncation=True, return_tensors='pt')

    # Generate embeddings for the tokenized input sentence
    with torch.no_grad():
        input_model_output = model(**encoded_input)

    # compute the sentence embedding for the input sentence
    input_sentence_embedding = mean_pooling(input_model_output, encoded_input['attention_mask'])

    # Normalize the computed sentence embedding
    input_sentence_embedding = NNF.normalize(input_sentence_embedding, p=2, dim=1)

    # Tokenize list of predefined example sentences
    encoded_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Generate embeddings for list of sentences
    with torch.no_grad():
        sentences_model_output = model(**encoded_sentences)

    # compute the sentence embedding for the list of sentences
    sentences_embeddings = mean_pooling(sentences_model_output, encoded_sentences['attention_mask'])

    # Normalize the embeddings
    sentences_embeddings = NNF.normalize(sentences_embeddings, p=2, dim=1)

    # Compute cosine similarity between input sentence and list of sentences
    similarities = cosine_similarity(input_sentence_embedding, sentences_embeddings)

    # Zip sentences and their similarity scores
    sentences_with_scores = list(zip(sentences, similarities[0]))

    # Sort sentences based on similarity scores in descending order
    sorted_sentences = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)

    # Print sorted sentences and their similarity scores
    for i, (sentence, score) in enumerate(sorted_sentences, start=1):
        print(f"{sentence} : {score}")


while True:
    input_sentence = input("Enter your sentence (press 'q' to quit): ")
    if input_sentence.lower() == 'q':
        print("Exit...")
        break
    else:
        compute_similarity(input_sentence)

