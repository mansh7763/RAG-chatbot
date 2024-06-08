from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from sentence_transformers import util
import logging

# Load embedding model
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(embed_model_name)

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_embeddings(text):
    embeddings = embed_model.encode(text, convert_to_tensor=True)
    return embeddings


def get_response(query_embeddings, query_text, passages, passage_embeddings):
    # Find the most similar passages
    cosine_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
    top_scores, top_indices = torch.topk(cosine_scores, k=3)

    input_passages = ""
    for idx in top_indices[0]:  # Iterate through top indices
        input_passages += passages[idx.item()] + " "  # Concatenate with space  # Extract top passages
    logging.debug(f"Top passages: {input_passages}")

    # Prepare input for question-answering pipeline
    QA_input = {
        'question': query_text,
        'context': input_passages
    }
    logging.debug(f"QA input: {QA_input}")
    # Get predictions using question-answering pipeline
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    res = nlp(QA_input)
    logging.debug(f"Response: {res}")
    return res['answer']



def split_text_into_passages(text):
    """
    Split text into passages for better embedding comparison.
    
    Args:
    text (str): The input text to be split into passages.
    
    Returns:
    list of str: List of passages.
    """
    # Split by paragraphs or sentences, adjust as needed
    passages = text.split('\n')
    passages = [p for p in passages if p.strip()]  # Remove empty passages
    return passages

def split_text_into_chunks(text, max_chunk_size):
    """
    Recursively split text into chunks of a maximum specified size.
    
    Args:
    text (str): The input text to be split into chunks.
    max_chunk_size (int): The maximum size of each chunk.
    
    Returns:
    list of str: List of text chunks.
    """
    # If the text is already small enough, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    # Try to split at a natural break point within the limit
    split_points = ['\n', '. ', '!', '?', ';', ':']
    for split_point in split_points:
        index = text.rfind(split_point, 0, max_chunk_size)
        if index != -1:
            left_chunk = text[:index + len(split_point)]
            right_chunk = text[index + len(split_point):]
            return split_text_into_chunks(left_chunk.strip(), max_chunk_size) + split_text_into_chunks(right_chunk.strip(), max_chunk_size)
    
    # If no natural break point is found, split at the max_chunk_size
    left_chunk = text[:max_chunk_size]
    right_chunk = text[max_chunk_size:]
    return split_text_into_chunks(left_chunk.strip(), max_chunk_size) + split_text_into_chunks(right_chunk.strip(), max_chunk_size)