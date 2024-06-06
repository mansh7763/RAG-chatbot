from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
# Load embedding model
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(embed_model_name)

# Load language model
llm_model_name = "gpt2"
llm_tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
llm_model = GPT2LMHeadModel.from_pretrained(llm_model_name)

def get_embeddings(text):
    """
    Generate embeddings for a given text using the embedding model.
    
    Args:
    text (str): The input text to be embedded.
    
    Returns:
    torch.Tensor: The embeddings for the input text.
    """
    embeddings = embed_model.encode(text, convert_to_tensor=True)
    return embeddings

# def get_response(query_embeddings, passages, passage_embeddings):
#     """
#     Generate a response based on the query embeddings and PDF content embeddings.
    
#     Args:
#     query_embeddings (torch.Tensor): Embeddings for the user's query.
#     passages (list of str): List of passages from the PDF content.
#     passage_embeddings (torch.Tensor): Embeddings for the PDF passages.
    
#     Returns:
#     str: The generated response.
#     """
#     # Find the most similar passage in the PDF content
#     cosine_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
#     best_idx = torch.argmax(cosine_scores)
#     best_passage = passages[best_idx]

#     # Prepare the prompt for the language model
#     prompt = f"Answer the query based on the following passage:\n{best_passage}\nQuery: {query_embeddings}\nAnswer:"
    
#     inputs = llm_tokenizer.encode(prompt, return_tensors='pt')
#     outputs = llm_model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
#     response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return response


# def get_response(query_embeddings, query, passages, passage_embeddings):
#     """
#     Generate responses based on the query embeddings and PDF content embeddings for the top-k chunks.
    
#     Args:
#     query_embeddings (torch.Tensor): Embeddings for the user's query.
#     query (str): The user's query.
#     passages (list of str): List of passages from the PDF content.
#     passage_embeddings (torch.Tensor): Embeddings for the PDF passages.
#     k (int): Number of top chunks to consider.
    
#     Returns:
#     list of str: The generated responses for the top-k chunks.
#     """
#     # Find the most similar passages in the PDF content
#     try:
#         cosine_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
#         top_scores, top_indices = torch.topk(cosine_scores, k=3)

#         input_passages = []
#         for idx in top_indices:
#             input_passages.append(str(passages[idx]))

#         logging.debug(f"Top passages: {input_passages}")

#         # Concatenate the passages into a single string
#         passages_text = "\n".join(input_passages)
#         logging.debug(f"Concatenated passages: {passages_text}")

#         # Prepare the prompt for the language model
#         prompt = f"Answer the query based on the passages passed in the list named input_passages:\n{passages_text}\nQuery: {query}\nAnswer:"

#         inputs = llm_tokenizer.encode(prompt, return_tensors='pt')
#         logging.debug(f"Inputs: {inputs}")
#         outputs = llm_model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
#         logging.debug(f"Outputs: {outputs}")
#         response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logging.debug(f"Response: {response}")
        
#         return response
#     except Exception as e:
#         logging.error(f"Error generating response: {e}")
#         return "An error occurred while generating the response."


import torch
import openai
from sentence_transformers import util

# Set your OpenAI API key
openai.api_key = 'sk-proj-vTOrRVi46LLlSxv2PlQTT3BlbkFJDtANSn2QCWJugE3wZGXS'

def get_response(query_embeddings,query_text, passages, passage_embeddings):
    """
    Generate responses based on the query embeddings and PDF content embeddings for the top-k chunks.
    
    Args:
    query_embeddings (torch.Tensor): Embeddings for the user's query.
    passages (list of str): List of passages from the PDF content.
    passage_embeddings (torch.Tensor): Embeddings for the PDF passages.
    k (int): Number of top chunks to consider.
    
    Returns:
    str: The generated response.
    """
    # Find the most similar passages in the PDF content
    cosine_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
    top_scores, top_indices = torch.topk(cosine_scores, k=3)
    
    best_passage = ""
    for idx in top_indices[0]:  # Get the first tensor in top_indices and iterate
        best_passage += passages[idx.item()]  # Convert to Python integer with .item()

    # Prepare the prompt for the language model
    prompt = f"Answer the query based on the following passage:\n{best_passage}\nQuery: {query_text}\nAnswer:"

    # Query the OpenAI API
    try:
        response = openai.Completion.create(
            engine="gpt-4-turbo",  # Use 'gpt-4' or 'gpt-4-turbo' as appropriate
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=None
        )
        answer = response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return None

    return answer
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