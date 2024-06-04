from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def get_response(query_embeddings, passages, passage_embeddings):
    """
    Generate a response based on the query embeddings and PDF content embeddings.
    
    Args:
    query_embeddings (torch.Tensor): Embeddings for the user's query.
    passages (list of str): List of passages from the PDF content.
    passage_embeddings (torch.Tensor): Embeddings for the PDF passages.
    
    Returns:
    str: The generated response.
    """
    # Find the most similar passage in the PDF content
    cosine_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
    best_idx = torch.argmax(cosine_scores)
    best_passage = passages[best_idx]

    # Prepare the prompt for the language model
    prompt = f"Answer the query based on the following passage:\n{best_passage}\nQuery: {query_embeddings}\nAnswer:"
    
    inputs = llm_tokenizer.encode(prompt, return_tensors='pt')
    outputs = llm_model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

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
