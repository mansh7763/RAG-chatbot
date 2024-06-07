from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import get_embeddings, get_response, split_text_into_passages, split_text_into_chunks
import os
from sqlalchemy import create_engine
from supabase import create_client, Client
from utils import extract_text_from_pdf
from dotenv import load_dotenv
import logging
import json

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app) 
DATABASE_URL = os.getenv('POSTGRES_URL')

engine = create_engine(DATABASE_URL)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
print(SUPABASE_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # previous code:
        text = extract_text_from_pdf(file)
        logging.debug(f"Extracted text: {text[:500]}") 

        chunks = split_text_into_chunks(text,50)
        logging.debug(f"converting to passages: {chunks}")

        embedding = get_embeddings(chunks)
        logging.debug(f"getting the embeddings: {embedding}")

        # Convert the tensor to a list
        embedding_list = embedding.tolist()
        data = {'content':text, 'embeddings': embedding_list}
        logging.debug(f"Data to be sent to Supabase: {data}")

        # Insert data into Supabase
        response = supabase_client.table('pdfs').insert(data).execute()
        logging.debug(f"Supabase response: {response}")


        if 'error' in response:  
            logging.error(f"Error response from Supabase: {response}")
            return jsonify({'error': 'Failed to upload PDF'}), 500
        else:
            pdf_id = response.data[0]['id']
            return jsonify({'id': pdf_id}), 201
    except Exception as e:
        logging.error(f"Error uploading PDF: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data['query']
        pdf_id = data['pdf_id']
        
        logging.debug(f"Query text: {query_text}, PDF ID: {pdf_id}")

      
        response = supabase_client.table('pdfs').select('embeddings', 'content').eq('id', pdf_id).execute()
        # logging.debug(f"Supabase response yes: {response}")

        if response.data:
            pdf_content = response.data[0]['content']
          
            passages = split_text_into_chunks(pdf_content, 50)
            logging.debug(f"Passages: {passages[:5]}")

          
            # passage_embeddings = get_embeddings(passages)
            # logging.debug("Generated passage embeddings")
            passage_embeddings = response.data[0]['embeddings']
          
            query_embeddings = get_embeddings(query_text)
            logging.debug("Generated query embeddings")

            response_text = get_response(query_embeddings,query_text, passages, passage_embeddings)
            logging.debug(f"Response text: {response_text}")

            return jsonify({'response': response_text})
        else:
            logging.error(f"Error response from Supabase: {response}")
            return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        logging.error(f"Error querying: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
