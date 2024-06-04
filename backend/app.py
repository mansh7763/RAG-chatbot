from flask import Flask, request, jsonify, render_template
from model import get_embeddings, get_response, split_text_into_passages
import os
from supabase import create_client, Client
from utils import extract_text_from_pdf
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        file = request.files['file']
        # Extract text from the PDF
        text = extract_text_from_pdf(file)
        # Save to Supabase
        response = supabase_client.table('pdfs').insert({'content': text}).execute()
        if response.status_code == 201:
            pdf_id = response.data[0]['id']
            return jsonify({'id': pdf_id}), 201
        else:
            return jsonify({'error': 'Failed to upload PDF'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data['query']
        pdf_id = data['pdf_id']
        
        # Get PDF content from Supabase
        response = supabase_client.table('pdfs').select('content').eq('id', pdf_id).execute()
        if response.status_code == 200 and response.data:
            pdf_content = response.data[0]['content']
            
            # Split PDF content into passages
            passages = split_text_into_passages(pdf_content)
            
            # Get embeddings for passages
            passage_embeddings = embed_model.encode(passages, convert_to_tensor=True)
            
            # Get query embeddings
            query_embeddings = get_embeddings(query_text)
            
            # Get response
            response_text = get_response(query_embeddings, passages, passage_embeddings)
            return jsonify({'response': response_text})
        else:
            return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
