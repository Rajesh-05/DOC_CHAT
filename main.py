from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer


from pinecone import Pinecone, ServerlessSpec

from groq import Groq

client = Groq(
    api_key="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",
)


from flask import Flask, render_template, request, jsonify
import os
paths = []
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "final1"
pc = Pinecone(api_key="a7bdaea1-66dd-4525-ae19-6ab0191ff9cb")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200  
)
tot_chunks = []
def make_chunks(text) :
    return text_splitter.split_text(text)

app = Flask(__name__)

def get_context(ques) :
    index = pc.Index(index_name)

    ques_emb = model.encode(ques)
    DB_response = index.query(
        vector=ques_emb.tolist(),
        top_k=3,
        include_values=True
    )

    cont = ""
    for i in range(len(DB_response['matches'])) :
        cont += tot_chunks[int(DB_response['matches'][i]['id'][3:])-1]
    return cont

def extract_pdf(path) :
    pdf_path = path
    reader = PdfReader(pdf_path)

    extracted_text = ""

    for page in reader.pages:
        extracted_text += page.extract_text()

    print(extracted_text)
    return extracted_text

@app.route('/')
def index():
    return render_template('index-cpy.html')

@app.route('/upload_files', methods=['POST'])
def upload_file():
    global tot_chunks
    files = request.files.getlist('files')
    for file in files:
        file_path = file.filename
        file.save(file_path)
        paths.append(file_path)
        print(f"Uploaded file path: {file_path}")

    extracted = ""
    for path in paths: 
        extracted += extract_pdf(path)
    tot_chunks = make_chunks(extracted)
    tot_embeddings = model.encode(tot_chunks)

    tot_vectors = []
    for i, vec in enumerate(tot_embeddings):
        tot_vectors.append({"id": f"vec{i+1}", "values": vec.tolist()})
    try :
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 
    except :
        pass
    index = pc.Index(index_name)
    index.upsert( tot_vectors )
    return "Success !"


@app.route('/send', methods=['POST'])
def send_text():
    query = request.json['text']
    context = get_context(query)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Context : {context} , Analyse and understand the above context completely and answer the below query , Query : {query}",
            }
        ],
        model="llama3-8b-8192",
    )
    response_text = chat_completion.choices[0].message.content
    return response_text



if __name__ == '__main__':
    app.run()
