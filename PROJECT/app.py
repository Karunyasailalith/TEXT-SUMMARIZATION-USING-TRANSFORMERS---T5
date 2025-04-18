import os
import nltk
import fitz  # PyMuPDF
from flask import Flask, request, render_template, redirect, url_for, session
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/summarizer_db"
mongo = PyMongo(app)

# uploads 0.
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def abstractive_summarization(text, num_sentences):
    """Perform abstractive summarization using the T5 model."""
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    input_text = f"summarize: {text[:1000]}" 
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs, max_length=num_sentences * 20, min_length=num_sentences * 10, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        
        if mongo.db.users.find_one({'username': username}):
            return "Username already exists. Try logging in."
        
        mongo.db.users.insert_one({'username': username, 'password': password})
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            return redirect(url_for('dashboard'))
        
        return "Invalid credentials."
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    summaries = list(mongo.db.summaries.find({'user_id': user_id}))
    return render_template('dashboard.html', summaries=summaries)

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    text = request.form.get('text', '').strip()
    num_sentences = request.form.get('num_sentences', '').strip()

    if not num_sentences.isdigit() or int(num_sentences) <= 0:
        return "Error: Please provide a valid number of sentences."

    num_sentences = int(num_sentences)

    if not text and 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            return "Error: Invalid file type. Please upload a PDF file."

    if not text:
        return "Error: No text provided for summarization."

    summary = abstractive_summarization(text, num_sentences)

    # Saveing to DB
    mongo.db.summaries.insert_one({
        'user_id': session['user_id'],
        'text': text,
        'summary': summary
    })

    return render_template("summary_result.html", summary=summary, original=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
