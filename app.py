import os
from flask import Flask, request, render_template, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
import fitz  # PyMuPDF
import docx
import nltk
from nltk.tokenize import sent_tokenize
import speech_recognition as sr
import heapq
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import torch
import tensorflow as tf
import yt_dlp  # Use yt-dlp for downloading videos

nltk.data.path.append("./nltk_data")  # Set NLTK data path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

app = Flask(__name__)

# Load T5 model and tokenizer with non-legacy behavior
model_name = 't5-small'  # Switching to a smaller model to save memory
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_text(text):
    return text

def calculate_accuracy(original_text, summary_text):
    original_words = set(original_text.split())
    summary_words = set(summary_text.split())
    common_words = original_words.intersection(summary_words)
    return (len(common_words) / len(original_words)) * 100 if original_words else 0

def summarize_text_t5(text, summary_type='paragraph', summary_mode='abstractive'):
    text = preprocess_text(text)
    input_length = len(text.split())
    input_sentences = len(sent_tokenize(text))

    if summary_mode == 'abstractive':
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        # Extractive summarization using TF-IDF
        sentences = sent_tokenize(text)
        cv = CountVectorizer(max_df=0.85, stop_words='english')
        word_count_vector = cv.fit_transform(sentences)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)
        tfidf_vector = tfidf_transformer.transform(cv.transform(sentences))
        scores = tfidf_vector.sum(axis=1).A1
        ranked_sentences = [sentences[i] for i in heapq.nlargest(5, range(len(scores)), scores.take)]
        summary = " ".join(ranked_sentences)

    if summary_type == 'bullets':
        summary = "\n".join(f"- {sentence.strip()}" for sentence in sent_tokenize(summary))

    output_length = len(summary.split())
    output_sentences = len(sent_tokenize(summary))
    accuracy = calculate_accuracy(text, summary)

    return {
        'summary': summary,
        'input_length': {'words': input_length, 'sentences': input_sentences},
        'output_length': {'words': output_length, 'sentences': output_sentences},
        'accuracy': accuracy
    }

def extract_text_from_audio(audio_clip):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_clip) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Unable to recognize speech"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    content = request.json
    text = content.get('text', '')
    summary_type = content.get('summary_type', 'paragraph')
    summary_mode = content.get('summary_mode', 'abstractive')
    result = summarize_text_t5(text, summary_type, summary_mode)
    return jsonify(result)

@app.route('/summarize_document', methods=['POST'])
def summarize_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document part'})
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    text = extract_text_from_file(file)
    summary_type = request.form.get('summary_type', 'paragraph')
    summary_mode = request.form.get('summary_mode', 'abstractive')
    result = summarize_text_t5(text, summary_type, summary_mode)
    return jsonify(result)

@app.route('/summarize_video', methods=['POST'])
def summarize_video():
    content = request.json
    video_url = content.get('video_url', '')
    video_id = video_url.split('v=')[-1]

    try:
        # Fetch video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([t['text'] for t in transcript])

        # Summarize the transcript text
        text_summary_result = summarize_text_t5(text, summary_type='bullets')

        return jsonify({
            'text_summary': text_summary_result['summary'],
            'input_length': text_summary_result['input_length'],
            'output_length': text_summary_result['output_length'],
            'accuracy': text_summary_result['accuracy']
        })
    except (NoTranscriptFound, VideoUnavailable) as e:
        return jsonify({'error': str(e)})
    except Exception as e:
        return jsonify({'error': str(e)})

def extract_text_from_file(file):
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext == 'pdf':
        doc = fitz.open(stream=file.read(), filetype='pdf')
        text = ""
        for page in doc:
            text += page.get_text()
    elif file_ext == 'docx':
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = ""
    return text

if __name__ == '__main__':
    app.run(debug=True)
