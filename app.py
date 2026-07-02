from flask import Flask, render_template, request, jsonify
import os
import requests as http_requests
from bs4 import BeautifulSoup
from truthnn import load_truthnn

app = Flask(__name__)

model = None
tokenizer = None


def scrape_article(url):
    """Scrape article text from a URL."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = http_requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    if not text:
        text = soup.get_text(separator=' ', strip=True)

    return text


def load_trained_model():
    """Load the trained TruthNN model on startup."""
    global model, tokenizer

    model_path = 'truthnn.pth'
    if os.path.exists(model_path):
        try:
            print("Loading TruthNN model...")
            model, tokenizer = load_truthnn(model_path)
            print("TruthNN model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    print("WARNING: No model file found. Please train first:")
    print("  python train_bert_improved.py combined_full_articles_training.csv")
    return False


def interpret_bias(x):
    """Interpret bias coordinate into human-readable description."""
    if x < -6:
        return "Far-Left"
    elif x < -3:
        return "Left"
    elif x < 3:
        return "Center"
    elif x < 6:
        return "Right"
    else:
        return "Far-Right"


if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or os.environ.get('FLASK_RUN_FROM_CLI') != 'true':
    load_trained_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if model is None:
        return render_template('index.html', error='Model not loaded. Please train the model first.')

    url = request.form.get('url', '').strip()
    text = request.form.get('text', '').strip()

    if url:
        try:
            text = scrape_article(url)
        except Exception as e:
            return render_template('index.html', error=f'Failed to scrape URL: {str(e)}', url=url)

    if not text:
        return render_template('index.html', error='Please provide a URL or article text.')

    if len(text) < 20:
        return render_template('index.html', error='Article text is too short. Please provide at least 20 characters.')

    try:
        x, confidence = model.predict_with_confidence(text, tokenizer)
        x_rounded = round(float(x), 2)
        confidence_rounded = round(float(confidence), 1)
        classification = interpret_bias(x_rounded)

        return render_template('index.html',
                             text=text,
                             url=url,
                             x=x_rounded,
                             confidence=confidence_rounded,
                             classification=classification,
                             model_type="TruthNN")

    except Exception as e:
        return render_template('index.html', error=f'Analysis failed: {str(e)}')


@app.route('/analyze', methods=['POST'])
def analyze():
    """JSON API endpoint."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    url = data.get('url', '').strip()
    text = data.get('text', '').strip()

    if url:
        try:
            text = scrape_article(url)
        except Exception as e:
            return jsonify({'error': f'Failed to scrape URL: {str(e)}'}), 400

    if not text:
        return jsonify({'error': 'Please provide a URL or article text'}), 400

    if len(text) < 20:
        return jsonify({'error': 'Article text is too short. Please provide at least 20 characters.'}), 400

    try:
        x, confidence = model.predict_with_confidence(text, tokenizer)
        influential_words = model.get_influential_words(text, tokenizer, top_n=8)
        x = round(float(x), 2)
        classification = interpret_bias(x)

        return jsonify({
            'coordinates': {'x': x},
            'confidence': round(float(confidence), 1),
            'classification': classification,
            'model_type': 'TruthNN',
            'influential_words': {
                'left': [word for word, _ in influential_words['left_words'][:5]],
                'right': [word for word, _ in influential_words['right_words'][:5]]
            }
        })

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("\nStarting TruthAI Web App...")
    print("Visit: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
