"""
Flask Web Application for Gap Filling MLM
Simple web interface accessible via browser
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load model once at startup (cached)
print("🔄 Loading BERT model...")
mlm_model = pipeline('fill-mask', model='bert-base-uncased')
print("✅ Model loaded! Starting server...")

@app.route('/')
def home():
    """Serve the main HTML page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gap Filling with MLM</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 10px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .examples {
            margin: 20px 0;
        }
        
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        .example-btn:hover {
            background: #e0e0e0;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .prediction {
            padding: 15px;
            margin: 10px 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .prediction-word {
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .confidence-bar {
            height: 25px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }
        
        .completed-sentence {
            margin-top: 20px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 10px;
            font-size: 1.1em;
            color: #2e7d32;
            font-weight: bold;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Gap Filling with MLM</h1>
            <p>Intelligent text completion using BERT's Masked Language Modeling</p>
        </div>
        
        <div class="content">
            <div class="input-group">
                <label for="sentence">Enter a sentence with [MASK]:</label>
                <input 
                    type="text" 
                    id="sentence" 
                    placeholder="e.g., The weather is [MASK] today."
                    value="The cat sat on the [MASK]."
                >
            </div>
            
            <div class="examples">
                <strong>Try these examples:</strong><br>
                <span class="example-btn" onclick="loadExample('The cat sat on the [MASK].')">Example 1</span>
                <span class="example-btn" onclick="loadExample('I love to eat [MASK] for breakfast.')">Example 2</span>
                <span class="example-btn" onclick="loadExample('The [MASK] is shining brightly today.')">Example 3</span>
                <span class="example-btn" onclick="loadExample('Python is a [MASK] programming language.')">Example 4</span>
            </div>
            
            <button onclick="predict()" id="predictBtn">Predict</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing sentence...</p>
            </div>
            
            <div id="result"></div>
        </div>
    </div>
    
    <script>
        function loadExample(text) {
            document.getElementById('sentence').value = text;
        }
        
        function predict() {
            const sentence = document.getElementById('sentence').value;
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            const predictBtn = document.getElementById('predictBtn');
            
            // Validate input
            if (!sentence.includes('[MASK]')) {
                resultDiv.innerHTML = '<div class="error"⚠️ Please include [MASK] in your sentence!</div>';
                return;
            }
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            predictBtn.disabled = true;
            
            // Make API call
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sentence: sentence })
            })
            .then(response => response.json())
            .then(data => {
                loadingDiv.style.display = 'none';
                predictBtn.disabled = false;
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="error">❌ ${data.error}</div>`;
                    return;
                }
                
                // Display results
                let html = '<div class="result">';
                html += '<h3>🎯 Top Predictions:</h3>';
                
                data.predictions.forEach((pred, index) => {
                    const confidence = (pred.confidence * 100).toFixed(2);
                    html += `
                        <div class="prediction">
                            <div class="prediction-word">${index + 1}. ${pred.word.toUpperCase()}</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%">
                                    ${confidence}%
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                html += `<div class="completed-sentence">
                    ✅ Best Completion: ${data.completed_sentence}
                </div>`;
                html += '</div>';
                
                resultDiv.innerHTML = html;
            })
            .catch(error => {
                loadingDiv.style.display = 'none';
                predictBtn.disabled = false;
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            });
        }
        
        // Allow Enter key to trigger prediction
        document.getElementById('sentence').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                predict();
            }
        });
    </script>
</body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        sentence = data.get('sentence', '')
        
        # Validate
        if '[MASK]' not in sentence:
            return jsonify({'error': 'Please include [MASK] in your sentence'}), 400
        
        # Get predictions
        predictions = mlm_model(sentence)
        
        # Format results
        results = [{
            'word': pred['token_str'],
            'confidence': round(pred['score'], 4)
        } for pred in predictions]
        
        # Create completed sentence
        completed = sentence.replace('[MASK]', f"**{results[0]['word'].upper()}**")
        
        return jsonify({
            'predictions': results,
            'completed_sentence': completed
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'bert-base-uncased'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Flask Web Server")
    print("="*70)
    print("\n📍 Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print("\n⏹️  Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)