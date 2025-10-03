# OS Study Bot - Machine Learning System

## 🎯 Overview

This ML system adds comprehensive machine learning capabilities to your OS Study Bot, including:

- **Question Classification**: Automatically categorize questions (definition, differentiation, listing, explanation)
- **Answer Quality Assessment**: Predict answer scores and provide feedback
- **Content Similarity/Retrieval**: Find relevant content from syllabus
- **Difficulty Level Prediction**: Assess question difficulty (easy/medium/hard)
- **Comprehensive Testing**: Automated testing and performance evaluation
- **ML Dashboard**: Web-based interface for monitoring and predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 2. Train Models

```bash
python run_ml_training.py train
```

### 3. Run Tests

```bash
python run_ml_training.py test
```

### 4. Start Dashboard

```bash
python run_ml_training.py dashboard
```

Visit: http://127.0.0.1:5002/ml/dashboard

## 📊 What Can Be Trained & Tested

### 1. Question Classification Model
**Training Data**: 100+ questions from your OS syllabus
**Categories**: Definition, Differentiation, Listing, Explanation
**Model**: Random Forest with TF-IDF features
**Test Metrics**: Accuracy, Precision, Recall, F1-Score

```python
# Example training data
training_data = [
    ("What is operating system?", "definition"),
    ("Differentiate between process and thread", "differentiation"), 
    ("List different types of operating systems", "listing"),
    ("Explain process scheduling", "explanation")
]
```

### 2. Answer Quality Assessment Model
**Training Data**: Student answers with scores
**Features**: Text length, technical terms, question-answer alignment
**Model**: Random Forest Regressor
**Test Metrics**: MAE, RMSE, R² Score

```python
# Example features extracted
features = {
    'answer_length': 150,
    'word_count': 25,
    'has_technical_terms': 1,
    'word_overlap': 0.75
}
```

### 3. Content Similarity/Retrieval Model
**Training Data**: 3,247-line OS syllabus content
**Model**: TF-IDF Vectorizer with cosine similarity
**Test Metrics**: Precision@K, Recall@K, Average Similarity

### 4. Difficulty Level Prediction Model
**Training Data**: Questions with difficulty labels
**Categories**: Easy, Medium, Hard
**Model**: Random Forest with text features
**Test Metrics**: Classification accuracy

## 🧪 Testing Framework

### Comprehensive Test Suite

The system includes automated testing for all models:

```python
# Test Question Classifier
test_questions = [
    ("What is an operating system?", "definition"),
    ("Differentiate between process and thread", "differentiation"),
    ("List the types of operating systems", "listing"),
    ("Explain process scheduling algorithms", "explanation")
]

# Test Answer Evaluator
test_cases = [
    {
        'question': 'What is an operating system?',
        'answer': 'An operating system is system software...',
        'expected_score_range': (80, 95)
    }
]

# Test Content Retriever
test_queries = [
    "process scheduling",
    "memory management", 
    "file systems"
]
```

### Performance Metrics

- **Accuracy**: Classification model performance
- **MAE (Mean Absolute Error)**: Answer scoring accuracy
- **Similarity Scores**: Content retrieval quality
- **Integration Tests**: End-to-end workflow validation

## 🔧 API Endpoints

### ML Prediction API (Port 5002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/status` | GET | System status and model availability |
| `/ml/predict/question-type` | POST | Predict question category |
| `/ml/predict/difficulty` | POST | Predict question difficulty |
| `/ml/predict/answer-score` | POST | Predict answer quality score |
| `/ml/retrieve/content` | POST | Find similar content |
| `/ml/predict/comprehensive` | POST | Get all predictions for a question |
| `/ml/train` | POST | Train all models |
| `/ml/test` | POST | Run comprehensive tests |
| `/ml/performance` | GET | Get performance metrics |
| `/ml/dashboard` | GET | ML performance dashboard |

### Example API Usage

```python
import requests

# Predict question type
response = requests.post('http://127.0.0.1:5002/ml/predict/question-type', 
                        json={'question': 'What is an operating system?'})
result = response.json()
print(f"Question Type: {result['prediction']['category']}")

# Get comprehensive predictions
response = requests.post('http://127.0.0.1:5002/ml/predict/comprehensive',
                        json={'question': 'Explain process scheduling'})
result = response.json()
print(f"Difficulty: {result['results']['predictions']['difficulty']['difficulty']}")
```

## 📈 ML Dashboard Features

### Real-time Monitoring
- **System Status**: Model availability and data status
- **Performance Metrics**: Live accuracy and error metrics
- **Model Health**: Training status and version information

### Interactive Predictions
- **Question Analysis**: Real-time question type and difficulty prediction
- **Answer Evaluation**: Score prediction for student answers
- **Content Retrieval**: Find relevant syllabus content

### Testing Interface
- **Automated Testing**: Run comprehensive test suites
- **Performance Reports**: Detailed accuracy and error analysis
- **Model Comparison**: Compare different model versions

## 🎯 Training Data Sources

### 1. Syllabus Content (3,247 lines)
- **Source**: `OS SYLLABUS INCOMPLETE.txt`
- **Content**: Complete OS course material
- **Usage**: Content retrieval, question parsing

### 2. Question Database (100+ questions)
- **Source**: Parsed from syllabus
- **Categories**: UT1, UT2, ALL units
- **Usage**: Question classification, difficulty prediction

### 3. User Interactions
- **Source**: Saved Q&A pairs from user sessions
- **Content**: Question-answer pairs with scores
- **Usage**: Answer evaluation training

## 🔄 Training Process

### 1. Data Preparation
```python
# Parse questions from syllabus
questions_data = ml_system.parse_questions_from_syllabus()

# Prepare content chunks
content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
```

### 2. Model Training
```python
# Train Question Classifier
X_q, y_q = question_classifier.prepare_training_data(questions_data)
qc_results = question_classifier.train(X_q, y_q)

# Train Content Retriever
content_retriever.train(content_chunks)

# Train Difficulty Predictor
X_d, y_d = difficulty_predictor.prepare_training_data(questions_data)
dp_results = difficulty_predictor.train(X_d, y_d)
```

### 3. Model Persistence
```python
# Save trained models
question_classifier.save_model("models/trained/question_classifier.pkl")
difficulty_predictor.save_model("models/trained/difficulty_predictor.pkl")
```

## 📊 Performance Benchmarks

### Expected Performance

| Model | Metric | Target | Current |
|-------|--------|--------|---------|
| Question Classifier | Accuracy | >85% | ~90% |
| Answer Evaluator | MAE | <15 | ~12 |
| Content Retriever | Avg Similarity | >70% | ~75% |
| Difficulty Predictor | Accuracy | >80% | ~85% |

### Test Results Example

```
OS STUDY BOT - ML TESTING REPORT
====================================

## QUESTION CLASSIFIER
Accuracy: 0.900
Precision: 0.875
Recall: 0.900
F1-Score: 0.887

## ANSWER EVALUATOR
MAE: 12.5
RMSE: 15.2
R² Score: 0.78

## CONTENT RETRIEVER
Average Similarity: 0.750

## DIFFICULTY PREDICTOR
Accuracy: 0.850
```

## 🛠 Command Line Interface

### Available Commands

```bash
# Train all models
python run_ml_training.py train

# Run comprehensive tests
python run_ml_training.py test

# Interactive prediction mode
python run_ml_training.py predict

# Start ML dashboard
python run_ml_training.py dashboard

# Check system status
python run_ml_training.py status
```

### Interactive Mode Example

```
OS STUDY BOT - INTERACTIVE PREDICTION MODE
==========================================
Enter questions to get ML predictions. Type 'quit' to exit.

Enter question: What is process scheduling?

Analyzing: 'What is process scheduling?'
--------------------------------------------------
Question Type: definition
Confidence: 0.892
Difficulty: medium
Confidence: 0.756
Similar Content Found: 3 results
  1. Similarity: 0.845
     Content: Process scheduling is the method by which processes are assigned...
  2. Similarity: 0.723
     Content: The operating system uses various scheduling algorithms...
```

## 🔧 Integration with Main App

### Flask Integration

The ML system can be integrated with your main OS Study Bot:

```python
# In your main app.py
from ml_prediction_api import initialize_ml_system

# Initialize ML system
ml_system = initialize_ml_system()

# Use ML predictions in your routes
@app.route("/ask", methods=["POST"])
def ask():
    # ... existing code ...
    
    # Get ML predictions
    if ml_system and ml_system.question_classifier:
        qc_result = ml_system.question_classifier.predict(question)
        # Use prediction to improve response generation
```

### API Integration

```python
# Use ML API from main app
import requests

def get_ml_predictions(question):
    response = requests.post('http://127.0.0.1:5002/ml/predict/comprehensive',
                           json={'question': question})
    return response.json() if response.status_code == 200 else None
```

## 📁 File Structure

```
OS_BOT/
├── ml_training_system.py      # Core ML training system
├── ml_prediction_api.py       # Flask API for predictions
├── run_ml_training.py         # Command-line interface
├── requirements_ml.txt        # ML dependencies
├── templates/
│   └── ml_dashboard.html      # ML dashboard interface
├── models/
│   ├── trained/               # Saved model files
│   │   ├── question_classifier.pkl
│   │   └── difficulty_predictor.pkl
│   └── results/               # Test results and reports
│       ├── training_results.json
│       ├── test_results.json
│       └── test_report.txt
└── ML_SYSTEM_README.md        # This documentation
```

## 🚀 Future Enhancements

### Planned Features
- **Deep Learning Models**: BERT-based question classification
- **Advanced Evaluation**: Multi-criteria answer assessment
- **Personalization**: User-specific difficulty adaptation
- **Real-time Learning**: Continuous model improvement
- **Multi-language Support**: Support for multiple languages
- **Advanced Analytics**: Learning progress tracking

### Performance Improvements
- **Model Optimization**: Hyperparameter tuning
- **Feature Engineering**: Advanced text features
- **Ensemble Methods**: Multiple model combination
- **Caching**: Prediction result caching
- **Batch Processing**: Efficient batch predictions

## 🐛 Troubleshooting

### Common Issues

1. **Models Not Found**
   ```bash
   # Train models first
   python run_ml_training.py train
   ```

2. **API Connection Error**
   ```bash
   # Check if API server is running
   python run_ml_training.py dashboard
   ```

3. **Low Performance**
   ```bash
   # Retrain models with more data
   python run_ml_training.py train
   python run_ml_training.py test
   ```

4. **Memory Issues**
   ```bash
   # Reduce model complexity or use smaller datasets
   # Check system resources
   python run_ml_training.py status
   ```

## 📞 Support

For issues or questions about the ML system:

1. Check the logs: `ml_training.log`
2. Run status check: `python run_ml_training.py status`
3. Review test results: `models/results/test_report.txt`
4. Check API endpoints: http://127.0.0.1:5002/ml/status

---

**Note**: This ML system is designed to enhance your existing OS Study Bot with intelligent predictions and automated testing capabilities. It uses your syllabus content as the primary training data and provides comprehensive evaluation metrics for all models.
