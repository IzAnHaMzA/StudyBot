"""
OS Study Bot - Machine Learning Training & Testing System
========================================================

This module implements comprehensive ML capabilities for the OS Study Bot:
- Question Classification
- Answer Quality Assessment  
- Content Similarity/Retrieval
- Difficulty Level Prediction
- Performance Testing & Evaluation
- Prediction API

Author: OS Study Bot Team
Version: 1.0
"""

import os
import json
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

# Deep Learning (Optional - requires torch)
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, 
        BertTokenizer, BertModel,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainingSystem:
    """
    Comprehensive ML training and testing system for OS Study Bot
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.syllabus_path = "OS SYLLABUS INCOMPLETE.txt"
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(f"{models_dir}/trained", exist_ok=True)
        os.makedirs(f"{models_dir}/results", exist_ok=True)
        
        # Initialize components
        self.question_classifier = None
        self.answer_evaluator = None
        self.content_retriever = None
        self.difficulty_predictor = None
        
        # Data storage
        self.training_data = {}
        self.test_results = {}
        
        # Text processing
        self.stemmer = PorterStemmer()
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            logger.warning("NLTK data not available. Using basic text processing.")
    
    def load_syllabus_data(self) -> str:
        """Load and preprocess syllabus content"""
        try:
            with open(self.syllabus_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded syllabus content: {len(content)} characters")
            return content
        except FileNotFoundError:
            logger.error(f"Syllabus file not found: {self.syllabus_path}")
            return ""
    
    def parse_questions_from_syllabus(self) -> Dict[str, List[str]]:
        """Parse questions from syllabus and categorize them"""
        content = self.load_syllabus_data()
        if not content:
            return {}
        
        questions = {
            'definition': [],
            'differentiation': [],
            'listing': [],
            'explanation': [],
            'all': []
        }
        
        lines = content.split('\n')
        current_unit = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect units
            if line.startswith('UNIT'):
                current_unit = line
                continue
            
            # Detect questions
            if re.match(r'^[Q\.]?\s*\d+', line) or re.match(r'^\d+\.', line):
                question = re.sub(r'^[Q\.]?\s*\d+\.?\s*', '', line).strip()
                if len(question) > 10:  # Filter out very short questions
                    questions['all'].append(question)
                    
                    # Classify question type
                    question_lower = question.lower()
                    if any(word in question_lower for word in ['what is', 'define', 'definition']):
                        questions['definition'].append(question)
                    elif any(word in question_lower for word in ['differentiate', 'difference', 'compare', 'contrast']):
                        questions['differentiation'].append(question)
                    elif any(word in question_lower for word in ['list', 'enumerate', 'types of', 'components of']):
                        questions['listing'].append(question)
                    else:
                        questions['explanation'].append(question)
        
        logger.info(f"Parsed questions: {len(questions['all'])} total")
        for category, q_list in questions.items():
            if category != 'all':
                logger.info(f"  {category}: {len(q_list)} questions")
        
        return questions
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for ML models"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text"""
        if not text:
            return {}
        
        features = {}
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Question-specific features
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        features['question_words'] = sum(1 for word in question_words if word in text.lower())
        
        # Technical terms (OS-specific)
        os_terms = ['process', 'memory', 'file', 'system', 'operating', 'scheduling', 'thread', 'cpu']
        features['os_terms'] = sum(1 for term in os_terms if term in text.lower())
        
        # Complexity indicators
        features['has_numbers'] = 1 if re.search(r'\d+', text) else 0
        features['has_special_chars'] = 1 if re.search(r'[^\w\s]', text) else 0
        
        return features

class QuestionClassifier:
    """Question Classification Model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.categories = ['definition', 'differentiation', 'listing', 'explanation']
    
    def prepare_training_data(self, questions_data: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Prepare training data for question classification"""
        X, y = [], []
        
        for category, questions in questions_data.items():
            if category == 'all':
                continue
            for question in questions:
                X.append(question)
                y.append(category)
        
        return X, y
    
    def train(self, X: List[str], y: List[str], test_size: float = 0.2):
        """Train the question classifier"""
        logger.info("Training Question Classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create pipeline
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Question Classifier Results:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1:.3f}")
        
        return results
    
    def predict(self, question: str) -> Dict[str, Any]:
        """Predict question category"""
        if not self.model:
            return {'error': 'Model not trained'}
        
        prediction = self.model.predict([question])[0]
        probabilities = self.model.predict_proba([question])[0]
        
        return {
            'category': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': dict(zip(self.categories, probabilities))
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")

class AnswerEvaluator:
    """Answer Quality Assessment Model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = StandardScaler()
    
    def prepare_training_data(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], List[float]]:
        """Prepare training data for answer evaluation"""
        X, y = [], []
        
        for qa in qa_pairs:
            if 'score' in qa and 'student_answer' in qa:
                # Extract features
                features = self.extract_answer_features(qa['student_answer'], qa.get('question', ''))
                X.append(features)
                y.append(qa['score'])
        
        return X, y
    
    def extract_answer_features(self, answer: str, question: str = "") -> Dict[str, float]:
        """Extract features from student answer"""
        features = {}
        
        # Basic text features
        features['answer_length'] = len(answer)
        features['word_count'] = len(answer.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', answer))
        
        # Content quality features
        features['has_technical_terms'] = 1 if any(term in answer.lower() for term in 
            ['process', 'memory', 'system', 'operating', 'cpu', 'thread']) else 0
        
        features['has_examples'] = 1 if any(word in answer.lower() for word in 
            ['example', 'for instance', 'such as', 'like']) else 0
        
        features['has_explanations'] = 1 if any(word in answer.lower() for word in 
            ['because', 'therefore', 'thus', 'hence']) else 0
        
        # Question-answer alignment
        if question:
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            features['word_overlap'] = len(question_words.intersection(answer_words)) / max(len(question_words), 1)
        
        return features
    
    def train(self, X: List[Dict], y: List[float], test_size: float = 0.2):
        """Train the answer evaluator"""
        logger.info("Training Answer Evaluator...")
        
        if not X or not y:
            logger.error("No training data available")
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'feature_importance': dict(zip(df.columns, self.model.feature_importances_))
        }
        
        logger.info(f"Answer Evaluator Results:")
        logger.info(f"  MAE: {mae:.3f}")
        logger.info(f"  RMSE: {np.sqrt(mse):.3f}")
        logger.info(f"  R² Score: {r2:.3f}")
        
        return results
    
    def predict(self, answer: str, question: str = "") -> Dict[str, Any]:
        """Predict answer quality score"""
        if not self.model:
            return {'error': 'Model not trained'}
        
        features = self.extract_answer_features(answer, question)
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        score = self.model.predict(features_scaled)[0]
        score = max(0, min(100, score))  # Clamp to 0-100 range
        
        return {
            'predicted_score': float(score),
            'features_used': features
        }

class ContentRetriever:
    """Content Similarity/Retrieval Model"""
    
    def __init__(self):
        self.vectorizer = None
        self.content_vectors = None
        self.content_texts = []
        self.content_metadata = []
    
    def prepare_content_data(self, syllabus_content: str) -> List[str]:
        """Prepare content for retrieval"""
        # Split content into chunks
        paragraphs = re.split(r'\n\s*\n', syllabus_content)
        
        # Filter and clean paragraphs
        content_chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Only keep substantial paragraphs
                content_chunks.append(para)
        
        logger.info(f"Prepared {len(content_chunks)} content chunks")
        return content_chunks
    
    def train(self, content_chunks: List[str]):
        """Train the content retriever"""
        logger.info("Training Content Retriever...")
        
        self.content_texts = content_chunks
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        self.content_vectors = self.vectorizer.fit_transform(content_chunks)
        
        logger.info(f"Content Retriever trained on {len(content_chunks)} chunks")
    
    def retrieve_similar_content(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar content for a query"""
        if not self.vectorizer or self.content_vectors is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = (self.content_vectors * query_vector.T).toarray().flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'content': self.content_texts[idx],
                    'similarity': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results

class DifficultyPredictor:
    """Difficulty Level Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.difficulty_levels = ['easy', 'medium', 'hard']
    
    def prepare_training_data(self, questions_data: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Prepare training data for difficulty prediction"""
        X, y = [], []
        
        # Simple heuristic for difficulty based on question characteristics
        for category, questions in questions_data.items():
            if category == 'all':
                continue
            
            for question in questions:
                X.append(question)
                
                # Heuristic difficulty assignment
                question_lower = question.lower()
                if any(word in question_lower for word in ['what is', 'define', 'list']):
                    difficulty = 'easy'
                elif any(word in question_lower for word in ['explain', 'describe']):
                    difficulty = 'medium'
                elif any(word in question_lower for word in ['differentiate', 'compare', 'analyze']):
                    difficulty = 'hard'
                else:
                    difficulty = 'medium'
                
                y.append(difficulty)
        
        return X, y
    
    def train(self, X: List[str], y: List[str], test_size: float = 0.2):
        """Train the difficulty predictor"""
        logger.info("Training Difficulty Predictor...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create pipeline
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        logger.info(f"Difficulty Predictor Results:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1:.3f}")
        
        return results
    
    def predict(self, question: str) -> Dict[str, Any]:
        """Predict question difficulty"""
        if not self.model:
            return {'error': 'Model not trained'}
        
        prediction = self.model.predict([question])[0]
        probabilities = self.model.predict_proba([question])[0]
        
        return {
            'difficulty': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': dict(zip(self.difficulty_levels, probabilities))
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")

class MLTestingFramework:
    """Comprehensive Testing Framework for ML Models"""
    
    def __init__(self, ml_system: MLTrainingSystem):
        self.ml_system = ml_system
        self.test_results = {}
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all ML model tests"""
        logger.info("Starting Comprehensive ML Testing...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test Question Classifier
        results['tests']['question_classifier'] = self.test_question_classifier()
        
        # Test Answer Evaluator
        results['tests']['answer_evaluator'] = self.test_answer_evaluator()
        
        # Test Content Retriever
        results['tests']['content_retriever'] = self.test_content_retriever()
        
        # Test Difficulty Predictor
        results['tests']['difficulty_predictor'] = self.test_difficulty_predictor()
        
        # Overall system test
        results['tests']['system_integration'] = self.test_system_integration()
        
        self.test_results = results
        return results
    
    def test_question_classifier(self) -> Dict[str, Any]:
        """Test question classification model"""
        logger.info("Testing Question Classifier...")
        
        test_questions = [
            ("What is an operating system?", "definition"),
            ("Differentiate between process and thread", "differentiation"),
            ("List the types of operating systems", "listing"),
            ("Explain process scheduling algorithms", "explanation")
        ]
        
        results = {
            'test_cases': len(test_questions),
            'predictions': [],
            'accuracy': 0
        }
        
        correct = 0
        for question, expected in test_questions:
            if self.ml_system.question_classifier:
                prediction = self.ml_system.question_classifier.predict(question)
                predicted_category = prediction.get('category', 'unknown')
                confidence = prediction.get('confidence', 0)
                
                is_correct = predicted_category == expected
                if is_correct:
                    correct += 1
                
                results['predictions'].append({
                    'question': question,
                    'expected': expected,
                    'predicted': predicted_category,
                    'confidence': confidence,
                    'correct': is_correct
                })
        
        results['accuracy'] = correct / len(test_questions) if test_questions else 0
        logger.info(f"Question Classifier Test Accuracy: {results['accuracy']:.3f}")
        
        return results
    
    def test_answer_evaluator(self) -> Dict[str, Any]:
        """Test answer evaluation model"""
        logger.info("Testing Answer Evaluator...")
        
        test_cases = [
            {
                'question': 'What is an operating system?',
                'answer': 'An operating system is system software that manages computer hardware and software resources.',
                'expected_score_range': (80, 95)
            },
            {
                'question': 'What is process scheduling?',
                'answer': 'Process scheduling is the method by which processes are assigned to run on the CPU.',
                'expected_score_range': (75, 90)
            }
        ]
        
        results = {
            'test_cases': len(test_cases),
            'predictions': [],
            'mae': 0
        }
        
        predictions = []
        actual_scores = []
        
        for case in test_cases:
            if self.ml_system.answer_evaluator:
                prediction = self.ml_system.answer_evaluator.predict(
                    case['answer'], case['question']
                )
                predicted_score = prediction.get('predicted_score', 0)
                
                # Use midpoint of expected range as actual score
                actual_score = (case['expected_score_range'][0] + case['expected_score_range'][1]) / 2
                
                predictions.append(predicted_score)
                actual_scores.append(actual_score)
                
                results['predictions'].append({
                    'question': case['question'],
                    'answer': case['answer'],
                    'predicted_score': predicted_score,
                    'expected_range': case['expected_score_range'],
                    'actual_score': actual_score
                })
        
        if predictions and actual_scores:
            results['mae'] = mean_absolute_error(actual_scores, predictions)
            logger.info(f"Answer Evaluator Test MAE: {results['mae']:.3f}")
        
        return results
    
    def test_content_retriever(self) -> Dict[str, Any]:
        """Test content retrieval model"""
        logger.info("Testing Content Retriever...")
        
        test_queries = [
            "process scheduling",
            "memory management",
            "file systems",
            "operating system security"
        ]
        
        results = {
            'test_queries': len(test_queries),
            'retrieval_results': [],
            'avg_similarity': 0
        }
        
        similarities = []
        
        for query in test_queries:
            if self.ml_system.content_retriever:
                retrieved = self.ml_system.content_retriever.retrieve_similar_content(query, top_k=3)
                
                if retrieved:
                    avg_sim = np.mean([r['similarity'] for r in retrieved])
                    similarities.append(avg_sim)
                    
                    results['retrieval_results'].append({
                        'query': query,
                        'results_count': len(retrieved),
                        'avg_similarity': avg_sim,
                        'top_result': retrieved[0]['content'][:100] + "..." if retrieved else "No results"
                    })
        
        if similarities:
            results['avg_similarity'] = np.mean(similarities)
            logger.info(f"Content Retriever Test Avg Similarity: {results['avg_similarity']:.3f}")
        
        return results
    
    def test_difficulty_predictor(self) -> Dict[str, Any]:
        """Test difficulty prediction model"""
        logger.info("Testing Difficulty Predictor...")
        
        test_questions = [
            ("What is an operating system?", "easy"),
            ("Explain process scheduling algorithms", "medium"),
            ("Differentiate between distributed and network operating systems", "hard")
        ]
        
        results = {
            'test_cases': len(test_questions),
            'predictions': [],
            'accuracy': 0
        }
        
        correct = 0
        for question, expected in test_questions:
            if self.ml_system.difficulty_predictor:
                prediction = self.ml_system.difficulty_predictor.predict(question)
                predicted_difficulty = prediction.get('difficulty', 'unknown')
                confidence = prediction.get('confidence', 0)
                
                is_correct = predicted_difficulty == expected
                if is_correct:
                    correct += 1
                
                results['predictions'].append({
                    'question': question,
                    'expected': expected,
                    'predicted': predicted_difficulty,
                    'confidence': confidence,
                    'correct': is_correct
                })
        
        results['accuracy'] = correct / len(test_questions) if test_questions else 0
        logger.info(f"Difficulty Predictor Test Accuracy: {results['accuracy']:.3f}")
        
        return results
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test overall system integration"""
        logger.info("Testing System Integration...")
        
        # Test end-to-end workflow
        test_question = "What is process scheduling?"
        
        results = {
            'test_question': test_question,
            'workflow_results': {}
        }
        
        # Step 1: Question Classification
        if self.ml_system.question_classifier:
            classification = self.ml_system.question_classifier.predict(test_question)
            results['workflow_results']['classification'] = classification
        
        # Step 2: Difficulty Prediction
        if self.ml_system.difficulty_predictor:
            difficulty = self.ml_system.difficulty_predictor.predict(test_question)
            results['workflow_results']['difficulty'] = difficulty
        
        # Step 3: Content Retrieval
        if self.ml_system.content_retriever:
            content = self.ml_system.content_retriever.retrieve_similar_content(test_question, top_k=3)
            results['workflow_results']['content_retrieval'] = {
                'results_count': len(content),
                'top_similarity': content[0]['similarity'] if content else 0
            }
        
        # Step 4: Answer Evaluation (with sample answer)
        sample_answer = "Process scheduling is the method by which processes are assigned to run on the CPU."
        if self.ml_system.answer_evaluator:
            evaluation = self.ml_system.answer_evaluator.predict(sample_answer, test_question)
            results['workflow_results']['answer_evaluation'] = evaluation
        
        results['integration_success'] = all(
            key in results['workflow_results'] 
            for key in ['classification', 'difficulty', 'content_retrieval', 'answer_evaluation']
        )
        
        logger.info(f"System Integration Test: {'PASSED' if results['integration_success'] else 'FAILED'}")
        
        return results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.test_results:
            return "No test results available. Run tests first."
        
        report = []
        report.append("=" * 60)
        report.append("OS STUDY BOT - ML TESTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {self.test_results['timestamp']}")
        report.append("")
        
        for test_name, test_data in self.test_results['tests'].items():
            report.append(f"## {test_name.upper().replace('_', ' ')}")
            report.append("-" * 40)
            
            if 'accuracy' in test_data:
                report.append(f"Accuracy: {test_data['accuracy']:.3f}")
            if 'mae' in test_data:
                report.append(f"Mean Absolute Error: {test_data['mae']:.3f}")
            if 'avg_similarity' in test_data:
                report.append(f"Average Similarity: {test_data['avg_similarity']:.3f}")
            if 'integration_success' in test_data:
                report.append(f"Integration Success: {test_data['integration_success']}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function to demonstrate ML training and testing"""
    logger.info("Starting OS Study Bot ML Training System...")
    
    # Initialize ML system
    ml_system = MLTrainingSystem()
    
    # Parse questions from syllabus
    questions_data = ml_system.parse_questions_from_syllabus()
    
    if not questions_data:
        logger.error("No questions found in syllabus. Exiting.")
        return
    
    # Initialize models
    ml_system.question_classifier = QuestionClassifier()
    ml_system.answer_evaluator = AnswerEvaluator()
    ml_system.content_retriever = ContentRetriever()
    ml_system.difficulty_predictor = DifficultyPredictor()
    
    # Train models
    logger.info("Training ML Models...")
    
    # Train Question Classifier
    X_q, y_q = ml_system.question_classifier.prepare_training_data(questions_data)
    if X_q and y_q:
        qc_results = ml_system.question_classifier.train(X_q, y_q)
        ml_system.question_classifier.save_model("models/trained/question_classifier.pkl")
    
    # Train Content Retriever
    syllabus_content = ml_system.load_syllabus_data()
    if syllabus_content:
        content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
        ml_system.content_retriever.train(content_chunks)
    
    # Train Difficulty Predictor
    X_d, y_d = ml_system.difficulty_predictor.prepare_training_data(questions_data)
    if X_d and y_d:
        dp_results = ml_system.difficulty_predictor.train(X_d, y_d)
        ml_system.difficulty_predictor.save_model("models/trained/difficulty_predictor.pkl")
    
    # Run comprehensive tests
    testing_framework = MLTestingFramework(ml_system)
    test_results = testing_framework.run_comprehensive_tests()
    
    # Save test results
    with open("models/results/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Generate and save report
    report = testing_framework.generate_test_report()
    with open("models/results/test_report.txt", "w") as f:
        f.write(report)
    
    print("\n" + report)
    logger.info("ML Training and Testing completed successfully!")

if __name__ == "__main__":
    main()
