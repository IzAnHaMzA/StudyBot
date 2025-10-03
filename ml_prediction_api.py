"""
OS Study Bot - ML Prediction API
===============================

Flask API endpoints for ML model predictions and training management.
Integrates with the main OS Study Bot application.

Author: OS Study Bot Team
Version: 1.0
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import logging
from typing import Dict, Any, List

# Import our ML system
from ml_training_system import (
    MLTrainingSystem, QuestionClassifier, AnswerEvaluator, 
    ContentRetriever, DifficultyPredictor, MLTestingFramework
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Enable CORS for all routes
CORS(app, origins=['*'], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type'])

# Global ML system instance
ml_system = None

def initialize_ml_system():
    """Initialize the ML system"""
    global ml_system
    try:
        ml_system = MLTrainingSystem()
        
        # Load trained models if they exist
        if os.path.exists("models/trained/question_classifier.pkl"):
            ml_system.question_classifier = QuestionClassifier()
            ml_system.question_classifier.load_model("models/trained/question_classifier.pkl")
            logger.info("Question Classifier loaded")
        
        if os.path.exists("models/trained/difficulty_predictor.pkl"):
            ml_system.difficulty_predictor = DifficultyPredictor()
            ml_system.difficulty_predictor.load_model("models/trained/difficulty_predictor.pkl")
            logger.info("Difficulty Predictor loaded")
        
        # Initialize other components
        if not ml_system.answer_evaluator:
            ml_system.answer_evaluator = AnswerEvaluator()
        
        if not ml_system.content_retriever:
            ml_system.content_retriever = ContentRetriever()
            # Load content if available
            syllabus_content = ml_system.load_syllabus_data()
            if syllabus_content:
                content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
                ml_system.content_retriever.train(content_chunks)
                logger.info("Content Retriever initialized")
        
        logger.info("ML System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize ML system: {e}")
        return False

# Initialize ML system on startup
initialize_ml_system()

@app.route("/ml/status")
def ml_status():
    """Get ML system status"""
    if not ml_system:
        return jsonify({
            "ok": False,
            "error": "ML system not initialized"
        })
    
    status = {
        "ok": True,
        "models": {
            "question_classifier": ml_system.question_classifier is not None,
            "answer_evaluator": ml_system.answer_evaluator is not None,
            "content_retriever": ml_system.content_retriever is not None,
            "difficulty_predictor": ml_system.difficulty_predictor is not None
        },
        "data": {
            "syllabus_loaded": bool(ml_system.load_syllabus_data()),
            "questions_parsed": len(ml_system.parse_questions_from_syllabus().get('all', []))
        }
    }
    
    return jsonify(status)

@app.route("/ml/predict/question-type", methods=["POST"])
def predict_question_type():
    """Predict question type/category"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"ok": False, "error": "Question is required"})
        
        if not ml_system or not ml_system.question_classifier:
            return jsonify({
                "ok": False,
                "error": "Question classifier not available. Train the model first."
            })
        
        prediction = ml_system.question_classifier.predict(question)
        
        return jsonify({
            "ok": True,
            "question": question,
            "prediction": prediction
        })
        
    except Exception as e:
        logger.error(f"Error in question type prediction: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/predict/difficulty", methods=["POST"])
def predict_difficulty():
    """Predict question difficulty level"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"ok": False, "error": "Question is required"})
        
        if not ml_system or not ml_system.difficulty_predictor:
            return jsonify({
                "ok": False,
                "error": "Difficulty predictor not available. Train the model first."
            })
        
        prediction = ml_system.difficulty_predictor.predict(question)
        
        return jsonify({
            "ok": True,
            "question": question,
            "prediction": prediction
        })
        
    except Exception as e:
        logger.error(f"Error in difficulty prediction: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/predict/answer-score", methods=["POST"])
def predict_answer_score():
    """Predict answer quality score"""
    try:
        data = request.get_json()
        answer = data.get("answer", "").strip()
        question = data.get("question", "").strip()
        
        if not answer:
            return jsonify({"ok": False, "error": "Answer is required"})
        
        if not ml_system or not ml_system.answer_evaluator:
            return jsonify({
                "ok": False,
                "error": "Answer evaluator not available. Train the model first."
            })
        
        prediction = ml_system.answer_evaluator.predict(answer, question)
        
        return jsonify({
            "ok": True,
            "question": question,
            "answer": answer,
            "prediction": prediction
        })
        
    except Exception as e:
        logger.error(f"Error in answer score prediction: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/retrieve/content", methods=["POST"])
def retrieve_content():
    """Retrieve similar content for a query"""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        top_k = data.get("top_k", 5)
        
        if not query:
            return jsonify({"ok": False, "error": "Query is required"})
        
        if not ml_system or not ml_system.content_retriever:
            return jsonify({
                "ok": False,
                "error": "Content retriever not available. Initialize the system first."
            })
        
        results = ml_system.content_retriever.retrieve_similar_content(query, top_k)
        
        return jsonify({
            "ok": True,
            "query": query,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in content retrieval: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/predict/comprehensive", methods=["POST"])
def comprehensive_prediction():
    """Get comprehensive predictions for a question"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"ok": False, "error": "Question is required"})
        
        if not ml_system:
            return jsonify({"ok": False, "error": "ML system not initialized"})
        
        results = {
            "question": question,
            "predictions": {}
        }
        
        # Question type prediction
        if ml_system.question_classifier:
            results["predictions"]["question_type"] = ml_system.question_classifier.predict(question)
        
        # Difficulty prediction
        if ml_system.difficulty_predictor:
            results["predictions"]["difficulty"] = ml_system.difficulty_predictor.predict(question)
        
        # Content retrieval
        if ml_system.content_retriever:
            content_results = ml_system.content_retriever.retrieve_similar_content(question, top_k=3)
            results["predictions"]["similar_content"] = {
                "results_count": len(content_results),
                "top_similarity": content_results[0]['similarity'] if content_results else 0,
                "results": content_results
            }
        
        return jsonify({
            "ok": True,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in comprehensive prediction: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/train", methods=["POST"])
def train_models():
    """Train ML models"""
    try:
        if not ml_system:
            return jsonify({"ok": False, "error": "ML system not initialized"})
        
        # Parse questions from syllabus
        questions_data = ml_system.parse_questions_from_syllabus()
        
        if not questions_data:
            return jsonify({"ok": False, "error": "No questions found in syllabus"})
        
        training_results = {}
        
        # Train Question Classifier
        if not ml_system.question_classifier:
            ml_system.question_classifier = QuestionClassifier()
        
        X_q, y_q = ml_system.question_classifier.prepare_training_data(questions_data)
        if X_q and y_q:
            qc_results = ml_system.question_classifier.train(X_q, y_q)
            ml_system.question_classifier.save_model("models/trained/question_classifier.pkl")
            training_results["question_classifier"] = qc_results
        
        # Train Difficulty Predictor
        if not ml_system.difficulty_predictor:
            ml_system.difficulty_predictor = DifficultyPredictor()
        
        X_d, y_d = ml_system.difficulty_predictor.prepare_training_data(questions_data)
        if X_d and y_d:
            dp_results = ml_system.difficulty_predictor.train(X_d, y_d)
            ml_system.difficulty_predictor.save_model("models/trained/difficulty_predictor.pkl")
            training_results["difficulty_predictor"] = dp_results
        
        # Train Content Retriever
        if not ml_system.content_retriever:
            ml_system.content_retriever = ContentRetriever()
        
        syllabus_content = ml_system.load_syllabus_data()
        if syllabus_content:
            content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
            ml_system.content_retriever.train(content_chunks)
            training_results["content_retriever"] = {"status": "trained", "chunks": len(content_chunks)}
        
        return jsonify({
            "ok": True,
            "training_results": training_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/test", methods=["POST"])
def run_tests():
    """Run comprehensive ML tests"""
    try:
        if not ml_system:
            return jsonify({"ok": False, "error": "ML system not initialized"})
        
        # Initialize testing framework
        testing_framework = MLTestingFramework(ml_system)
        
        # Run tests
        test_results = testing_framework.run_comprehensive_tests()
        
        # Save results
        os.makedirs("models/results", exist_ok=True)
        with open("models/results/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        # Generate report
        report = testing_framework.generate_test_report()
        with open("models/results/test_report.txt", "w") as f:
            f.write(report)
        
        return jsonify({
            "ok": True,
            "test_results": test_results,
            "report": report,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in running tests: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/performance")
def get_performance_metrics():
    """Get ML model performance metrics"""
    try:
        # Load latest test results
        results_file = "models/results/test_results.json"
        if not os.path.exists(results_file):
            return jsonify({
                "ok": False,
                "error": "No test results available. Run tests first."
            })
        
        with open(results_file, "r") as f:
            test_results = json.load(f)
        
        # Extract performance metrics
        metrics = {}
        for test_name, test_data in test_results.get("tests", {}).items():
            if "accuracy" in test_data:
                metrics[test_name] = {
                    "accuracy": test_data["accuracy"],
                    "metric_type": "classification"
                }
            elif "mae" in test_data:
                metrics[test_name] = {
                    "mae": test_data["mae"],
                    "metric_type": "regression"
                }
            elif "avg_similarity" in test_data:
                metrics[test_name] = {
                    "avg_similarity": test_data["avg_similarity"],
                    "metric_type": "retrieval"
                }
        
        return jsonify({
            "ok": True,
            "metrics": metrics,
            "last_updated": test_results.get("timestamp", "unknown")
        })
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/dashboard")
def ml_dashboard():
    """ML Performance Dashboard"""
    return render_template("ml_dashboard.html")

@app.route("/ml/export/results")
def export_results():
    """Export ML results and models"""
    try:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "results": {}
        }
        
        # Export model status
        if ml_system:
            export_data["models"] = {
                "question_classifier": ml_system.question_classifier is not None,
                "answer_evaluator": ml_system.answer_evaluator is not None,
                "content_retriever": ml_system.content_retriever is not None,
                "difficulty_predictor": ml_system.difficulty_predictor is not None
            }
        
        # Export test results
        results_file = "models/results/test_results.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                export_data["results"] = json.load(f)
        
        return jsonify({
            "ok": True,
            "export_data": export_data
        })
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/ml/batch/predict", methods=["POST"])
def batch_predict():
    """Batch prediction for multiple questions"""
    try:
        data = request.get_json()
        questions = data.get("questions", [])
        
        if not questions:
            return jsonify({"ok": False, "error": "Questions list is required"})
        
        if not ml_system:
            return jsonify({"ok": False, "error": "ML system not initialized"})
        
        results = []
        
        for question in questions:
            question_result = {
                "question": question,
                "predictions": {}
            }
            
            # Question type prediction
            if ml_system.question_classifier:
                question_result["predictions"]["question_type"] = ml_system.question_classifier.predict(question)
            
            # Difficulty prediction
            if ml_system.difficulty_predictor:
                question_result["predictions"]["difficulty"] = ml_system.difficulty_predictor.predict(question)
            
            results.append(question_result)
        
        return jsonify({
            "ok": True,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"ok": False, "error": str(e)})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"ok": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting ML Prediction API...")
    app.run(debug=True, host='127.0.0.1', port=5002)
