#!/usr/bin/env python3
"""
OS Study Bot - ML Training Runner
================================

This script provides a command-line interface for training, testing, and managing
ML models for the OS Study Bot system.

Usage:
    python run_ml_training.py train          # Train all models
    python run_ml_training.py test           # Run comprehensive tests
    python run_ml_training.py predict        # Interactive prediction mode
    python run_ml_training.py dashboard      # Start ML dashboard
    python run_ml_training.py status         # Check system status

Author: OS Study Bot Team
Version: 1.0
"""

import sys
import os
import argparse
import json
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_training_system import MLTrainingSystem, MLTestingFramework
from ml_prediction_api import initialize_ml_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_models():
    """Train all ML models"""
    logger.info("Starting ML model training...")
    
    try:
        # Initialize ML system
        ml_system = MLTrainingSystem()
        
        # Parse questions from syllabus
        questions_data = ml_system.parse_questions_from_syllabus()
        
        if not questions_data:
            logger.error("No questions found in syllabus. Please check the syllabus file.")
            return False
        
        logger.info(f"Found {len(questions_data['all'])} questions in syllabus")
        
        # Initialize models
        from ml_training_system import QuestionClassifier, AnswerEvaluator, ContentRetriever, DifficultyPredictor
        
        ml_system.question_classifier = QuestionClassifier()
        ml_system.answer_evaluator = AnswerEvaluator()
        ml_system.content_retriever = ContentRetriever()
        ml_system.difficulty_predictor = DifficultyPredictor()
        
        training_results = {}
        
        # Train Question Classifier
        logger.info("Training Question Classifier...")
        X_q, y_q = ml_system.question_classifier.prepare_training_data(questions_data)
        if X_q and y_q:
            qc_results = ml_system.question_classifier.train(X_q, y_q)
            ml_system.question_classifier.save_model("models/trained/question_classifier.pkl")
            training_results["question_classifier"] = qc_results
            logger.info(f"Question Classifier trained - Accuracy: {qc_results['accuracy']:.3f}")
        
        # Train Content Retriever
        logger.info("Training Content Retriever...")
        syllabus_content = ml_system.load_syllabus_data()
        if syllabus_content:
            content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
            ml_system.content_retriever.train(content_chunks)
            training_results["content_retriever"] = {"status": "trained", "chunks": len(content_chunks)}
            logger.info(f"Content Retriever trained on {len(content_chunks)} chunks")
        
        # Train Difficulty Predictor
        logger.info("Training Difficulty Predictor...")
        X_d, y_d = ml_system.difficulty_predictor.prepare_training_data(questions_data)
        if X_d and y_d:
            dp_results = ml_system.difficulty_predictor.train(X_d, y_d)
            ml_system.difficulty_predictor.save_model("models/trained/difficulty_predictor.pkl")
            training_results["difficulty_predictor"] = dp_results
            logger.info(f"Difficulty Predictor trained - Accuracy: {dp_results['accuracy']:.3f}")
        
        # Save training results
        os.makedirs("models/results", exist_ok=True)
        with open("models/results/training_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": training_results
            }, f, indent=2)
        
        logger.info("All models trained successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_tests():
    """Run comprehensive ML tests"""
    logger.info("Starting comprehensive ML testing...")
    
    try:
        # Initialize ML system
        ml_system = MLTrainingSystem()
        
        # Load trained models
        from ml_training_system import QuestionClassifier, DifficultyPredictor
        
        if os.path.exists("models/trained/question_classifier.pkl"):
            ml_system.question_classifier = QuestionClassifier()
            ml_system.question_classifier.load_model("models/trained/question_classifier.pkl")
            logger.info("Question Classifier loaded")
        
        if os.path.exists("models/trained/difficulty_predictor.pkl"):
            ml_system.difficulty_predictor = DifficultyPredictor()
            ml_system.difficulty_predictor.load_model("models/trained/difficulty_predictor.pkl")
            logger.info("Difficulty Predictor loaded")
        
        # Initialize other components
        from ml_training_system import AnswerEvaluator, ContentRetriever
        
        if not ml_system.answer_evaluator:
            ml_system.answer_evaluator = AnswerEvaluator()
        
        if not ml_system.content_retriever:
            ml_system.content_retriever = ContentRetriever()
            syllabus_content = ml_system.load_syllabus_data()
            if syllabus_content:
                content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
                ml_system.content_retriever.train(content_chunks)
                logger.info("Content Retriever initialized")
        
        # Run tests
        testing_framework = MLTestingFramework(ml_system)
        test_results = testing_framework.run_comprehensive_tests()
        
        # Save results
        os.makedirs("models/results", exist_ok=True)
        with open("models/results/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        # Generate and save report
        report = testing_framework.generate_test_report()
        with open("models/results/test_report.txt", "w") as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("ML TESTING REPORT")
        print("="*60)
        print(report)
        
        logger.info("Testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return False

def interactive_predict():
    """Interactive prediction mode"""
    logger.info("Starting interactive prediction mode...")
    
    try:
        # Initialize ML system
        ml_system = MLTrainingSystem()
        
        # Load trained models
        from ml_training_system import QuestionClassifier, DifficultyPredictor
        
        if os.path.exists("models/trained/question_classifier.pkl"):
            ml_system.question_classifier = QuestionClassifier()
            ml_system.question_classifier.load_model("models/trained/question_classifier.pkl")
            logger.info("Question Classifier loaded")
        else:
            logger.warning("Question Classifier not found. Train models first.")
            return False
        
        if os.path.exists("models/trained/difficulty_predictor.pkl"):
            ml_system.difficulty_predictor = DifficultyPredictor()
            ml_system.difficulty_predictor.load_model("models/trained/difficulty_predictor.pkl")
            logger.info("Difficulty Predictor loaded")
        else:
            logger.warning("Difficulty Predictor not found. Train models first.")
            return False
        
        # Initialize content retriever
        from ml_training_system import ContentRetriever
        if not ml_system.content_retriever:
            ml_system.content_retriever = ContentRetriever()
            syllabus_content = ml_system.load_syllabus_data()
            if syllabus_content:
                content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
                ml_system.content_retriever.train(content_chunks)
                logger.info("Content Retriever initialized")
        
        print("\n" + "="*60)
        print("OS STUDY BOT - INTERACTIVE PREDICTION MODE")
        print("="*60)
        print("Enter questions to get ML predictions. Type 'quit' to exit.")
        print()
        
        while True:
            try:
                question = input("Enter question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                print(f"\nAnalyzing: '{question}'")
                print("-" * 50)
                
                # Question type prediction
                if ml_system.question_classifier:
                    qc_result = ml_system.question_classifier.predict(question)
                    print(f"Question Type: {qc_result.get('category', 'Unknown')}")
                    print(f"Confidence: {qc_result.get('confidence', 0):.3f}")
                
                # Difficulty prediction
                if ml_system.difficulty_predictor:
                    diff_result = ml_system.difficulty_predictor.predict(question)
                    print(f"Difficulty: {diff_result.get('difficulty', 'Unknown')}")
                    print(f"Confidence: {diff_result.get('confidence', 0):.3f}")
                
                # Content retrieval
                if ml_system.content_retriever:
                    content_results = ml_system.content_retriever.retrieve_similar_content(question, top_k=3)
                    if content_results:
                        print(f"Similar Content Found: {len(content_results)} results")
                        for i, result in enumerate(content_results[:2], 1):
                            print(f"  {i}. Similarity: {result['similarity']:.3f}")
                            print(f"     Content: {result['content'][:100]}...")
                    else:
                        print("No similar content found")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nExiting interactive mode...")
        return True
        
    except Exception as e:
        logger.error(f"Interactive prediction failed: {e}")
        return False

def start_dashboard():
    """Start ML dashboard server"""
    logger.info("Starting ML dashboard server...")
    
    try:
        from ml_prediction_api import app
        print("\n" + "="*60)
        print("OS STUDY BOT - ML DASHBOARD")
        print("="*60)
        print("Starting ML dashboard server...")
        print("Dashboard will be available at: http://127.0.0.1:5002/ml/dashboard")
        print("API endpoints available at: http://127.0.0.1:5002/ml/")
        print("\nPress Ctrl+C to stop the server")
        print()
        
        app.run(debug=False, host='127.0.0.1', port=5002)
        
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}")
        return False

def check_status():
    """Check ML system status"""
    logger.info("Checking ML system status...")
    
    try:
        ml_system = MLTrainingSystem()
        
        print("\n" + "="*60)
        print("OS STUDY BOT - ML SYSTEM STATUS")
        print("="*60)
        
        # Check syllabus
        syllabus_content = ml_system.load_syllabus_data()
        print(f"Syllabus Status: {'✓ Loaded' if syllabus_content else '✗ Not Found'}")
        if syllabus_content:
            print(f"Syllabus Size: {len(syllabus_content)} characters")
        
        # Check questions
        questions_data = ml_system.parse_questions_from_syllabus()
        print(f"Questions Status: {'✓ Parsed' if questions_data else '✗ Not Found'}")
        if questions_data:
            print(f"Total Questions: {len(questions_data.get('all', []))}")
            for category, questions in questions_data.items():
                if category != 'all':
                    print(f"  {category}: {len(questions)} questions")
        
        # Check models
        print("\nModel Status:")
        models = [
            ("Question Classifier", "models/trained/question_classifier.pkl"),
            ("Difficulty Predictor", "models/trained/difficulty_predictor.pkl")
        ]
        
        for model_name, model_path in models:
            status = "✓ Trained" if os.path.exists(model_path) else "✗ Not Trained"
            print(f"  {model_name}: {status}")
        
        # Check results
        print("\nResults Status:")
        results_files = [
            ("Training Results", "models/results/training_results.json"),
            ("Test Results", "models/results/test_results.json"),
            ("Test Report", "models/results/test_report.txt")
        ]
        
        for result_name, result_path in results_files:
            status = "✓ Available" if os.path.exists(result_path) else "✗ Not Available"
            print(f"  {result_name}: {status}")
        
        print("\n" + "="*60)
        return True
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="OS Study Bot ML Training System")
    parser.add_argument("command", choices=["train", "test", "predict", "dashboard", "status"],
                       help="Command to execute")
    
    args = parser.parse_args()
    
    print("OS Study Bot - ML Training System")
    print("=" * 40)
    
    success = False
    
    if args.command == "train":
        success = train_models()
    elif args.command == "test":
        success = run_tests()
    elif args.command == "predict":
        success = interactive_predict()
    elif args.command == "dashboard":
        success = start_dashboard()
    elif args.command == "status":
        success = check_status()
    
    if success:
        print(f"\n✓ {args.command.title()} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n✗ {args.command.title()} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
