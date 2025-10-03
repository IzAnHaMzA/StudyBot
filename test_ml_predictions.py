#!/usr/bin/env python3
"""
Simple ML Prediction Test Script
===============================

This script demonstrates how to test the ML predictions without the interactive mode.
"""

import sys
import os
import json
import requests
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_predictions():
    """Test ML predictions using the API"""
    print("🤖 OS Study Bot - ML Prediction Test")
    print("=" * 50)
    
    # Test questions
    test_questions = [
        "What is an operating system?",
        "Differentiate between process and thread",
        "List the types of operating systems",
        "Explain process scheduling algorithms",
        "What is memory management?",
        "How does virtual memory work?"
    ]
    
    # Test answers
    test_answers = [
        "An operating system is system software that manages computer hardware and software resources.",
        "A process is a program in execution while a thread is a lightweight process within a process.",
        "Batch, time-sharing, real-time, distributed, and network operating systems.",
        "Process scheduling is the method by which processes are assigned to run on the CPU using various algorithms.",
        "Memory management is the process of controlling and coordinating computer memory.",
        "Virtual memory allows programs to use more memory than physically available by using disk storage."
    ]
    
    print(f"Testing {len(test_questions)} questions...")
    print()
    
    for i, (question, answer) in enumerate(zip(test_questions, test_answers), 1):
        print(f"Test {i}: {question}")
        print("-" * 60)
        
        try:
            # Test comprehensive prediction
            response = requests.post('http://127.0.0.1:5002/ml/predict/comprehensive', 
                                   json={'question': question}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['ok']:
                    predictions = data['results']['predictions']
                    
                    # Question type
                    if 'question_type' in predictions:
                        qc = predictions['question_type']
                        print(f"📝 Question Type: {qc['category']} (confidence: {qc['confidence']:.3f})")
                    
                    # Difficulty
                    if 'difficulty' in predictions:
                        diff = predictions['difficulty']
                        print(f"📊 Difficulty: {diff['difficulty']} (confidence: {diff['confidence']:.3f})")
                    
                    # Content retrieval
                    if 'similar_content' in predictions:
                        content = predictions['similar_content']
                        print(f"🔍 Similar Content: {content['results_count']} results found")
                        if content['results_count'] > 0:
                            print(f"   Top similarity: {content['top_similarity']:.3f}")
                    
                    # Answer evaluation
                    try:
                        answer_response = requests.post('http://127.0.0.1:5002/ml/predict/answer-score',
                                                      json={'question': question, 'answer': answer}, timeout=10)
                        if answer_response.status_code == 200:
                            answer_data = answer_response.json()
                            if answer_data['ok']:
                                score = answer_data['prediction']['predicted_score']
                                print(f"📈 Predicted Score: {score:.1f}/100")
                    except:
                        print("📈 Predicted Score: API not available")
                    
                else:
                    print(f"❌ Error: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: ML API server not running")
            print("   Start it with: python run_ml_training.py dashboard")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 50)
    print("✅ ML Prediction Test Complete!")

def test_direct_ml_system():
    """Test ML system directly without API"""
    print("🤖 OS Study Bot - Direct ML System Test")
    print("=" * 50)
    
    try:
        from ml_training_system import MLTrainingSystem
        from ml_training_system import QuestionClassifier, DifficultyPredictor, ContentRetriever
        
        # Initialize ML system
        ml_system = MLTrainingSystem()
        
        # Load models
        if os.path.exists("models/trained/question_classifier.pkl"):
            ml_system.question_classifier = QuestionClassifier()
            ml_system.question_classifier.load_model("models/trained/question_classifier.pkl")
            print("✅ Question Classifier loaded")
        else:
            print("❌ Question Classifier not found")
            return
        
        if os.path.exists("models/trained/difficulty_predictor.pkl"):
            ml_system.difficulty_predictor = DifficultyPredictor()
            ml_system.difficulty_predictor.load_model("models/trained/difficulty_predictor.pkl")
            print("✅ Difficulty Predictor loaded")
        else:
            print("❌ Difficulty Predictor not found")
            return
        
        # Initialize content retriever
        ml_system.content_retriever = ContentRetriever()
        syllabus_content = ml_system.load_syllabus_data()
        if syllabus_content:
            content_chunks = ml_system.content_retriever.prepare_content_data(syllabus_content)
            ml_system.content_retriever.train(content_chunks)
            print("✅ Content Retriever initialized")
        
        print()
        
        # Test questions
        test_questions = [
            "What is an operating system?",
            "Differentiate between process and thread",
            "List the types of operating systems",
            "Explain process scheduling algorithms"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"Test {i}: {question}")
            print("-" * 50)
            
            # Question classification
            qc_result = ml_system.question_classifier.predict(question)
            print(f"📝 Question Type: {qc_result.get('category', 'Unknown')}")
            print(f"   Confidence: {qc_result.get('confidence', 0):.3f}")
            
            # Difficulty prediction
            diff_result = ml_system.difficulty_predictor.predict(question)
            print(f"📊 Difficulty: {diff_result.get('difficulty', 'Unknown')}")
            print(f"   Confidence: {diff_result.get('confidence', 0):.3f}")
            
            # Content retrieval
            content_results = ml_system.content_retriever.retrieve_similar_content(question, top_k=3)
            print(f"🔍 Similar Content: {len(content_results)} results")
            if content_results:
                print(f"   Top similarity: {content_results[0]['similarity']:.3f}")
                print(f"   Content preview: {content_results[0]['content'][:100]}...")
            
            print()
        
        print("=" * 50)
        print("✅ Direct ML System Test Complete!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all ML dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("Choose testing method:")
    print("1. API-based testing (requires ML dashboard running)")
    print("2. Direct ML system testing")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_ml_predictions()
    elif choice == "2":
        test_direct_ml_system()
    elif choice == "3":
        test_direct_ml_system()
        print("\n" + "="*60 + "\n")
        test_ml_predictions()
    else:
        print("Invalid choice. Running direct test...")
        test_direct_ml_system()

if __name__ == "__main__":
    main()
