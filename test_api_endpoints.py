#!/usr/bin/env python3
"""
Test ML API Endpoints
====================

This script tests all the ML API endpoints to show you how to use them.
"""

import requests
import json
import time

def test_api_endpoints():
    """Test all ML API endpoints"""
    base_url = "http://127.0.0.1:5002/ml"
    
    print("🤖 OS Study Bot - ML API Endpoint Testing")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print()
    
    # Test 1: System Status
    print("1️⃣ Testing System Status...")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['ok']}")
            print(f"   Models Available: {data['models']}")
            print(f"   Data Status: {data['data']}")
        else:
            print(f"❌ Status Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("   Make sure the ML dashboard is running: python run_ml_training.py dashboard")
        return
    
    print()
    
    # Test 2: Question Type Prediction
    print("2️⃣ Testing Question Type Prediction...")
    test_questions = [
        "What is an operating system?",
        "Differentiate between process and thread",
        "List the types of operating systems",
        "Explain process scheduling algorithms"
    ]
    
    for question in test_questions:
        try:
            response = requests.post(f"{base_url}/predict/question-type", 
                                   json={"question": question}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['ok']:
                    prediction = data['prediction']
                    print(f"   Q: {question[:40]}...")
                    print(f"   📝 Type: {prediction['category']} (confidence: {prediction['confidence']:.3f})")
                else:
                    print(f"   ❌ Error: {data.get('error', 'Unknown')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
    
    # Test 3: Difficulty Prediction
    print("3️⃣ Testing Difficulty Prediction...")
    for question in test_questions[:2]:  # Test first 2 questions
        try:
            response = requests.post(f"{base_url}/predict/difficulty", 
                                   json={"question": question}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['ok']:
                    prediction = data['prediction']
                    print(f"   Q: {question[:40]}...")
                    print(f"   📊 Difficulty: {prediction['difficulty']} (confidence: {prediction['confidence']:.3f})")
                else:
                    print(f"   ❌ Error: {data.get('error', 'Unknown')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
    
    # Test 4: Content Retrieval
    print("4️⃣ Testing Content Retrieval...")
    test_queries = ["process scheduling", "memory management", "operating system"]
    
    for query in test_queries:
        try:
            response = requests.post(f"{base_url}/retrieve/content", 
                                   json={"query": query, "top_k": 3}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['ok']:
                    results = data['results']
                    print(f"   Query: '{query}'")
                    print(f"   🔍 Found: {len(results)} results")
                    if results:
                        print(f"   Top similarity: {results[0]['similarity']:.3f}")
                        print(f"   Content preview: {results[0]['content'][:80]}...")
                else:
                    print(f"   ❌ Error: {data.get('error', 'Unknown')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
    
    # Test 5: Comprehensive Prediction
    print("5️⃣ Testing Comprehensive Prediction...")
    test_question = "What is process scheduling?"
    try:
        response = requests.post(f"{base_url}/predict/comprehensive", 
                               json={"question": test_question}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['ok']:
                predictions = data['results']['predictions']
                print(f"   Q: {test_question}")
                print(f"   📝 Question Type: {predictions.get('question_type', {}).get('category', 'N/A')}")
                print(f"   📊 Difficulty: {predictions.get('difficulty', {}).get('difficulty', 'N/A')}")
                print(f"   🔍 Similar Content: {predictions.get('similar_content', {}).get('results_count', 0)} results")
            else:
                print(f"   ❌ Error: {data.get('error', 'Unknown')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 6: Answer Score Prediction
    print("6️⃣ Testing Answer Score Prediction...")
    test_answer = "Process scheduling is the method by which processes are assigned to run on the CPU."
    try:
        response = requests.post(f"{base_url}/predict/answer-score", 
                               json={"question": test_question, "answer": test_answer}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['ok']:
                prediction = data['prediction']
                print(f"   Q: {test_question}")
                print(f"   A: {test_answer[:50]}...")
                print(f"   📈 Predicted Score: {prediction['predicted_score']:.1f}/100")
            else:
                print(f"   ❌ Error: {data.get('error', 'Unknown')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("=" * 60)
    print("✅ API Endpoint Testing Complete!")
    print()
    print("🌐 Available URLs:")
    print(f"   ML Dashboard: http://127.0.0.1:5002/ml/dashboard")
    print(f"   API Status: http://127.0.0.1:5002/ml/status")
    print(f"   API Base: http://127.0.0.1:5002/ml/")
    print()
    print("📚 Available Endpoints:")
    print("   GET  /ml/status                    - System status")
    print("   POST /ml/predict/question-type     - Question classification")
    print("   POST /ml/predict/difficulty        - Difficulty prediction")
    print("   POST /ml/predict/answer-score      - Answer evaluation")
    print("   POST /ml/retrieve/content          - Content retrieval")
    print("   POST /ml/predict/comprehensive     - All predictions")
    print("   POST /ml/train                     - Train models")
    print("   POST /ml/test                      - Run tests")
    print("   GET  /ml/performance               - Performance metrics")
    print("   GET  /ml/dashboard                 - Web dashboard")

if __name__ == "__main__":
    test_api_endpoints()
