#!/usr/bin/env python3
"""
Simple test script to verify the mental health chat application components.
"""

import pandas as pd
import os
import sys

def test_dataset():
    """Test if the dataset file exists and has correct format."""
    print("Testing dataset...")
    
    if not os.path.exists("Dataset.csv"):
        print("❌ Dataset.csv not found!")
        return False
    
    try:
        df = pd.read_csv("Dataset.csv")
        print(f"✅ Dataset loaded successfully with {len(df)} rows")
        
        # Check required columns
        required_columns = ["Context", "Response"]
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing required column: {col}")
                return False
        
        print("✅ Dataset has required columns")
        
        # Check for missing values
        missing_context = df["Context"].isna().sum()
        missing_response = df["Response"].isna().sum()
        
        if missing_context > 0 or missing_response > 0:
            print(f"⚠️  Found {missing_context} missing contexts and {missing_response} missing responses")
        else:
            print("✅ No missing values found")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_requirements():
    """Test if required packages can be imported."""
    print("\nTesting package imports...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "requests",
        "sentence_transformers",
        "faiss",
        "numpy",
        "torch",
        "transformers"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages imported successfully")
    return True

def test_embedding_model():
    """Test if the embedding model can be loaded."""
    print("\nTesting embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load a small model for testing
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded successfully")
        
        # Test encoding
        test_text = ["This is a test sentence"]
        embeddings = model.encode(test_text, show_progress_bar=False)
        print(f"✅ Test encoding successful, shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with embedding model: {e}")
        return False

def test_faiss():
    """Test if FAISS can be used."""
    print("\nTesting FAISS...")
    
    try:
        import faiss
        import numpy as np
        
        # Create a simple test index
        dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        index = faiss.IndexFlatIP(dim)
        
        # Add some test vectors
        test_vectors = np.random.random((10, dim)).astype('float32')
        faiss.normalize_L2(test_vectors)
        index.add(test_vectors)
        
        print(f"✅ FAISS index created successfully with {index.ntotal} vectors")
        
        # Test search
        query = np.random.random((1, dim)).astype('float32')
        faiss.normalize_L2(query)
        D, I = index.search(query, k=3)
        print("✅ FAISS search successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with FAISS: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Mental Health Chat Application Components\n")
    
    tests = [
        test_dataset,
        test_requirements,
        test_embedding_model,
        test_faiss
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        print("\nTo run the application:")
        print("  streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the application.")
        sys.exit(1)

if __name__ == "__main__":
    main() 