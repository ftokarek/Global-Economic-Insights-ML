#!/usr/bin/env python3
"""
Test script to verify all components are working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.data.loaders import load_dataset, list_countries
        print("✅ Data loaders imported successfully")
    except Exception as e:
        print(f"❌ Data loaders import failed: {e}")
        return False
    
    try:
        from src.features.benchmarks import attach_benchmarks
        print("✅ Benchmarks imported successfully")
    except Exception as e:
        print(f"❌ Benchmarks import failed: {e}")
        return False
    
    try:
        from src.models.train import train_elasticnet, train_lightgbm
        print("✅ Model training imported successfully")
    except Exception as e:
        print(f"❌ Model training import failed: {e}")
        return False
    
    try:
        from src.models.clustering import fit_kmeans_pca
        print("✅ Clustering imported successfully")
    except Exception as e:
        print(f"❌ Clustering import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from src.data.loaders import load_dataset, list_countries
        df = load_dataset()
        
        if df.empty:
            print("❌ Dataset is empty")
            return False
        
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        countries = list_countries(df)
        print(f"✅ Countries loaded: {len(countries)} countries")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_benchmarks():
    """Test benchmarking functionality"""
    print("\nTesting benchmarks...")
    
    try:
        from src.data.loaders import load_dataset
        from src.features.benchmarks import attach_benchmarks
        
        df = load_dataset()
        if df.empty:
            print("❌ Cannot test benchmarks with empty dataset")
            return False
        
        df_with_benchmarks = attach_benchmarks(df)
        print(f"✅ Benchmarks attached: {df_with_benchmarks.shape[1]} columns")
        
        return True
    except Exception as e:
        print(f"❌ Benchmarks failed: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("\nTesting model training...")
    
    try:
        from src.data.loaders import load_dataset
        from src.models.train import train_elasticnet, train_lightgbm
        
        df = load_dataset()
        if df.empty:
            print("❌ Cannot test models with empty dataset")
            return False
        
        # Test ElasticNet
        elasticnet_result = train_elasticnet(df)
        print(f"✅ ElasticNet trained: R² = {elasticnet_result.cv_metrics['R2']:.3f}")
        
        # Test LightGBM
        try:
            lightgbm_result = train_lightgbm(df)
            print(f"✅ LightGBM trained: R² = {lightgbm_result.cv_metrics['R2']:.3f}")
        except Exception as e:
            print(f"⚠️ LightGBM failed (expected if not available): {e}")
        
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_clustering():
    """Test clustering functionality"""
    print("\nTesting clustering...")
    
    try:
        from src.data.loaders import load_dataset
        from src.models.clustering import fit_kmeans_pca
        
        df = load_dataset()
        if df.empty:
            print("❌ Cannot test clustering with empty dataset")
            return False
        
        cluster_result = fit_kmeans_pca(df, k=4, random_state=42)
        print(f"✅ Clustering completed: {len(cluster_result.pca_coords)} countries clustered")
        
        return True
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("WORLD ECONOMIC INSIGHTS 2025 - COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_loading,
        test_benchmarks,
        test_model_training,
        test_clustering
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
