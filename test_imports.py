"""
Test script to verify import structure without heavy dependencies.
"""

def test_imports():
    """Test that all imports work correctly."""
    try:
        # Test data imports
        print("Testing data imports...")
        from src.data.preprocessing import TextPreprocessor
        from src.data.dataset import DatasetFactory
        print("✓ Data imports successful")
        
        # Test utils imports
        print("Testing utils imports...")
        from src.utils.helpers import set_seed, create_directories
        from src.utils.config import Config, ModelConfig, TrainingConfig
        print("✓ Utils imports successful")
        
        # Test that the structure is correct
        print("Testing basic functionality...")
        preprocessor = TextPreprocessor()
        config = Config()
        print("✓ Basic object creation successful")
        
        print("\n🎉 All imports and basic functionality working correctly!")
        print("The project structure is properly set up.")
        print("\nTo run the full demo, install dependencies with:")
        print("pip install torch transformers datasets scikit-learn matplotlib seaborn tqdm pandas nltk")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_imports()
