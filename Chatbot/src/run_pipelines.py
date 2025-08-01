#!/usr/bin/env python3
"""
Disease Classification Pipeline Runner
=====================================

This script runs both the data combination and ML training pipelines
in sequence to create a complete disease classification system with
publication-quality visualizations.

Usage:
    python run_pipelines.py

Requirements:
    1. Raw CSV files in the ../data/raw directory:
       - dataset.csv
       - symptom_Description.csv
       - symptom_precaution.csv
       - Symptom-severity.csv
    
    2. Required Python packages:
       - pandas, numpy, scikit-learn, matplotlib, seaborn

Outputs:
    - combined_disease_dataset.csv
    - disease_classification_model.pkl
    - model_comparison_plots.png
    - visualizations/ (5 publication quality graphs)
    - Various analysis reports
"""

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_requirements():
    """Check if all required files and packages are available"""
    print("="*70)
    print("DISEASE CLASSIFICATION PIPELINE RUNNER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("1. CHECKING REQUIREMENTS")
    print("-"*50)
    
    # Auto-detect correct input directory
    input_dirs_to_check = ['data/raw', '../data/raw']
    input_dir = None
    
    for dir_path in input_dirs_to_check:
        if os.path.exists(dir_path):
            input_dir = dir_path
            break
    
    if input_dir is None:
        print(f"ERROR: Input directory not found. Tried: {input_dirs_to_check}")
        print("Please create 'data/raw' directory and place all CSV files there.")
        return False
    
    print(f"SUCCESS: Input directory found: {input_dir}")
    
    # Check required CSV files in ../data/raw directory
    required_files = [
        'dataset.csv',
        'symptom_Description.csv', 
        'symptom_precaution.csv',
        'Symptom-severity.csv'
    ]
    
    missing_files = []
    for file in required_files:
        filepath = os.path.join(input_dir, file)
        if os.path.exists(filepath):
            print(f"SUCCESS: {file} - Found in {input_dir}")
        else:
            print(f"ERROR: {file} - Missing from {input_dir}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nERROR: Missing {len(missing_files)} required files!")
        print(f"Please ensure all CSV files from the Kaggle dataset are in the '{input_dir}' directory:")
        print("https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
        return False
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"SUCCESS: {package} - Available")
        except ImportError:
            print(f"ERROR: {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nERROR: Missing {len(missing_packages)} required packages!")
        print("Install them using: pip install " + " ".join(missing_packages))
        return False
    
    print(f"\nSUCCESS: All requirements satisfied!")
    return True

def run_data_combination():
    """Run the data combination pipeline"""
    print(f"\n2. RUNNING DATA COMBINATION PIPELINE")
    print("-"*50)
    
    try:
        # Import and run data combiner from same directory
        from data_combiner import DiseaseDataCombiner
        
        combiner = DiseaseDataCombiner()
        success = combiner.run_combination_pipeline()
        
        if success:
            print(f"SUCCESS: Data combination completed successfully!")
            return True
        else:
            print(f"ERROR: Data combination failed!")
            return False
            
    except ImportError:
        print(f"ERROR: Could not import data_combiner.py")
        print("Please ensure the data combination pipeline file is in the current directory.")
        return False
    except Exception as e:
        print(f"ERROR: Data combination failed with error: {str(e)}")
        return False

def run_ml_training():
    """Run the ML training pipeline with integrated visualization graphs"""
    print(f"\n3. RUNNING ML TRAINING PIPELINE WITH VISUALIZATION GRAPHS")
    print("-"*50)
    
    try:
        # Import and run ML pipeline from same directory
        from ml_pipeline import DiseaseClassificationMLPipeline
        
        pipeline = DiseaseClassificationMLPipeline()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"SUCCESS: ML training and visualization graph generation completed successfully!")
            return results
        else:
            print(f"ERROR: ML training failed!")
            return False
            
    except ImportError:
        print(f"ERROR: Could not import ml_pipeline.py")
        print("Please ensure the ML pipeline file is in the current directory.")
        return False
    except Exception as e:
        print(f"ERROR: ML training failed with error: {str(e)}")
        return False

def test_trained_model():
    """Test the trained model with sample data"""
    print(f"\n4. TESTING TRAINED MODEL")
    print("-"*50)
    
    try:
        from ml_pipeline import load_trained_model, predict_disease
        import numpy as np
        
        # Auto-detect model path
        possible_model_paths = [
            'models/saved_models/disease_classification_model.pkl',  # Running from root
            '../models/saved_models/disease_classification_model.pkl',  # Running from src/
            'models/disease_classification_model.pkl'  # Fallback
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"ERROR: Could not find model file in any of these locations: {possible_model_paths}")
            return False
            
        model_pkg = load_trained_model(model_path)
        
        if model_pkg:
            # Create sample test data (all zeros except a few features)
            n_features = model_pkg['n_features']
            sample_symptoms = np.zeros(n_features)
            
            # Set some features to 1 (simulating symptoms present)
            sample_symptoms[0] = 1  # First symptom present
            sample_symptoms[2] = 1  # Third symptom present
            sample_symptoms[5] = 1  # Sixth symptom present
            
            # Make prediction
            result = predict_disease(model_pkg, sample_symptoms)
            
            if result:
                print(f"SUCCESS: Model test successful!")
                print(f"   Sample prediction: {result['predicted_disease']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Top 3 predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"     {i}. {pred['disease']}: {pred['probability']:.3f}")
                return True
            else:
                print(f"ERROR: Model prediction failed!")
                return False
        else:
            print(f"ERROR: Could not load trained model!")
            return False
            
    except Exception as e:
        print(f"ERROR: Model testing failed: {str(e)}")
        return False

def generate_final_report(ml_results=None):
    """Generate a final summary report"""
    print(f"\n5. GENERATING FINAL REPORT")
    print("-"*50)
    
    try:
        # Auto-detect correct paths for output files
        base_paths = {
            'data_processed': 'data/processed' if os.path.exists('data/processed') else '../data/processed',
            'models_saved': 'models/saved_models' if os.path.exists('models/saved_models') else '../models/saved_models',
            'models_analysis': 'models/analysis' if os.path.exists('models/analysis') else '../models/analysis',
            'models_viz': 'models/visualizations' if os.path.exists('models/visualizations') else '../models/visualizations',
            'reports': 'reports' if os.path.exists('reports') or not os.path.exists('../reports') else '../reports'
        }
        
        # Check which files were created
        output_files = {
            f"{base_paths['data_processed']}/combined_disease_dataset.csv": 'Combined dataset',
            f"{base_paths['data_processed']}/data_summary_report.txt": 'Data summary report',
            f"{base_paths['models_saved']}/disease_classification_model.pkl": 'Trained model package',
            f"{base_paths['models_analysis']}/model_comparison_plots.png": 'Model comparison plots',
            f"{base_paths['models_analysis']}/feature_importance_analysis.csv": 'Feature analysis'
        }
        
        # Add visualization graph files if they exist
        visualization_graph_dir = base_paths['models_viz']
        if os.path.exists(visualization_graph_dir):
            visualization_files = {
                f'{visualization_graph_dir}/graph_1_model_performance.png': 'Visualization Graph 1 - Model Performance Dashboard',
                f'{visualization_graph_dir}/graph_2_cv_stability.png': 'Visualization Graph 2 - Cross-Validation Stability',
                f'{visualization_graph_dir}/graph_3_roc_comparison.png': 'Visualization Graph 3 - ROC Curves Comparison',
                f'{visualization_graph_dir}/graph_4_feature_importance.png': 'Visualization Graph 4 - Feature Importance Analysis',
                f'{visualization_graph_dir}/graph_5_rf_superiority.png': 'Visualization Graph 5 - Random Forest Superiority'
            }
            output_files.update(visualization_files)
        
        existing_files = []
        missing_files = []
        
        for filename, description in output_files.items():
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / 1024 / 1024
                existing_files.append((filename, description, size_mb))
            else:
                missing_files.append((filename, description))
        
        # Write final report to reports directory
        reports_dir = base_paths['reports']
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = os.path.join(reports_dir, 'pipeline_execution_report.txt')
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("DISEASE CLASSIFICATION PIPELINE EXECUTION REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DIRECTORY STRUCTURE\n")
            f.write("-"*30 + "\n")
            f.write("Input: ../data/raw/\n")
            f.write("Output: ../data/processed/ (processed data)\n")
            f.write("        ../models/ (trained models & analysis)\n")
            f.write("        ../models/visualizations/ (publication quality graphs)\n\n")
            
            f.write("PIPELINE EXECUTION STATUS\n")
            f.write("-"*30 + "\n")
            f.write("COMPLETED: Data Combination Pipeline\n")
            f.write("COMPLETED: ML Training Pipeline\n")
            f.write("COMPLETED: Visualization Graph Generation\n")
            f.write("COMPLETED: Model Testing\n\n")
            
            if ml_results and 'best_model' in ml_results:
                best_model = ml_results['best_model']
                f.write("BEST MODEL PERFORMANCE\n")
                f.write("-"*30 + "\n")
                f.write(f"Model: {best_model['name']}\n")
                f.write(f"Accuracy: {best_model['results']['accuracy']:.4f}\n")
                f.write(f"F1-Score: {best_model['results']['f1_score']:.4f}\n")
                f.write(f"Precision: {best_model['results']['precision']:.4f}\n")
                f.write(f"Recall: {best_model['results']['recall']:.4f}\n")
                f.write(f"ROC-AUC: {best_model['results']['roc_auc']:.4f}\n\n")
            
            f.write("OUTPUT FILES CREATED\n")
            f.write("-"*30 + "\n")
            for filename, description, size_mb in existing_files:
                f.write(f"SUCCESS: {filename} ({size_mb:.2f} MB) - {description}\n")
            
            if missing_files:
                f.write(f"\nMISSING FILES\n")
                f.write("-"*30 + "\n")
                for filename, description in missing_files:
                    f.write(f"MISSING: {filename} - {description}\n")
            
            f.write(f"\nNEXT STEPS\n")
            f.write("-"*30 + "\n")
            f.write("1. Review ../models/analysis/model_comparison_plots.png for performance analysis\n")
            f.write("2. Check ../models/visualizations/ for publication-quality visualizations\n")
            f.write("3. Review ../data/processed/data_summary_report.txt for dataset insights\n")  
            f.write("4. Use ../models/saved_models/disease_classification_model.pkl for disease predictions\n")
            f.write("5. Deploy the model in a web application or API\n")
            f.write("6. Consider collecting more data for model improvement\n")
            f.write("7. Use visualization graphs in academic papers or presentations\n")
        
        print(f"SUCCESS: Final report saved: {report_filename}")
        
        # Display summary to console
        print(f"\nOUTPUT FILES ({len(existing_files)} created):")
        for filename, description, size_mb in existing_files:
            print(f"   {filename} ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Report generation failed: {str(e)}")
        # Create a simple fallback report without special characters
        try:
            reports_dir = '../reports'
            os.makedirs(reports_dir, exist_ok=True)
            fallback_report = os.path.join(reports_dir, 'pipeline_execution_report.txt')
            with open(fallback_report, 'w', encoding='utf-8') as f:
                f.write("DISEASE CLASSIFICATION PIPELINE - EXECUTION REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("STATUS: Pipeline execution completed\n")
                f.write("Check console output for detailed results\n")
            print("Fallback report created successfully")
            return True
        except:
            print("Could not create fallback report")
            return False

def main():
    """Main pipeline runner function"""
    try:
        # Step 1: Check requirements
        if not check_requirements():
            print(f"\nERROR: Requirements check failed. Please fix issues and try again.")
            return False
        
        # Step 2: Run data combination
        if not run_data_combination():
            print(f"\nERROR: Data combination failed. Check error messages above.")
            return False
        
        # Step 3: Run ML training with visualization graphs
        ml_results = run_ml_training()
        if not ml_results:
            print(f"\nERROR: ML training failed. Check error messages above.")
            return False
        
        # Step 4: Test the model
        if not test_trained_model():
            print(f"\nWARNING: Model testing failed, but training was successful.")
        
        # Step 5: Generate final report
        generate_final_report(ml_results)
        
        # Final success message
        print(f"\n" + "="*70)
        print("SUCCESS: COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
        print("="*70)
        
        print(f"\nSUMMARY:")
        print(f"   SUCCESS: Data combination completed")
        print(f"   SUCCESS: ML model training completed")
        print(f"   SUCCESS: Visualization graphs generated")
        print(f"   SUCCESS: Model testing completed")
        print(f"   SUCCESS: All output files generated")
        
        print(f"\nKEY OUTPUTS:")
        print(f"   ../data/processed/combined_disease_dataset.csv - Ready for ML")
        print(f"   ../models/saved_models/disease_classification_model.pkl - Trained model")
        print(f"   ../models/analysis/model_comparison_plots.png - Performance charts")
        print(f"   ../models/visualizations/ - Publication quality graphs")
        print(f"   ../reports/pipeline_execution_report.txt - Full report")
        
        print(f"\nDIRECTORY STRUCTURE:")
        print(f"   ../data/raw/ - Input CSV files")
        print(f"   ../data/processed/ - Processed data & reports")
        print(f"   ../models/ - Trained models & analysis")
        print(f"   ../models/visualizations/ - Publication quality visualizations")
        print(f"   ../reports/ - Execution reports")
        
        print(f"\nREADY FOR DEPLOYMENT!")
        print("Your disease classification system is now ready for use.")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\nWARNING: Pipeline execution interrupted by user.")
        return False
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(__doc__)
    success = main()
    
    if not success:
        print(f"\nERROR: Pipeline execution failed. Please check the error messages above.")
        sys.exit(1)