#!/usr/bin/env python3
"""
Disease Dataset Combination Pipeline
===================================

This pipeline combines multiple disease-related CSV files into a single 
comprehensive dataset ready for machine learning.

Required input files (in ../data/raw/ directory):
- dataset.csv: Main dataset with diseases and symptoms
- symptom_Description.csv: Disease descriptions  
- symptom_precaution.csv: Disease precautions
- Symptom-severity.csv: Symptom severity weights

Output (in ../data/processed/ directory):
- combined_disease_dataset.csv: Combined dataset with all information
- data_summary_report.txt: Summary statistics and validation report
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DiseaseDataCombiner:
    def __init__(self, input_dir=None, output_dir=None):
        # Auto-detect correct paths based on current working directory
        if input_dir is None:
            if os.path.exists('data/raw'):
                input_dir = 'data/raw'  # Running from root
            elif os.path.exists('../data/raw'):
                input_dir = '../data/raw'  # Running from src/
            else:
                input_dir = 'data/raw'  # Default
        
        if output_dir is None:
            if os.path.exists('data/processed') or os.path.exists('data'):
                output_dir = 'data/processed'  # Running from root
            else:
                output_dir = '../data/processed'  # Running from src/
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset = None
        self.descriptions = None
        self.precautions = None
        self.severity = None
        self.combined_df = None
        self.severity_lookup = {}
        
    def load_raw_files(self):
        """Load all required CSV files from input directory"""
        print("="*70)
        print("DISEASE DATASET COMBINATION PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n1. LOADING RAW CSV FILES")
        print("-"*50)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        required_files = {
            'dataset.csv': 'Main symptom dataset',
            'symptom_Description.csv': 'Disease descriptions',
            'symptom_precaution.csv': 'Disease precautions', 
            'Symptom-severity.csv': 'Symptom severity weights'
        }
        
        # Check if input directory exists, create if not
        if not os.path.exists(self.input_dir):
            print(f"ERROR: Input directory not found: {self.input_dir}")
            print("Please ensure the input directory exists with required CSV files")
            return False
        
        # Check if all files exist in input directory
        missing_files = []
        for filename in required_files.keys():
            filepath = os.path.join(self.input_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            print(f"ERROR: Missing files in {self.input_dir}: {', '.join(missing_files)}")
            print(f"\nPlease ensure all required files are in the '{self.input_dir}' directory:")
            for file, desc in required_files.items():
                filepath = os.path.join(self.input_dir, file)
                status = "FOUND" if os.path.exists(filepath) else "MISSING"
                print(f"  {status}: {file} - {desc}")
            return False
        
        try:
            # Load main dataset
            dataset_path = os.path.join(self.input_dir, 'dataset.csv')
            self.dataset = pd.read_csv(dataset_path)
            print(f"SUCCESS: dataset.csv loaded: {len(self.dataset):,} rows, {len(self.dataset.columns)} columns")
            
            # Load descriptions
            desc_path = os.path.join(self.input_dir, 'symptom_Description.csv')
            self.descriptions = pd.read_csv(desc_path)
            print(f"SUCCESS: symptom_Description.csv loaded: {len(self.descriptions):,} rows")
            
            # Load precautions
            prec_path = os.path.join(self.input_dir, 'symptom_precaution.csv')
            self.precautions = pd.read_csv(prec_path)
            print(f"SUCCESS: symptom_precaution.csv loaded: {len(self.precautions):,} rows")
            
            # Load severity weights
            severity_path = os.path.join(self.input_dir, 'Symptom-severity.csv')
            self.severity = pd.read_csv(severity_path)
            print(f"SUCCESS: Symptom-severity.csv loaded: {len(self.severity):,} rows")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading files: {str(e)}")
            return False
    
    def validate_raw_data(self):
        """Validate the loaded raw data"""
        print(f"\n2. VALIDATING RAW DATA")
        print("-"*50)
        
        issues = []
        
        # Check main dataset
        unique_diseases = self.dataset['Disease'].nunique()
        total_records = len(self.dataset)
        print(f"SUCCESS: Main dataset: {unique_diseases} unique diseases, {total_records:,} total records")
        
        # Check for missing values in main dataset
        missing_symptoms = self.dataset.isnull().sum().sum()
        if missing_symptoms > 0:
            print(f"WARNING: Main dataset has {missing_symptoms:,} missing values")
        
        # Validate descriptions
        desc_diseases = set(self.descriptions['Disease'].unique())
        main_diseases = set(self.dataset['Disease'].unique())
        
        desc_coverage = len(desc_diseases.intersection(main_diseases))
        print(f"SUCCESS: Descriptions: {desc_coverage}/{unique_diseases} diseases have descriptions")
        
        # Validate precautions
        prec_diseases = set(self.precautions['Disease'].unique())
        prec_coverage = len(prec_diseases.intersection(main_diseases))
        print(f"SUCCESS: Precautions: {prec_coverage}/{unique_diseases} diseases have precautions")
        
        # Validate symptom severity
        symptom_cols = [col for col in self.dataset.columns if col.startswith('Symptom_')]
        all_symptoms = set()
        
        for col in symptom_cols:
            symptoms_in_col = set(self.dataset[col].dropna().unique())
            all_symptoms.update(symptoms_in_col)
        
        severity_symptoms = set(self.severity['Symptom'].unique())
        symptom_coverage = len(all_symptoms.intersection(severity_symptoms))
        print(f"SUCCESS: Severity weights: {symptom_coverage}/{len(all_symptoms)} symptoms have weights")
        
        if len(issues) == 0:
            print("SUCCESS: All data validation checks passed!")
        else:
            print(f"WARNING: Found {len(issues)} validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        return len(issues) == 0
    
    def create_severity_lookup(self):
        """Create symptom severity lookup dictionary"""
        print(f"\n3. CREATING SYMPTOM SEVERITY LOOKUP")
        print("-"*50)
        
        # Clean symptom names and create lookup
        self.severity['Symptom_Clean'] = self.severity['Symptom'].str.strip().str.lower().str.replace('_', ' ')
        self.severity_lookup = dict(zip(self.severity['Symptom'], self.severity['weight']))
        
        print(f"SUCCESS: Created severity lookup for {len(self.severity_lookup)} symptoms")
        
        # Show weight distribution
        weight_stats = self.severity['weight'].describe()
        print(f"SUCCESS: Weight range: {weight_stats['min']:.0f} - {weight_stats['max']:.0f}")
        print(f"SUCCESS: Average weight: {weight_stats['mean']:.2f}")
        
        return self.severity_lookup
    
    def combine_datasets(self):
        """Combine all datasets into one comprehensive dataset"""
        print(f"\n4. COMBINING DATASETS")
        print("-"*50)
        
        # Start with main dataset
        self.combined_df = self.dataset.copy()
        print(f"SUCCESS: Starting with main dataset: {len(self.combined_df):,} rows")
        
        # Clean symptom columns
        symptom_columns = [col for col in self.combined_df.columns if col.startswith('Symptom_')]
        
        for col in symptom_columns:
            if col in self.combined_df.columns:
                self.combined_df[col] = self.combined_df[col].astype(str).str.strip()
                self.combined_df[col] = self.combined_df[col].replace(['nan', 'None', ''], np.nan)
        
        print(f"SUCCESS: Cleaned {len(symptom_columns)} symptom columns")
        
        # Add symptom weights
        print("SUCCESS: Adding symptom weights...")
        for col in symptom_columns:
            weight_col = f"{col}_Weight"
            self.combined_df[weight_col] = self.combined_df[col].map(self.severity_lookup)
        
        # Calculate symptom statistics
        print("SUCCESS: Calculating symptom statistics...")
        
        # Count total symptoms per record
        self.combined_df['Total_Symptoms'] = self.combined_df[symptom_columns].notna().sum(axis=1)
        
        # Create concatenated symptom list
        def create_symptom_list(row):
            symptoms = []
            for col in symptom_columns:
                if pd.notna(row[col]) and row[col] != '':
                    symptoms.append(str(row[col]))
            return ', '.join(symptoms) if symptoms else ''
        
        self.combined_df['All_Symptoms'] = self.combined_df.apply(create_symptom_list, axis=1)
        
        # Calculate weight statistics
        weight_columns = [f"{col}_Weight" for col in symptom_columns]
        
        self.combined_df['Average_Symptom_Weight'] = self.combined_df[weight_columns].mean(axis=1, skipna=True)
        self.combined_df['Max_Symptom_Weight'] = self.combined_df[weight_columns].max(axis=1, skipna=True)
        self.combined_df['Min_Symptom_Weight'] = self.combined_df[weight_columns].min(axis=1, skipna=True)
        self.combined_df['Total_Symptom_Weight'] = self.combined_df[weight_columns].sum(axis=1, skipna=True)
        self.combined_df['Weight_Std'] = self.combined_df[weight_columns].std(axis=1, skipna=True)
        
        # Handle inf values
        stat_cols = ['Average_Symptom_Weight', 'Max_Symptom_Weight', 'Min_Symptom_Weight', 'Weight_Std']
        for col in stat_cols:
            self.combined_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Merge with descriptions
        print("SUCCESS: Merging disease descriptions...")
        self.combined_df = self.combined_df.merge(self.descriptions, on='Disease', how='left')
        
        # Merge with precautions
        print("SUCCESS: Merging disease precautions...")
        self.combined_df = self.combined_df.merge(self.precautions, on='Disease', how='left')
        
        # Create concatenated precautions
        def create_precaution_list(row):
            precautions = []
            for i in range(1, 5):
                col = f'Precaution_{i}'
                if col in row and pd.notna(row[col]) and row[col] != '':
                    precautions.append(str(row[col]))
            return ', '.join(precautions) if precautions else ''
        
        self.combined_df['All_Precautions'] = self.combined_df.apply(create_precaution_list, axis=1)
        
        print(f"SUCCESS: Final combined dataset: {len(self.combined_df):,} rows, {len(self.combined_df.columns)} columns")
        
        return self.combined_df
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print(f"\n5. GENERATING SUMMARY STATISTICS")
        print("-"*50)
        
        stats = {}
        
        # Basic statistics
        stats['total_records'] = len(self.combined_df)
        stats['unique_diseases'] = self.combined_df['Disease'].nunique()
        stats['unique_symptoms'] = len(self.severity_lookup)
        
        # Symptom statistics
        stats['avg_symptoms_per_record'] = self.combined_df['Total_Symptoms'].mean()
        stats['min_symptoms_per_record'] = self.combined_df['Total_Symptoms'].min()
        stats['max_symptoms_per_record'] = self.combined_df['Total_Symptoms'].max()
        
        # Weight statistics
        stats['avg_symptom_weight'] = self.combined_df['Average_Symptom_Weight'].mean()
        stats['weight_coverage'] = (self.combined_df['Average_Symptom_Weight'].notna().sum() / len(self.combined_df)) * 100
        
        # Data completeness
        stats['records_with_description'] = self.combined_df['Description'].notna().sum()
        stats['records_with_precautions'] = self.combined_df['Precaution_1'].notna().sum()
        stats['description_coverage'] = (stats['records_with_description'] / stats['total_records']) * 100
        stats['precaution_coverage'] = (stats['records_with_precautions'] / stats['total_records']) * 100
        
        # Disease distribution
        disease_counts = self.combined_df['Disease'].value_counts()
        stats['most_common_disease'] = disease_counts.index[0]
        stats['most_common_disease_count'] = disease_counts.iloc[0]
        stats['least_common_disease'] = disease_counts.index[-1]
        stats['least_common_disease_count'] = disease_counts.iloc[-1]
        
        # Print summary
        print(f"DATASET SUMMARY")
        print(f"  Total Records: {stats['total_records']:,}")
        print(f"  Unique Diseases: {stats['unique_diseases']}")
        print(f"  Unique Symptoms: {stats['unique_symptoms']}")
        print(f"  Avg Symptoms/Record: {stats['avg_symptoms_per_record']:.2f}")
        print(f"  Symptom Range: {stats['min_symptoms_per_record']}-{stats['max_symptoms_per_record']}")
        print(f"  Avg Symptom Weight: {stats['avg_symptom_weight']:.2f}")
        print(f"  Weight Coverage: {stats['weight_coverage']:.1f}%")
        print(f"  Description Coverage: {stats['description_coverage']:.1f}%")
        print(f"  Precaution Coverage: {stats['precaution_coverage']:.1f}%")
        print(f"  Most Common Disease: {stats['most_common_disease']} ({stats['most_common_disease_count']} records)")
        
        return stats
    
    def save_combined_dataset(self, output_filename='combined_disease_dataset.csv'):
        """Save the combined dataset to output directory"""
        print(f"\n6. SAVING COMBINED DATASET")
        print("-"*50)
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"SUCCESS: Output directory created/verified: {self.output_dir}")
            
            # Full output path
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Reorder columns for better readability
            column_order = ['Disease', 'Description', 'Total_Symptoms', 'All_Symptoms']
            
            # Add individual symptom columns
            symptom_cols = [col for col in self.combined_df.columns if col.startswith('Symptom_') and not 'Weight' in col]
            column_order.extend(sorted(symptom_cols))
            
            # Add weight columns
            weight_cols = [col for col in self.combined_df.columns if col.endswith('_Weight')]
            column_order.extend(sorted(weight_cols))
            
            # Add calculated statistics
            stat_cols = ['Average_Symptom_Weight', 'Max_Symptom_Weight', 'Min_Symptom_Weight', 
                        'Total_Symptom_Weight', 'Weight_Std']
            column_order.extend([col for col in stat_cols if col in self.combined_df.columns])
            
            # Add precautions
            prec_cols = ['All_Precautions'] + [f'Precaution_{i}' for i in range(1, 5)]
            column_order.extend([col for col in prec_cols if col in self.combined_df.columns])
            
            # Filter to existing columns and reorder
            final_columns = [col for col in column_order if col in self.combined_df.columns]
            final_df = self.combined_df[final_columns]
            
            # Save to CSV
            final_df.to_csv(output_path, index=False)
            print(f"SUCCESS: Combined dataset saved: {output_path}")
            print(f"   Shape: {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
            print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"ERROR: Error saving dataset: {str(e)}")
            return None
    
    def save_summary_report(self, stats, output_filename='data_summary_report.txt'):
        """Save comprehensive summary report to output directory"""
        print(f"SUCCESS: Saving summary report: {output_filename}")
        
        try:
            # Full output path
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("DISEASE DATASET COMBINATION SUMMARY REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input Directory: {self.input_dir}\n")
                f.write(f"Output Directory: {self.output_dir}\n\n")
                
                f.write("DATASET OVERVIEW\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Records: {stats['total_records']:,}\n")
                f.write(f"Unique Diseases: {stats['unique_diseases']}\n")
                f.write(f"Unique Symptoms: {stats['unique_symptoms']}\n\n")
                
                f.write("SYMPTOM STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Symptoms per Record: {stats['avg_symptoms_per_record']:.2f}\n")
                f.write(f"Symptom Range per Record: {stats['min_symptoms_per_record']}-{stats['max_symptoms_per_record']}\n")
                f.write(f"Average Symptom Weight: {stats['avg_symptom_weight']:.2f}\n")
                f.write(f"Weight Coverage: {stats['weight_coverage']:.1f}%\n\n")
                
                f.write("DATA COMPLETENESS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Records with Descriptions: {stats['records_with_description']:,} ({stats['description_coverage']:.1f}%)\n")
                f.write(f"Records with Precautions: {stats['records_with_precautions']:,} ({stats['precaution_coverage']:.1f}%)\n\n")
                
                f.write("DISEASE DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Most Common Disease: {stats['most_common_disease']} ({stats['most_common_disease_count']} records)\n")
                f.write(f"Least Common Disease: {stats['least_common_disease']} ({stats['least_common_disease_count']} records)\n\n")
                
                # Top 10 diseases
                f.write("TOP 10 DISEASES BY FREQUENCY\n")
                f.write("-" * 30 + "\n")
                disease_counts = self.combined_df['Disease'].value_counts().head(10)
                for i, (disease, count) in enumerate(disease_counts.items(), 1):
                    f.write(f"{i:2d}. {disease}: {count} records\n")
                
                f.write(f"\nFILES CREATED\n")
                f.write("-" * 30 + "\n")
                f.write("- combined_disease_dataset.csv (main output)\n")
                f.write("- data_summary_report.txt (this report)\n")
                
        except Exception as e:
            print(f"ERROR: Error saving report: {str(e)}")
    
    def run_combination_pipeline(self):
        """Run the complete data combination pipeline"""
        print("Starting Disease Dataset Combination Pipeline...\n")
        
        # Step 1: Load raw files
        if not self.load_raw_files():
            return False
        
        # Step 2: Validate data
        if not self.validate_raw_data():
            print("WARNING: Data validation issues found, but continuing...")
        
        # Step 3: Create severity lookup
        self.create_severity_lookup()
        
        # Step 4: Combine datasets
        self.combine_datasets()
        
        # Step 5: Generate statistics
        stats = self.generate_summary_statistics()
        
        # Step 6: Save outputs
        output_file = self.save_combined_dataset()
        self.save_summary_report(stats)
        
        # Final summary
        print(f"\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("OUTPUT FILES:")
        print(f"  SUCCESS: {os.path.join(self.output_dir, 'combined_disease_dataset.csv')} - Main combined dataset")
        print(f"  SUCCESS: {os.path.join(self.output_dir, 'data_summary_report.txt')} - Detailed summary report")
        print(f"\nFINAL STATS:")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique diseases: {stats['unique_diseases']}")
        print(f"  Unique symptoms: {stats['unique_symptoms']}")
        print(f"  Description coverage: {stats['description_coverage']:.1f}%")
        print(f"  Precaution coverage: {stats['precaution_coverage']:.1f}%")
        print(f"\nINPUT SOURCE: {self.input_dir}")
        print(f"OUTPUT LOCATION: {self.output_dir}")
        print(f"\nREADY FOR MACHINE LEARNING PIPELINE!")
        
        return True

def main():
    """Main function"""
    combiner = DiseaseDataCombiner()
    success = combiner.run_combination_pipeline()
    
    if success:
        print(f"\nNEXT STEPS:")
        print(f"1. Review the combined dataset: {os.path.join(combiner.output_dir, 'combined_disease_dataset.csv')}")
        print(f"2. Check the summary report: {os.path.join(combiner.output_dir, 'data_summary_report.txt')}")
        print("3. Run the ML pipeline using the combined dataset")
        print("4. The dataset is now ready for disease prediction models!")
        print(f"\nAll outputs saved to: {combiner.output_dir}")
    else:
        print(f"\nERROR: Pipeline failed. Please check the error messages above.")
        print(f"NOTE: Make sure all CSV files are in the '{combiner.input_dir}' directory")

if __name__ == "__main__":
    main()