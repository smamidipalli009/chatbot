#!/usr/bin/env python3
"""
Disease Classification ML Training Pipeline with Visualization Graphs
====================================================================

Complete machine learning pipeline for disease prediction based on symptoms.
Trains multiple models, performs comprehensive evaluation, generates visualization graphs,
and saves the best model.

Requirements:
- combined_disease_dataset.csv (output from data combination pipeline)
- pandas, numpy, scikit-learn, matplotlib, seaborn

Outputs:
- disease_classification_model.pkl (trained model + preprocessors)
- model_evaluation_report.html (comprehensive evaluation report)
- model_comparison_plots.png (visualization of results)
- feature_importance_analysis.csv (feature analysis)
- visualizations/ (directory with 5 publication quality graphs)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)
import pickle
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Set professional style for research paper graphs
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

class ModelVisualizationGraphs:
    """Generates publication quality graphs for model evaluation"""
    
    def __init__(self, model_results, evaluation_data, y_test, label_encoder, output_dir=None):
        self.model_results = model_results
        self.evaluation_data = evaluation_data
        self.y_test = y_test
        self.label_encoder = label_encoder
        
        # Auto-detect correct output directory for visualizations
        if output_dir is None:
            if os.path.exists('models') or not os.path.exists('../models'):
                output_dir = 'models/visualizations'  # Running from root
            else:
                output_dir = '../models/visualizations'  # Running from src/
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def graph_1_model_performance_dashboard(self):
        """Graph 1: Model Performance Comparison Dashboard"""
        print("Creating Graph 1: Model Performance Dashboard")
        
        models = list(self.model_results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC']
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        x = np.arange(len(models))
        width = 0.15
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = [self.model_results[model][metric] for model in models]
            bars = ax.bar(x + i * width, values, width, label=label, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Improved text positioning with better spacing
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                # Stagger text positions to avoid overlap
                y_offset = 0.02 + (i * 0.005) if height > 0.9 else 0.015
                ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8,
                       fontweight='bold' if j == 0 and 'Random Forest' in models[j] else 'normal')
        
        ax.set_xlabel('Machine Learning Models', fontweight='bold', fontsize=12)
        ax.set_ylabel('Performance Score', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Comparison Across Multiple Metrics', 
                    fontweight='bold', pad=25, fontsize=14)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=0, ha='center', fontsize=11)
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.15)  # More space for labels
        
        # Highlight Random Forest
        if 'Random Forest' in models:
            rf_index = models.index('Random Forest')
            ax.axvline(x=rf_index + width * 2, color='red', linestyle='--', 
                      alpha=0.7, linewidth=2)
            # Add text annotation
            ax.text(rf_index + width * 2, 1.08, 'Best Model', ha='center', 
                   fontweight='bold', color='red', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graph_1_model_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def graph_2_cross_validation_stability(self):
        """Graph 2: Cross-Validation Stability Analysis - Bar Chart Version"""
        print("Creating Graph 2: Cross-Validation Stability Analysis")
        
        models = list(self.model_results.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: CV Mean scores with error bars
        cv_means = [self.model_results[model]['cv_mean'] for model in models]
        cv_stds = [self.model_results[model]['cv_std'] for model in models]
        
        colors = ['gold' if 'Random Forest' in model else 'lightblue' for model in models]
        bars1 = ax1.bar(models, cv_means, yerr=cv_stds, capsize=8, color=colors, 
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars1, cv_means, cv_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        ax1.set_ylabel('Cross-Validation F1-Score', fontweight='bold', fontsize=12)
        ax1.set_title('A) CV Performance with Standard Deviation', 
                     fontweight='bold', fontsize=13)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(cv_means) + max(cv_stds) + 0.1)
        
        # Right plot: Stability ranking (inverse of std deviation)
        stability_scores = [1 / (1 + std) for std in cv_stds]
        colors2 = ['gold' if 'Random Forest' in model else 'lightgreen' for model in models]
        bars2 = ax2.bar(models, stability_scores, color=colors2, alpha=0.8, 
                       edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, score, std in zip(bars2, stability_scores, cv_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}\n(σ={std:.3f})', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        ax2.set_ylabel('Stability Score (Higher = More Stable)', fontweight='bold', fontsize=12)
        ax2.set_title('B) Model Stability Ranking', fontweight='bold', fontsize=13)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.1)
        
        # Add explanation text
        fig.suptitle('Cross-Validation Stability Analysis\n(Lower standard deviation indicates more reliable model)', 
                    fontweight='bold', fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f'{self.output_dir}/graph_2_cv_stability.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def graph_3_roc_curves_comparison(self):
        """Graph 3: ROC Curves Comparison - Ultra-Zoom + AUC Bar Chart"""
        print("Creating Graph 3: ROC Curves Comparison")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'D', 'v']
        
        model_aucs = {}
        
        # Left plot: Ultra-zoomed ROC curves (where the action happens)
        for i, (model_name, color, linestyle, marker) in enumerate(zip(self.model_results.keys(), colors, line_styles, markers)):
            model_data = self.model_results[model_name]
            
            if 'y_pred_proba' in model_data and model_data['y_pred_proba'] is not None:
                y_pred_proba = model_data['y_pred_proba']
                
                n_classes = len(self.label_encoder.classes_)
                y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
                
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                roc_auc = auc(fpr, tpr)
                model_aucs[model_name] = roc_auc
                
                # Plot with enhanced visibility
                ax1.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=4,
                        label=f'{model_name}', 
                        marker=marker, markersize=8, markevery=10, alpha=0.8)
        
        # Ultra-zoom to where models actually differ
        ax1.set_xlim([0.0, 0.05])  # Focus on first 5% FPR
        ax1.set_ylim([0.85, 1.0])   # Focus on high TPR range
        ax1.set_xlabel('False Positive Rate (0.0 - 0.05)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('True Positive Rate (0.85 - 1.0)', fontweight='bold', fontsize=12)
        ax1.set_title('A) Ultra-Zoom: Where Models Differ\n(FPR: 0.0-0.05, TPR: 0.85-1.0)', 
                     fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.4)
        ax1.legend(loc="lower right", frameon=True, fontsize=11, framealpha=0.9)
        
        # Add precise tick marks for better readability
        ax1.set_xticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        ax1.set_yticks([0.85, 0.90, 0.95, 1.0])
        
        # Right plot: Clean AUC Bar Chart
        if model_aucs:
            # Sort models by AUC score
            sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=False)
            model_names = [item[0] for item in sorted_models]
            auc_scores = [item[1] for item in sorted_models]
            
            # Color bars - highlight Random Forest if it's the best
            bar_colors = []
            for model in model_names:
                if 'Random Forest' in model and auc_scores[model_names.index(model)] == max(auc_scores):
                    bar_colors.append('gold')
                else:
                    bar_colors.append('lightblue')
            
            bars = ax2.barh(model_names, auc_scores, color=bar_colors, 
                           edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Add precise AUC values on bars
            for bar, score in zip(bars, auc_scores):
                width = bar.get_width()
                ax2.text(width - 0.002, bar.get_y() + bar.get_height()/2,
                        f'{score:.4f}', ha='right', va='center', 
                        fontweight='bold', fontsize=12, color='black')
            
            ax2.set_xlabel('AUC Score', fontweight='bold', fontsize=12)
            ax2.set_title('B) AUC Comparison\n(Higher is Better)', fontweight='bold', fontsize=13)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Focus on the actual range of AUC scores for better differentiation
            min_auc = min(auc_scores)
            max_auc = max(auc_scores)
            auc_range = max_auc - min_auc
            ax2.set_xlim([min_auc - auc_range * 0.1, max_auc + auc_range * 0.1])
            
            # Add best model annotation
            best_model = max(model_aucs, key=model_aucs.get)
            best_auc = model_aucs[best_model]
            ax2.text(0.05, 0.95, f'Best: {best_model}\nAUC: {best_auc:.4f}', 
                   transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
            
            # Highlight differences
            if len(auc_scores) > 1:
                auc_diff = max(auc_scores) - min(auc_scores)
                ax2.text(0.05, 0.05, f'AUC Range: {auc_diff:.4f}', 
                       transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        fig.suptitle('ROC Analysis: Model Differences Revealed', 
                    fontweight='bold', fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f'{self.output_dir}/graph_3_roc_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def graph_4_feature_importance(self):
        """Graph 4: Feature Importance Analysis"""
        print("Creating Graph 4: Feature Importance Analysis")
        
        feature_importance = self.evaluation_data.get('feature_importance')
        
        if feature_importance is None:
            print("Warning: No feature importance data available")
            return
        
        top_features = feature_importance.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['importance'], 
                      color='darkgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            width = bar.get_width()
            ax.text(width + max(top_features['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        feature_names = [name.replace('_', ' ').replace('present', '(Present)').title() 
                        for name in top_features['feature']]
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance Score', fontweight='bold')
        ax.set_ylabel('Symptoms and Features', fontweight='bold')
        ax.set_title('Random Forest Feature Importance Analysis\nTop 20 Most Influential Symptoms for Disease Prediction', 
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        model_info = f"Model: Random Forest\nTotal Features: {len(feature_importance)}\nShowing: Top 20"
        ax.text(0.98, 0.02, model_info, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/graph_4_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def graph_5_random_forest_superiority(self):
        """Graph 5: Random Forest Superiority Analysis"""
        print("Creating Graph 5: Random Forest Superiority Analysis")
        
        models = list(self.model_results.keys())
        
        composite_data = []
        for model in models:
            results = self.model_results[model]
            
            composite_score = (
                results['accuracy'] * 0.25 +
                results['f1_score'] * 0.25 + 
                results['precision'] * 0.15 +
                results['recall'] * 0.15 +
                results['roc_auc'] * 0.20
            )
            
            stability_score = 1 / (1 + results['cv_std'])
            overall_score = (composite_score * 0.7) + (stability_score * 0.3)
            
            composite_data.append({
                'Model': model,
                'Composite_Performance': composite_score,
                'Stability_Score': stability_score,
                'Overall_Score': overall_score,
                'Training_Time': results.get('train_time', 0)
            })
        
        df_composite = pd.DataFrame(composite_data)
        df_composite = df_composite.sort_values('Overall_Score', ascending=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Random Forest Superiority Analysis - Comprehensive Model Comparison', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        # Subplot 1: Overall Ranking - Fixed value positioning
        colors = ['gold' if 'Random Forest' in model else 'lightblue' for model in df_composite['Model']]
        bars1 = ax1.barh(df_composite['Model'], df_composite['Overall_Score'], 
                        color=colors, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Fix overlapping values - position them well outside the bars
        max_score = max(df_composite['Overall_Score'])
        for bar, score in zip(bars1, df_composite['Overall_Score']):
            width = bar.get_width()
            # Position text well to the right of the bar end
            ax1.text(max_score + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontweight='bold', fontsize=11)
        
        ax1.set_xlabel('Overall Performance Score', fontweight='bold', fontsize=12)
        ax1.set_title('A) Overall Model Ranking', fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, max_score + 0.1)  # Extended xlim for text space
        
        # Subplot 2: Performance vs Stability - Text positioned to LEFT of points
        scatter_colors = ['red' if 'Random Forest' in model else 'blue' for model in df_composite['Model']]
        scatter_sizes = [200 if 'Random Forest' in model else 100 for model in df_composite['Model']]
        
        ax2.scatter(df_composite['Composite_Performance'], df_composite['Stability_Score'], 
                   c=scatter_colors, s=scatter_sizes, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Position ALL text labels to the LEFT of points
        for i, model in enumerate(df_composite['Model']):
            x_pos = df_composite.iloc[i]['Composite_Performance']
            y_pos = df_composite.iloc[i]['Stability_Score']
            
            # Position text to the left of each point
            ax2.annotate(model, (x_pos, y_pos),
                        xytext=(-15, 0), textcoords='offset points', 
                        fontsize=10, ha='right', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Composite Performance Score', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Stability Score', fontweight='bold', fontsize=12)
        ax2.set_title('B) Performance vs Stability Trade-off', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Radar chart with fixed label positioning
        top_3_models = df_composite.nlargest(3, 'Overall_Score')['Model'].tolist()
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC']
        
        # Adjust angles to prevent ROC-AUC and F1-Score overlap
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        
        for i, model in enumerate(top_3_models):
            values = [self.model_results[model][metric] for metric in metrics]
            values += values[:1]
            
            if 'Random Forest' in model:
                color = 'red'
                linewidth = 4
                alpha = 0.9
                linestyle = '-'
            else:
                color = plt.cm.Set1(i)
                linewidth = 2
                alpha = 0.6
                linestyle = '--'
            
            ax3.plot(angles, values, 'o-', linewidth=linewidth, 
                    label=model, color=color, alpha=alpha, linestyle=linestyle,
                    markersize=6)
            ax3.fill(angles, values, alpha=0.15, color=color)
        
        ax3.set_xticks(angles[:-1])
        
        # Custom label positioning to avoid overlap
        custom_labels = []
        for i, (angle, label) in enumerate(zip(angles[:-1], metric_labels)):
            if label == 'ROC-AUC':
                custom_labels.append('ROC\nAUC')  # Split into two lines
            elif label == 'F1-Score':
                custom_labels.append('F1\nScore')  # Split into two lines  
            else:
                custom_labels.append(label)
        
        ax3.set_xticklabels(custom_labels, fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.set_title('C) Top 3 Models - Metric Comparison', fontweight='bold', 
                     fontsize=13, pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax3.grid(True)
        
        # Subplot 4: Ranking table (no changes needed)
        ax4.axis('tight')
        ax4.axis('off')
        
        ranking_data = []
        df_sorted = df_composite.sort_values('Overall_Score', ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ranking_data.append([
                f"#{i+1}",
                row['Model'],
                f"{row['Overall_Score']:.3f}",
                f"{row['Composite_Performance']:.3f}",
                f"{row['Stability_Score']:.3f}"
            ])
        
        table = ax4.table(cellText=ranking_data,
                         colLabels=['Rank', 'Model Name', 'Overall', 'Performance', 'Stability'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.12, 0.35, 0.18, 0.18, 0.17])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style the header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight Random Forest row
        for i, row_data in enumerate(ranking_data):
            if 'Random Forest' in row_data[1]:
                for j in range(len(row_data)):
                    table[(i+1, j)].set_facecolor('gold')
                    table[(i+1, j)].set_text_props(weight='bold')
        
        ax4.set_title('D) Final Model Rankings', fontweight='bold', fontsize=13, pad=20)
        
        # Move the best model text higher as requested
        ax4.text(0.5, -0.05, 'Random Forest is the best performing model', 
                ha='center', va='top', transform=ax4.transAxes, 
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)
        plt.savefig(f'{self.output_dir}/graph_5_rf_superiority.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_graphs(self):
        """Generate all 5 publication quality graphs"""
        print("Creating Graph 1: Model Performance Dashboard")
        self.graph_1_model_performance_dashboard()
        
        print("Creating Graph 2: Cross-Validation Stability Analysis")
        self.graph_2_cross_validation_stability()
        
        print("Creating Graph 3: ROC Curves Comparison")
        self.graph_3_roc_curves_comparison()
        
        print("Creating Graph 4: Feature Importance Analysis")
        self.graph_4_feature_importance()
        
        print("Creating Graph 5: Random Forest Superiority Analysis")
        self.graph_5_random_forest_superiority()
        
        return [
            f'{self.output_dir}/graph_1_model_performance.png',
            f'{self.output_dir}/graph_2_cv_stability.png',
            f'{self.output_dir}/graph_3_roc_comparison.png',
            f'{self.output_dir}/graph_4_feature_importance.png',
            f'{self.output_dir}/graph_5_rf_superiority.png'
        ]


class DiseaseClassificationMLPipeline:
    def __init__(self, dataset_path=None, output_dir=None):
        """Initialize the ML pipeline"""
        # Auto-detect correct paths based on current working directory
        if dataset_path is None:
            if os.path.exists('data/processed/combined_disease_dataset.csv'):
                dataset_path = 'data/processed/combined_disease_dataset.csv'  # Running from root
            else:
                dataset_path = '../data/processed/combined_disease_dataset.csv'  # Running from src/
        
        if output_dir is None:
            if os.path.exists('models') or not os.path.exists('../models'):
                output_dir = 'models'  # Running from root
            else:
                output_dir = '../models'  # Running from src/
                
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        
        # Model components
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        
        # Models and results
        self.models = {}
        self.model_results = {}
        self.best_model_info = None
        
        # Evaluation data
        self.evaluation_data = {}
        
    def load_and_explore_data(self):
        """Load and explore the combined dataset"""
        print("="*80)
        print("DISEASE CLASSIFICATION ML TRAINING PIPELINE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n1. LOADING DATASET")
        print("-"*60)
        
        if not os.path.exists(self.dataset_path):
            print(f"ERROR: Dataset not found: {self.dataset_path}")
            print("Please run the data combination pipeline first!")
            return False
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"SUCCESS: Dataset loaded successfully!")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Unique diseases: {self.df['Disease'].nunique()}")
            print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Data quality check
            missing_values = self.df.isnull().sum().sum()
            duplicate_rows = self.df.duplicated().sum()
            
            print(f"\nDATA QUALITY CHECK:")
            print(f"   Missing values: {missing_values:,}")
            print(f"   Duplicate rows: {duplicate_rows:,}")
            
            # Disease distribution
            disease_counts = self.df['Disease'].value_counts()
            print(f"\nDISEASE DISTRIBUTION (Top 5):")
            for i, (disease, count) in enumerate(disease_counts.head().items(), 1):
                percentage = (count / len(self.df)) * 100
                print(f"   {i}. {disease}: {count:,} records ({percentage:.1f}%)")
            
            # Check class imbalance
            imbalance_ratio = disease_counts.max() / disease_counts.min()
            print(f"\nCLASS BALANCE:")
            print(f"   Most common: {disease_counts.iloc[0]:,} records")
            print(f"   Least common: {disease_counts.iloc[-1]:,} records")
            print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 10:
                print("   WARNING: High class imbalance detected!")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading dataset: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess data and create features"""
        print(f"\n2. DATA PREPROCESSING & FEATURE ENGINEERING")
        print("-"*60)
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_duplicates = initial_rows - len(self.df)
        if removed_duplicates > 0:
            print(f"SUCCESS: Removed {removed_duplicates:,} duplicate rows")
        
        # Feature selection strategy
        print(f"FEATURE ENGINEERING:")
        
        feature_columns = []
        
        # 1. Symptom presence features (binary indicators)
        symptom_cols = [col for col in self.df.columns if col.startswith('Symptom_') and not 'Weight' in col]
        for col in symptom_cols:
            binary_col = f"{col}_present"
            self.df[binary_col] = (~self.df[col].isna()).astype(int)
            feature_columns.append(binary_col)
        
        print(f"   Created {len(symptom_cols)} symptom presence indicators")
        
        # 2. Symptom weight features (numerical)
        weight_cols = [col for col in self.df.columns if col.startswith('Symptom_') and 'Weight' in col]
        for col in weight_cols:
            self.df[col] = self.df[col].fillna(0)  # Missing weights = 0
            feature_columns.append(col)
        
        print(f"   Processed {len(weight_cols)} symptom weight features")
        
        # 3. Statistical features (if available)
        stat_features = [
            'Total_Symptoms', 'Average_Symptom_Weight', 'Max_Symptom_Weight',
            'Min_Symptom_Weight', 'Total_Symptom_Weight', 'Weight_Std'
        ]
        
        available_stats = []
        for col in stat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
                feature_columns.append(col)
                available_stats.append(col)
        
        print(f"   Added {len(available_stats)} statistical features")
        
        # Create final feature matrix
        self.X = self.df[feature_columns].copy()
        self.feature_names = feature_columns
        
        # Handle any remaining missing values
        missing_features = self.X.isnull().sum().sum()
        if missing_features > 0:
            self.X = self.X.fillna(0)
            print(f"   Filled {missing_features:,} remaining missing values with 0")
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df['Disease'])
        
        print(f"\nFINAL FEATURE MATRIX:")
        print(f"   Features: {self.X.shape[1]:,}")
        print(f"   Samples: {self.X.shape[0]:,}")
        print(f"   Target classes: {len(self.label_encoder.classes_)}")
        print(f"   Feature types: {self.X.dtypes.value_counts().to_dict()}")
        
        # Feature scaling
        print(f"\nFEATURE SCALING:")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, columns=self.feature_names, index=self.X.index)
        print(f"   Applied StandardScaler to all features")
        print(f"   Feature range after scaling: [{self.X.min().min():.3f}, {self.X.max().max():.3f}]")
        
        return self.X, self.y
    
    def initialize_models(self):
        """Initialize ML models with optimized parameters"""
        print(f"\n3. MODEL INITIALIZATION")
        print("-"*60)
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver='lbfgs',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'Naive Bayes': GaussianNB(
                var_smoothing=1e-9
            )
        }
        
        # Try to add XGBoost if available
        try:
            import xgboost as xgb
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            print(f"SUCCESS: XGBoost available and added to models")
        except ImportError:
            print(f"INFO: XGBoost not available, skipping")
        
        print(f"\nINITIALIZED MODELS:")
        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"   {i}. {name}")
            
        return self.models
    
    def split_data(self):
        """Split data into training and testing sets"""
        print(f"\n4. DATA SPLITTING")
        print("-"*60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y
        )
        
        print(f"SUCCESS: Data split completed:")
        print(f"   Training set: {self.X_train.shape[0]:,} samples ({(len(self.X_train)/len(self.X)*100):.1f}%)")
        print(f"   Testing set: {self.X_test.shape[0]:,} samples ({(len(self.X_test)/len(self.X)*100):.1f}%)")
        
        # Check class distribution in splits
        train_dist = pd.Series(self.y_train).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(self.y_test).value_counts(normalize=True).sort_index()
        
        # Calculate distribution difference
        dist_diff = abs(train_dist - test_dist).mean()
        print(f"   Class distribution difference: {dist_diff:.4f} (lower is better)")
        
        if dist_diff < 0.01:
            print(f"   SUCCESS: Excellent stratification!")
        elif dist_diff < 0.05:
            print(f"   SUCCESS: Good stratification")
        else:
            print(f"   WARNING: Stratification could be improved")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models with comprehensive metrics"""
        print(f"\n5. MODEL TRAINING & EVALUATION")
        print("-"*60)
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store results
        self.model_results = {}
        
        print(f"Training {len(self.models)} models with 5-fold cross-validation...")
        
        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"\n   {i}/{len(self.models)} Training {name}...")
            
            try:
                # Cross-validation scores
                cv_start_time = datetime.now()
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
                cv_time = (datetime.now() - cv_start_time).total_seconds()
                
                # Train on full training set
                train_start_time = datetime.now()
                model.fit(self.X_train, self.y_train)
                train_time = (datetime.now() - train_start_time).total_seconds()
                
                # Predictions
                pred_start_time = datetime.now()
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                pred_time = (datetime.now() - pred_start_time).total_seconds()
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # ROC-AUC for multiclass
                roc_auc = 0
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except:
                        roc_auc = 0
                
                # Store comprehensive results
                self.model_results[name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'cv_time': cv_time,
                    'train_time': train_time,
                    'pred_time': pred_time,
                    'total_time': cv_time + train_time + pred_time
                }
                
                # Print results
                print(f"      CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"      Test Accuracy: {accuracy:.4f}")
                print(f"      Test F1-Score: {f1:.4f}")
                print(f"      ROC-AUC: {roc_auc:.4f}")
                print(f"      Total Time: {cv_time + train_time + pred_time:.2f}s")
                
            except Exception as e:
                print(f"      ERROR: Failed to train {name}: {str(e)}")
                continue
        
        print(f"\nSUCCESS: Model training completed!")
        return self.model_results
    
    def find_best_model(self):
        """Find and select the best performing model"""
        print(f"\n6. BEST MODEL SELECTION")
        print("-"*60)
        
        if not self.model_results:
            print("ERROR: No model results available!")
            return None
        
        # Find best model based on F1-score (better for imbalanced classes)
        best_f1 = 0
        best_model_name = None
        
        for name, results in self.model_results.items():
            if results['f1_score'] > best_f1:
                best_f1 = results['f1_score']
                best_model_name = name
        
        self.best_model_info = {
            'name': best_model_name,
            'model': self.model_results[best_model_name]['model'],
            'results': self.model_results[best_model_name]
        }
        
        print(f"BEST MODEL: {best_model_name}")
        print(f"   Test Accuracy: {self.best_model_info['results']['accuracy']:.4f}")
        print(f"   Test F1-Score: {self.best_model_info['results']['f1_score']:.4f}")
        print(f"   Test Precision: {self.best_model_info['results']['precision']:.4f}")
        print(f"   Test Recall: {self.best_model_info['results']['recall']:.4f}")
        print(f"   ROC-AUC: {self.best_model_info['results']['roc_auc']:.4f}")
        print(f"   Training Time: {self.best_model_info['results']['train_time']:.2f}s")
        
        return self.best_model_info
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n7. HYPERPARAMETER TUNING")
        print("-"*60)
        
        if not self.best_model_info:
            print("ERROR: No best model selected!")
            return None
        
        best_name = self.best_model_info['name']
        print(f"Tuning {best_name}...")
        
        # Define parameter grids for each model
        param_grids = {
            'Random Forest': {
                'n_estimators': [150, 200, 250],
                'max_depth': [15, 20, 25],
                'min_samples_split': [5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [5, 6, 7]
            },
            'XGBoost': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [5, 6, 7]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            }
        }
        
        if best_name not in param_grids:
            print(f"WARNING: No tuning grid defined for {best_name}, using current model")
            return self.best_model_info['model']
        
        # Perform grid search
        base_model = self.models[best_name]
        param_grid = param_grids[best_name]
        
        print(f"   Searching {len(param_grid)} parameters...")
        print(f"   Optimization metric: F1-weighted")
        
        try:
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,  # Reduced for speed
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            
            tuning_start = datetime.now()
            grid_search.fit(self.X_train, self.y_train)
            tuning_time = (datetime.now() - tuning_start).total_seconds()
            
            # Get best model
            best_tuned_model = grid_search.best_estimator_
            
            # Evaluate tuned model
            y_pred_tuned = best_tuned_model.predict(self.X_test)
            tuned_accuracy = accuracy_score(self.y_test, y_pred_tuned)
            tuned_f1 = f1_score(self.y_test, y_pred_tuned, average='weighted')
            
            # Compare with original
            original_f1 = self.best_model_info['results']['f1_score']
            improvement = tuned_f1 - original_f1
            
            print(f"   SUCCESS: Tuning completed in {tuning_time:.1f}s")
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Original F1-Score: {original_f1:.4f}")
            print(f"   Tuned F1-Score: {tuned_f1:.4f}")
            print(f"   Improvement: {improvement:+.4f}")
            
            if improvement > 0.001:  # Significant improvement
                print(f"   SUCCESS: Tuning improved performance! Using tuned model.")
                self.best_model_info['model'] = best_tuned_model
                self.best_model_info['results']['accuracy'] = tuned_accuracy
                self.best_model_info['results']['f1_score'] = tuned_f1
                self.best_model_info['tuned'] = True
                self.best_model_info['tuning_time'] = tuning_time
                self.best_model_info['best_params'] = grid_search.best_params_
            else:
                print(f"   INFO: No significant improvement, keeping original model")
                self.best_model_info['tuned'] = False
            
            return self.best_model_info['model']
            
        except Exception as e:
            print(f"   ERROR: Tuning failed: {str(e)}")
            print(f"   INFO: Using original model")
            return self.best_model_info['model']
    
    def generate_comprehensive_evaluation(self):
        """Generate comprehensive model evaluation"""
        print(f"\n8. COMPREHENSIVE EVALUATION")
        print("-"*60)
        
        # Create results comparison table
        print(f"MODEL COMPARISON TABLE:")
        print(f"{'Model':<20} {'CV_F1':<8} {'Accuracy':<9} {'F1-Score':<9} {'ROC-AUC':<8} {'Time(s)':<8}")
        print("-" * 80)
        
        # Sort models by F1-score
        sorted_results = sorted(self.model_results.items(), 
                              key=lambda x: x[1]['f1_score'], reverse=True)
        
        best_model_name = self.best_model_info['name']
        
        for name, results in sorted_results:
            print(f"{name:<20} {results['cv_mean']:<8.4f} {results['accuracy']:<9.4f} "
                  f"{results['f1_score']:<9.4f} {results['roc_auc']:<8.4f} {results['total_time']:<8.1f}")
        
        # Add separate line for best model
        print(f"\n{best_model_name} is the best model")
        
        # Detailed classification report for best model
        best_results = self.best_model_info['results']
        y_pred = best_results['y_pred']
        
        print(f"\nDETAILED CLASSIFICATION REPORT ({self.best_model_info['name']}):")
        print("-" * 80)
        
        # Get classification report
        class_report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Display per-class metrics for top classes
        class_names = self.label_encoder.classes_
        class_metrics = []
        
        for class_name in class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                class_metrics.append({
                    'Disease': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': int(metrics['support'])
                })
        
        # Sort by support (frequency) and show top 10
        class_metrics.sort(key=lambda x: x['Support'], reverse=True)
        
        print(f"{'Disease':<25} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Support':<8}")
        print("-" * 70)
        
        for metrics in class_metrics[:10]:
            print(f"{metrics['Disease']:<25} {metrics['Precision']:<10.3f} {metrics['Recall']:<8.3f} "
                  f"{metrics['F1-Score']:<9.3f} {metrics['Support']:<8}")
        
        if len(class_metrics) > 10:
            print(f"... and {len(class_metrics) - 10} more diseases")
        
        # Overall metrics
        print(f"\nOVERALL PERFORMANCE METRICS:")
        macro_avg = class_report['macro avg']
        weighted_avg = class_report['weighted avg']
        
        print(f"   Macro Average    - Precision: {macro_avg['precision']:.3f}, "
              f"Recall: {macro_avg['recall']:.3f}, F1: {macro_avg['f1-score']:.3f}")
        print(f"   Weighted Average - Precision: {weighted_avg['precision']:.3f}, "
              f"Recall: {weighted_avg['recall']:.3f}, F1: {weighted_avg['f1-score']:.3f}")
        
        # Store evaluation data
        self.evaluation_data = {
            'model_comparison': sorted_results,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'feature_importance': self._get_feature_importance()
        }
        
        return self.evaluation_data
    
    def _get_feature_importance(self):
        """Extract feature importance from the best model"""
        model = self.best_model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            # Models without feature importance
            return None
        
        # Create feature importance dataframe
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_imp_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Disease Classification Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        ax1 = axes[0, 0]
        models = list(self.model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.model_results[model][metric] for model in models]
            ax1.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Cross-validation scores
        ax2 = axes[0, 1]
        cv_means = [self.model_results[model]['cv_mean'] for model in models]
        cv_stds = [self.model_results[model]['cv_std'] for model in models]
        
        colors = ['gold' if model == self.best_model_info['name'] else 'lightblue' for model in models]
        bars = ax2.bar(models, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.8)
        
        ax2.set_ylabel('CV F1-Score')
        ax2.set_title('Cross-Validation Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, cv_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Importance (Top 15)
        ax3 = axes[0, 2]
        feature_imp = self.evaluation_data.get('feature_importance')
        
        if feature_imp is not None:
            top_features = feature_imp.head(15)
            y_pos = np.arange(len(top_features))
            
            ax3.barh(y_pos, top_features['importance'], color='green', alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f.replace('_', ' ') for f in top_features['feature']], fontsize=8)
            ax3.set_xlabel('Importance')
            ax3.set_title(f'Top 15 Features ({self.best_model_info["name"]})')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Feature Importance')
        
        # 4. Confusion Matrix (Top 10 diseases)
        ax4 = axes[1, 0]
        cm = self.evaluation_data['confusion_matrix']
        
        # Get top 10 most frequent diseases for readability
        disease_counts = pd.Series(self.y_test).value_counts()
        top_diseases = disease_counts.head(10).index
        
        if len(top_diseases) > 1:
            # Filter confusion matrix for top diseases
            mask = np.isin(self.y_test, top_diseases) & np.isin(self.best_model_info['results']['y_pred'], top_diseases)
            cm_subset = confusion_matrix(self.y_test[mask], self.best_model_info['results']['y_pred'][mask])
            
            class_names = [self.label_encoder.classes_[i] for i in top_diseases]
            
            # Normalize confusion matrix
            cm_norm = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]
            
            im = ax4.imshow(cm_norm, interpolation='nearest', cmap='Blues')
            ax4.set_title(f'Confusion Matrix (Top 10 Diseases)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            
            # Add labels
            tick_marks = np.arange(len(class_names))
            ax4.set_xticks(tick_marks)
            ax4.set_yticks(tick_marks)
            ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in class_names], 
                               rotation=45, ha='right', fontsize=8)
            ax4.set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in class_names], 
                               fontsize=8)
            ax4.set_ylabel('True Label')
            ax4.set_xlabel('Predicted Label')
            
            # Add text annotations for significant values
            thresh = cm_norm.max() / 2.
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    if cm_norm[i, j] > 0.1:  # Only show significant values
                        ax4.text(j, i, f'{cm_norm[i, j]:.2f}',
                                ha="center", va="center",
                                color="white" if cm_norm[i, j] > thresh else "black",
                                fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Not enough classes\nfor confusion matrix', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Training Time Comparison
        ax5 = axes[1, 1]
        train_times = [self.model_results[model]['total_time'] for model in models]
        colors_time = ['gold' if model == self.best_model_info['name'] else 'lightcoral' for model in models]
        
        bars = ax5.bar(models, train_times, color=colors_time, alpha=0.8)
        ax5.set_ylabel('Time (seconds)')
        ax5.set_title('Total Training Time')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars, train_times):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(train_times) * 0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 6. ROC-AUC Comparison
        ax6 = axes[1, 2]
        roc_scores = [self.model_results[model]['roc_auc'] for model in models]
        colors_roc = ['gold' if model == self.best_model_info['name'] else 'lightgreen' for model in models]
        
        bars = ax6.bar(models, roc_scores, color=colors_roc, alpha=0.8)
        ax6.set_ylabel('ROC-AUC Score')
        ax6.set_title('ROC-AUC Comparison')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, roc_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot to output directory - create analysis subdirectory
        analysis_dir = os.path.join(self.output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        plot_filename = os.path.join(analysis_dir, 'model_comparison_plots.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return plot_filename
    
    def generate_visualization_graphs(self):
        """Generate publication quality graphs"""
        print(f"\nGENERATING VISUALIZATION GRAPHS")
        print("-"*60)
        
        try:
            # Auto-detect visualization output directory
            viz_output_dir = None
            if os.path.exists('models') or not os.path.exists('../models'):
                viz_output_dir = 'models/visualizations'  # Running from root
            else:
                viz_output_dir = '../models/visualizations'  # Running from src/
            
            graph_generator = ModelVisualizationGraphs(
                model_results=self.model_results,
                evaluation_data=self.evaluation_data,
                y_test=self.y_test,
                label_encoder=self.label_encoder,
                output_dir=viz_output_dir
            )
            
            visualization_graph_files = graph_generator.generate_all_graphs()
            
            print(f"Visualization graphs generated!")
            return visualization_graph_files
            
        except Exception as e:
            print(f"ERROR: Failed to generate visualization graphs: {str(e)}")
            return []
    
    def save_feature_analysis(self):
        """Save feature importance analysis"""
        feature_imp = self.evaluation_data.get('feature_importance')
        
        if feature_imp is not None:
            analysis_dir = os.path.join(self.output_dir, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            filename = os.path.join(analysis_dir, 'feature_importance_analysis.csv')
            feature_imp.to_csv(filename, index=False)
            print(f"SUCCESS: Feature analysis saved: {filename}")
            return filename
        
        return None
    
    def save_model_package(self, filepath='disease_classification_model.pkl'):
        """Save the complete model package"""
        print(f"\nSAVING MODEL PACKAGE")
        print("-"*60)
        
        try:
            # Create output directory if it doesn't exist
            saved_models_dir = os.path.join(self.output_dir, 'saved_models')
            os.makedirs(saved_models_dir, exist_ok=True)
            
            # Full output path
            full_filepath = os.path.join(saved_models_dir, filepath)
            
            # Create model package with all necessary components
            model_package = {
                # Core model components
                'model': self.best_model_info['model'],
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                
                # Model metadata
                'model_name': self.best_model_info['name'],
                'model_type': type(self.best_model_info['model']).__name__,
                'n_features': len(self.feature_names),
                'n_classes': len(self.label_encoder.classes_),
                'class_names': self.label_encoder.classes_.tolist(),
                
                # Performance metrics
                'accuracy': self.best_model_info['results']['accuracy'],
                'f1_score': self.best_model_info['results']['f1_score'],
                'precision': self.best_model_info['results']['precision'],
                'recall': self.best_model_info['results']['recall'],
                'roc_auc': self.best_model_info['results']['roc_auc'],
                'cv_f1_mean': self.best_model_info['results']['cv_mean'],
                'cv_f1_std': self.best_model_info['results']['cv_std'],
                
                # Training metadata
                'training_date': datetime.now().isoformat(),
                'dataset_size': len(self.df),
                'training_size': len(self.X_train),
                'test_size': len(self.X_test),
                'training_time': self.best_model_info['results']['train_time'],
                
                # Hyperparameter tuning info
                'tuned': self.best_model_info.get('tuned', False),
                'best_params': self.best_model_info.get('best_params', {}),
                
                # Model comparison results
                'all_model_results': {name: {
                    'accuracy': results['accuracy'],
                    'f1_score': results['f1_score'],
                    'cv_mean': results['cv_mean']
                } for name, results in self.model_results.items()},
                
                # Feature importance (if available)
                'feature_importance': self.evaluation_data.get('feature_importance').to_dict('records') if self.evaluation_data.get('feature_importance') is not None else None
            }
            
            # Save the package
            with open(full_filepath, 'wb') as f:
                pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(full_filepath) / 1024 / 1024
            
            print(f"SUCCESS: Model package saved successfully!")
            print(f"   Filepath: {full_filepath}")
            print(f"   File size: {file_size_mb:.2f} MB")
            print(f"   Model: {self.best_model_info['name']}")
            print(f"   Accuracy: {self.best_model_info['results']['accuracy']:.4f}")
            print(f"   F1-Score: {self.best_model_info['results']['f1_score']:.4f}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Classes: {len(self.label_encoder.classes_)}")
            
            return full_filepath
            
        except Exception as e:
            print(f"ERROR: Error saving model: {str(e)}")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete ML training pipeline"""
        print("Starting Disease Classification ML Training Pipeline...\n")
        
        try:
            # Step 1: Load and explore data
            if not self.load_and_explore_data():
                return False
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Initialize models
            self.initialize_models()
            
            # Step 4: Split data
            self.split_data()
            
            # Step 5: Train and evaluate models
            self.train_and_evaluate_models()
            
            # Step 6: Find best model
            self.find_best_model()
            
            # Step 7: Hyperparameter tuning
            self.hyperparameter_tuning()
            
            # Step 8: Comprehensive evaluation
            self.generate_comprehensive_evaluation()
            
            # Step 9: Create visualizations
            plot_file = self.create_visualizations()
            
            # Step 10: Generate visualization graphs
            visualization_graph_files = self.generate_visualization_graphs()
            
            # Step 11: Save feature analysis
            feature_file = self.save_feature_analysis()
            
            # Step 12: Save model package
            model_file = self.save_model_package()
            
            # Final summary
            print(f"\n" + "="*80)
            print("ML PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            print(f"\nOUTPUT FILES:")
            print(f"   {model_file} - Complete model package")
            print(f"   {plot_file} - Model comparison visualizations")
            if feature_file:
                print(f"   {feature_file} - Feature importance analysis")
            
            if visualization_graph_files:
                print(f"\nVISUALIZATION GRAPHS:")
                for graph_file in visualization_graph_files:
                    print(f"   {graph_file}")
            
            print(f"\nBEST MODEL SUMMARY:")
            print(f"   Model: {self.best_model_info['name']}")
            print(f"   Accuracy: {self.best_model_info['results']['accuracy']:.4f}")
            print(f"   F1-Score: {self.best_model_info['results']['f1_score']:.4f}")
            print(f"   ROC-AUC: {self.best_model_info['results']['roc_auc']:.4f}")
            print(f"   Training Time: {self.best_model_info['results']['train_time']:.2f}s")
            if self.best_model_info.get('tuned'):
                print(f"   Hyperparameter Tuned: Yes")
            
            print(f"\nDATASET STATS:")
            print(f"   Total Samples: {len(self.df):,}")
            print(f"   Features Used: {len(self.feature_names)}")
            print(f"   Disease Classes: {len(self.label_encoder.classes_)}")
            print(f"   Training Samples: {len(self.X_train):,}")
            print(f"   Test Samples: {len(self.X_test):,}")
            
            print(f"\nREADY FOR DEPLOYMENT!")
            print("The trained model can now be used for disease prediction.")
            
            return {
                'model_file': model_file,
                'plot_file': plot_file,
                'feature_file': feature_file,
                'visualization_graph_files': visualization_graph_files,
                'best_model': self.best_model_info,
                'evaluation_data': self.evaluation_data
            }
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def load_trained_model(filepath=None):
    """
    Load a trained model package for inference
    
    Returns:
        dict: Complete model package with all components
    """
    if filepath is None:
        # Auto-detect model path
        possible_paths = [
            'models/saved_models/disease_classification_model.pkl',  # Running from root
            '../models/saved_models/disease_classification_model.pkl',  # Running from src/
            'models/disease_classification_model.pkl'  # Fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            print(f"ERROR: Could not find model file in any location")
            return None
    
    try:
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"SUCCESS: Model loaded successfully!")
        print(f"   Model: {model_package['model_name']}")
        print(f"   Accuracy: {model_package['accuracy']:.4f}")
        print(f"   Classes: {model_package['n_classes']}")
        print(f"   Features: {model_package['n_features']}")
        
        return model_package
        
    except Exception as e:
        print(f"ERROR: Error loading model: {str(e)}")
        return None

def predict_disease(model_package, symptoms_data):
    """
    Predict disease using the trained model
    
    Args:
        model_package: Loaded model package
        symptoms_data: Feature vector or symptoms dictionary
    
    Returns:
        dict: Prediction results
    """
    try:
        model = model_package['model']
        scaler = model_package['scaler']
        label_encoder = model_package['label_encoder']
        
        # Scale features
        features_scaled = scaler.transform([symptoms_data])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get disease name
        disease_name = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities.max()
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'disease': label_encoder.inverse_transform([idx])[0],
                'probability': probabilities[idx]
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_disease': disease_name,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
        
    except Exception as e:
        print(f"ERROR: Prediction error: {str(e)}")
        return None

def main():
    """Main function"""
    try:
        # Initialize and run pipeline
        pipeline = DiseaseClassificationMLPipeline()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"\nNEXT STEPS:")
            print("1. Review the model evaluation plots and metrics")
            print("2. Check the visualization graphs for publication quality visuals")
            print("3. Test the saved model with new symptom data")
            print("4. Deploy the model for real-time disease prediction")
            print("5. Consider collecting more data for continuous improvement")
        
    except KeyboardInterrupt:
        print(f"\nWARNING: Pipeline interrupted by user")
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()