import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             matthews_corrcoef, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class HeartDiseaseClassifier:
    """
    Heart Disease Classification using Random Forest with comprehensive analysis
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.results = {}

    def load_and_explore_data(self, filepath='heart.csv'):
        """Load and perform initial exploration of the dataset
           Check for NaN values
        """
        # Load the dataset
        self.df = pd.read_csv(filepath)

        print("------- DATASET OVERVIEW -------")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())

        print(f"\nDataset info:")
        print(self.df.info())

        print(f"\nDescriptive statistics:")
        print(self.df.describe())

        # Check for missing values
        missing_values = self.df.isnull().sum(axis=0)
        if missing_values.any():
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])  # displays the columns which have missing values, and their count
        else:
            print("\nNo missing values found!")

        # Explore categorical columns
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        print(f"\n--- CATEGORICAL FEATURES ANALYSIS ---")
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col} unique values:")
                print(self.df[col].value_counts())

        # Target variable distribution
        print(f"\n--- TARGET VARIABLE DISTRIBUTION ---")
        print(self.df['HeartDisease'].value_counts())
        print(f"Heart Disease percentage: {self.df['HeartDisease'].mean()*100:.2f}%")

    def preprocess_data(self):
        """One hot encoding of categorical features, and train test split"""
        print("\n----- PREPROCESSING -----")

        # One hot encoding of categorical variables
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        df_encoded = pd.get_dummies(self.df, columns=categorical_cols, dtype=int)

        # Separate features and target
        features = [col for col in df_encoded.columns if col != 'HeartDisease']
        X = df_encoded[features]
        y = df_encoded['HeartDisease']

        print(f"Number of features after encoding: {len(features)}")
        print(f"Features: {features}")

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        print(f"\nTrain set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")
        print(f"Train target distribution: {np.bincount(self.y_train)}")
        print(f"Test target distribution: {np.bincount(self.y_test)}")

        return features

    def perform_eda_visualizations(self):
        """Create comprehensive visualizations for analysis"""
        print("\n------- CREATING EDA VISUALIZATIONS -------")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Heart Disease Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

        # 1. Target distribution
        ax1 = axes[0, 0]
        target_counts = self.df['HeartDisease'].value_counts()
        ax1.pie(target_counts.values, labels=['No Disease', 'Heart Disease'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Target Variable Distribution')

        # 2. Age distribution by target
        ax2 = axes[0, 1]
        self.df.boxplot(column='Age', by='HeartDisease', ax=ax2)
        ax2.set_title('Age Distribution by Heart Disease')
        ax2.set_xlabel('Heart Disease (0=No, 1=Yes)')

        # 3. Chest pain type vs heart disease
        ax3 = axes[0, 2]
        chest_pain_crosstab = pd.crosstab(self.df['ChestPainType'], self.df['HeartDisease'])
        chest_pain_crosstab.plot(kind='bar', ax=ax3, rot=45)
        ax3.set_title('Chest Pain Type vs Heart Disease')
        ax3.legend(['No Disease', 'Heart Disease'])

        # 4. Correlation heatmap
        ax4 = axes[1, 0]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Feature Correlation Heatmap')

        # 5. MaxHR vs Age colored by target
        ax5 = axes[1, 1]
        scatter = ax5.scatter(self.df['Age'], self.df['MaxHR'], 
                              c=self.df['HeartDisease'], cmap='viridis', alpha=0.6)
        ax5.set_xlabel('Age')
        ax5.set_ylabel('Max Heart Rate')
        ax5.set_title('Max Heart Rate vs Age')
        plt.colorbar(scatter, ax=ax5, label='Heart Disease')

        # 6. Exercise Angina distribution
        ax6 = axes[1, 2]
        exercise_crosstab = pd.crosstab(self.df['ExerciseAngina'], self.df['HeartDisease'])
        exercise_crosstab.plot(kind='bar', ax=ax6)
        ax6.set_title('Exercise Angina vs Heart Disease')
        ax6.set_xticklabels(['No', 'Yes'], rotation=0)
        ax6.legend(['No Disease', 'Heart Disease'])

        plt.tight_layout()
        plt.savefig('results/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # See the correlation of numeric features with the target
        plt.figure(figsize=(10, 8))
        feature_importance = self.df[numeric_cols].corr()['HeartDisease'].abs().sort_values(ascending=True)
        feature_importance = feature_importance.drop('HeartDisease')
        feature_importance.plot(kind='barh')
        plt.title('Feature Importance (Absolute Correlation with Target)')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("EDA visualizations saved to 'results/' directory")

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning """
        print("\n------- HYPERPARAMETER TUNING -------")

        # Expanded parameter grid based on research
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 500, 800, 1000],
            "max_depth": [None, 5, 10, 15, 20, 25, 30, 50],
            "min_samples_split": [2, 5, 10, 15, 20, 50, 100, 200],
            "min_samples_leaf": [1, 2, 4, 8, 15, 25, 50],
            # not necessary hence commented out
            # "max_features": ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
            # "bootstrap": [True, False],
            # "criterion": ['gini', 'entropy', 'log_loss']
        }

        print(f"Total combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
        print("Using RandomizedSearchCV for efficiency...")

        rf_model = RandomForestClassifier(random_state=self.random_state)

        # randomized search
        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=100,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=-1,
            refit='f1',  # Refit on F1 score for balanced performance
            cv=5,
            verbose=1,
            random_state=self.random_state
        )

        print("Fitting RandomSearchCV...")
        random_search.fit(self.X_train, self.y_train)

        self.best_model = random_search.best_estimator_

        print(f"\nBest Hyperparameters: {random_search.best_params_}")
        print(f"Best Cross-Validation F1 Score: {random_search.best_score_:.4f}")

        # Store results
        self.results['best_params'] = random_search.best_params_
        self.results['best_cv_score'] = random_search.best_score_
        self.results['cv_results'] = random_search.cv_results_

        return random_search

    def evaluate_model(self):
        """Comprehensive model evaluation with multiple metrics"""
        print("\n------- MODEL EVALUATION -------")

        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]   # probabilities of classification class=1

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }

        print("\nTest Set Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        # Classification report
        cr = classification_report(self.y_test, y_pred, target_names=['No Disease', 'Heart Disease'])
        print(f"\nClassification Report:")
        print(cr)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Store results
        self.results.update(metrics)   # adds the metrics dictionary to self.results
        self.results['classification_report'] = cr
        self.results['confusion_matrix'] = cm

        return metrics

    def create_evaluation_visualizations(self):
        """Creates comprehensive evaluation visualizations"""
        print("\n------- CREATING EVALUATION VISUALIZATIONS -------")

        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # Create a figure with evaluation plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # 2. ROC Curve
        ax2 = axes[0, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")

        # 3. Feature Importance
        ax3 = axes[0, 2]
        feature_importance = self.best_model.feature_importances_
        feature_names = self.X_train.columns
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=True).tail(15)
        ax3.barh(importance_df['feature'], importance_df['importance'])
        ax3.set_title('Top 15 Feature Importances')
        ax3.set_xlabel('Importance')

        # 4. Prediction Distribution
        ax4 = axes[1, 0]
        ax4.hist(y_pred_proba[self.y_test == 0], bins=20, alpha=0.7, label='No Disease', color='blue')
        ax4.hist(y_pred_proba[self.y_test == 1], bins=20, alpha=0.7, label='Heart Disease', color='red')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Probability Distribution')
        ax4.legend()

        # 5. Learning Curve
        ax5 = axes[1, 1]
        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, self.X_train, self.y_train, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=self.random_state
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        ax5.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax5.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        ax5.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
        ax5.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
        ax5.set_xlabel('Training Set Size')
        ax5.set_ylabel('Accuracy Score')
        ax5.set_title('Learning Curve')
        ax5.legend(loc='best')
        ax5.grid(True)

        # 6. Metrics Comparison
        ax6 = axes[1, 2]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'ROC AUC']
        metrics_values = [
            self.results['accuracy'], self.results['precision'], 
            self.results['recall'], self.results['f1_score'],
            self.results['matthews_corrcoef'], self.results['roc_auc']
        ]

        bars = ax6.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 
                                                             'khaki', 'plum', 'salmon'])
        ax6.set_title('Performance Metrics Comparison')
        ax6.set_ylabel('Score')
        ax6.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{value:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Evaluation visualizations saved to 'results/' directory")

    def save_results_to_csv(self):
        """Save comprehensive results to CSV files"""
        print("\n=== SAVING RESULTS ===")

        # Model performance summary
        performance_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Matthews Correlation Coefficient', 'ROC AUC'],
            'Score': [
                self.results['accuracy'], self.results['precision'], 
                self.results['recall'], self.results['f1_score'],
                self.results['matthews_corrcoef'], self.results['roc_auc']
            ]
        })
        performance_df.to_csv('results/model_performance.csv', index=False)

        # Feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': self.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        feature_importance_df.to_csv('results/feature_importance.csv', index=False)

        # Best hyperparameters
        best_params_df = pd.DataFrame([self.results['best_params']])
        best_params_df.to_csv('results/best_hyperparameters.csv', index=False)

        # Predictions on test set
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        predictions_df = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': y_pred,
            'Probability': y_pred_proba
        })
        predictions_df.to_csv('results/test_predictions.csv', index=False)

        print("Results saved to CSV files in 'results/' directory:")
        print("- model_performance.csv")
        print("- feature_importance.csv") 
        print("- best_hyperparameters.csv")
        print("- test_predictions.csv")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 80)
        print("HEART DISEASE CLASSIFICATION - COMPLETE ANALYSIS")
        print("=" * 80)

        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)

        # Step 1: Load and explore data
        self.load_and_explore_data()

        # Step 2: Preprocess data
        features = self.preprocess_data()

        # Step 3: Create EDA visualizations
        self.perform_eda_visualizations()

        # Step 4: Hyperparameter tuning
        grid_search = self.hyperparameter_tuning()

        # Step 5: Evaluate model
        model_metrics = self.evaluate_model()

        # Step 6: Create evaluation visualizations
        self.create_evaluation_visualizations()

        # Step 7: Save results
        self.save_results_to_csv()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Final Model Performance:")
        print(f"- Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"- F1 Score: {model_metrics['f1_score']:.4f}")
        print(f"- ROC AUC: {model_metrics['roc_auc']:.4f}")
        print("\nAll results and visualizations saved in 'results/' directory")

        return self.best_model, model_metrics


if __name__ == "__main__":
    # Run the complete analysis
    classifier = HeartDiseaseClassifier()
    model, metrics = classifier.run_complete_analysis()
