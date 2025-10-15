import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

# Load data, please change to your file name
train_data = pd.read_csv("train_pre.csv")
test_data = pd.read_csv("ind_pre_total.csv")

# Features and label
features = train_data.columns[1:-1].tolist()
# label = 'LN_5'

X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

# Initialize genetic algorithm parameters
population_size = 500
generations = 200
crossover_prob = 0.7
mutation_prob = 0.3
elitism_size = 1

# Initialize the search scope
param_bounds = {
    "C": (0.1, 1000.0),         # Control the regularization strength, smaller values indicate stronger regularization
    "gamma": (0.0001, 10.0),    # Parameters of RBF kernel function
    "feature_count": (10, 30)   # How many features to use (from 10 to 30)
}

# Initialize k-fold cross-validation
n_folds = 5  # 5-fold cross-validation
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize individual
def create_individual():
    """
    Create a random individual with parameters within defined bounds
    """
    return {
        "C": random.uniform(*param_bounds["C"]),
        "gamma": random.uniform(*param_bounds["gamma"]),
        "feature_count": random.randint(*param_bounds["feature_count"])
    }
    
# Feature selection 
def select_features(X_train, y_train, X_test, feature_count):
    """
    Perform feature selection using SelectKBest
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        feature_count: Number of features to select
        
    Returns:
        X_train_selected: Selected training features
        X_test_selected: Selected test features
        selected_indices: Indices of selected features
    """
    selector = SelectKBest(f_classif, k=feature_count)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    # Get the selected feature index
    selected_indices = np.where(selector.get_support())[0]
    return X_train_selected, X_test_selected, selected_indices

# Fitness function using 5-fold cross-validation and MCC as the evaluation metric
def evaluate(individual):
    """
    Evaluate an individual using 5-fold cross-validation
    
    Args:
        individual: Dictionary containing model parameters
        
    Returns:
        avg_mcc: Average Matthews Correlation Coefficient across all folds
    """
    feature_count = int(individual["feature_count"])
    # If feature selection has been calculated before for this number of features, reuse the result
    if feature_count not in feature_selectors:
        X_train_selected, X_test_selected, selected_indices = select_features(
            X_train, y_train, X_test, feature_count
        )
        feature_selectors[feature_count] = {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'indices': selected_indices
        }
    else:
        X_train_selected = feature_selectors[feature_count]['X_train']
        X_test_selected = feature_selectors[feature_count]['X_test']

    # Cross-validation scores
    cv_scores = []
    
    # Run cross-validation
    for train_idx, val_idx in kf.split(X_train_selected, y_train):
        # Split data for current fold
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train SVM model
        clf = SVC(
            C=individual["C"],
            gamma=individual["gamma"],
            kernel='rbf', 
            probability=True,
            random_state=42
        )
        
        # Fit on training fold
        clf.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_pred = clf.predict(X_val_fold)
        
        # Calculate MCC for this fold
        fold_mcc = matthews_corrcoef(y_val_fold, y_pred)
        if not np.isnan(fold_mcc):
            cv_scores.append(fold_mcc)
    
    # Average MCC across all folds
    avg_mcc = np.mean(cv_scores) if cv_scores else -1.0
    return avg_mcc

# Global variables are used to store feature selection results
feature_selectors = {}

# Generate initial population
population = [create_individual() for _ in range(population_size)]

# Set output file path
output_path = "EL_output_cv.txt"

# Genetic algorithm main loop
with open(output_path, "w") as f:
    best_individual = None
    best_fitness = -float('inf')
    
    f.write("Starting genetic algorithm optimization with 5-fold cross-validation\n")
    f.write(f"Population size: {population_size}, Generations: {generations}\n")
    f.write(f"Crossover probability: {crossover_prob}, Mutation probability: {mutation_prob}\n\n")
    
    for gen in range(generations):
        # Parallel calculation of fitness
        fitness_scores = Parallel(n_jobs=-1)(delayed(evaluate)(ind) for ind in population)
        fitness_scores = list(zip(population, fitness_scores))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retain the best individual
        current_best_individual, current_best_fitness = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_individual = current_best_individual.copy()
            best_fitness = current_best_fitness
        next_population = [ind for ind, _ in fitness_scores[:elitism_size]]

        # Selection and crossover
        while len(next_population) < population_size:
            # Select two parents
            parent1 = random.choice([ind for ind, _ in fitness_scores[:int(len(fitness_scores)/2)]])
            parent2 = random.choice([ind for ind, _ in fitness_scores[:int(len(fitness_scores)/2)]])

            # Cross over
            if random.random() < crossover_prob:
                child = {
                    "C": (parent1["C"] + parent2["C"]) / 2,
                    "gamma": (parent1["gamma"] + parent2["gamma"]) / 2,
                    "feature_count": int((parent1["feature_count"] + parent2["feature_count"]) / 2)
                }
            else:
                child = parent1.copy()

            # Mutation
            if random.random() < mutation_prob:
                if random.random() < 0.33:  # 1/3 chance to change C
                    child["C"] = random.uniform(*param_bounds["C"])
                elif random.random() < 0.5:  # Change gamma with 1/3 chance
                    child["gamma"] = random.uniform(*param_bounds["gamma"])
                else:  # 1/3 chance to change the number of features
                    child["feature_count"] = random.randint(*param_bounds["feature_count"])

            next_population.append(child)

        # Update population
        population = next_population
        current_best_individual, current_best_fitness = fitness_scores[0]
        f.write(f"Generation {gen+1} | Best CV MCC: {current_best_fitness:.4f}\n")
        print(f"Generation {gen+1} | Best CV MCC: {current_best_fitness:.4f}")

    f.write("\n" + "="*50 + "\n")
    f.write("Genetic Algorithm Optimization Complete\n")
    f.write("="*50 + "\n\n")
    
    # Output best parameters
    f.write("Best individual parameters:\n")
    for param, value in best_individual.items():
        f.write(f"{param}: {value}\n")
    print("Best individual:", best_individual)
    
    # Use the optimal number of features for feature selection
    best_feature_count = int(best_individual["feature_count"])
    if best_feature_count in feature_selectors:
        # Reuse previously calculated selection results
        X_train_best = feature_selectors[best_feature_count]['X_train']
        X_test_best = feature_selectors[best_feature_count]['X_test']
        selected_feature_indices = feature_selectors[best_feature_count]['indices']
    else:
        # Recalculate feature selection performance
        X_train_best, X_test_best, selected_feature_indices = select_features(
            X_train, y_train, X_test, best_feature_count
        )
    
    # Detailed cross-validation performance
    f.write("\nDetailed Cross-Validation Performance:\n")
    f.write("-"*40 + "\n")
    
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_best, y_train)):
        # Split data for current fold
        X_train_fold, X_val_fold = X_train_best[train_idx], X_train_best[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train SVM model
        best_model_fold = SVC(
            C=best_individual["C"],
            gamma=best_individual["gamma"],
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Fit on training fold
        best_model_fold.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_pred_fold = best_model_fold.predict(X_val_fold)
        
        # Calculate metrics for this fold
        cm_fold = confusion_matrix(y_val_fold, y_pred_fold)
        sensitivity_fold = cm_fold[1, 1] / (cm_fold[1, 0] + cm_fold[1, 1]) if (cm_fold[1, 0] + cm_fold[1, 1]) > 0 else 0
        specificity_fold = cm_fold[0, 0] / (cm_fold[0, 0] + cm_fold[0, 1]) if (cm_fold[0, 0] + cm_fold[0, 1]) > 0 else 0
        accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
        mcc_fold = matthews_corrcoef(y_val_fold, y_pred_fold)
        
        fold_results.append({
            'sensitivity': sensitivity_fold,
            'specificity': specificity_fold,
            'accuracy': accuracy_fold,
            'mcc': mcc_fold
        })
        
        f.write(f"Fold {fold_idx+1}:\n")
        f.write(f"  Confusion Matrix:\n{cm_fold}\n")
        f.write(f"  Sensitivity: {sensitivity_fold:.2f}\n")
        f.write(f"  Specificity: {specificity_fold:.2f}\n")
        f.write(f"  Accuracy: {accuracy_fold:.2f}\n")
        f.write(f"  MCC: {mcc_fold:.2f}\n\n")
    
    # Average cross-validation results
    avg_sensitivity = np.mean([res['sensitivity'] for res in fold_results])
    avg_specificity = np.mean([res['specificity'] for res in fold_results])
    avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
    avg_mcc = np.mean([res['mcc'] for res in fold_results])
    
    std_sensitivity = np.std([res['sensitivity'] for res in fold_results])
    std_specificity = np.std([res['specificity'] for res in fold_results])
    std_accuracy = np.std([res['accuracy'] for res in fold_results])
    std_mcc = np.std([res['mcc'] for res in fold_results])
    
    f.write("Cross-Validation Average Results:\n")
    f.write(f"Sensitivity: {avg_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
    f.write(f"Specificity: {avg_specificity:.2f} ± {std_specificity:.2f}\n")
    f.write(f"Accuracy: {avg_accuracy:.2f} ± {std_accuracy:.2f}\n")
    f.write(f"MCC: {avg_mcc:.2f} ± {std_mcc:.2f}\n\n")
    
    # Train final model on all training data
    best_model = SVC(
        C=best_individual["C"],
        gamma=best_individual["gamma"],
        kernel='rbf',
        probability=True,
        random_state=42
    )
    best_model.fit(X_train_best, y_train)

    # Prediction on the full training set
    predictions_train = best_model.predict(X_train_best)
    cm_train = confusion_matrix(y_train, predictions_train)
    sensitivity_train = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1]) if (cm_train[1, 0] + cm_train[1, 1]) > 0 else 0
    specificity_train = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1]) if (cm_train[0, 0] + cm_train[0, 1]) > 0 else 0
    accuracy_train = accuracy_score(y_train, predictions_train)
    mcc_train = matthews_corrcoef(y_train, predictions_train)

    # Test set prediction
    predictions_test = best_model.predict(X_test_best)
    cm_test = confusion_matrix(y_test, predictions_test)
    sensitivity_test = cm_test[1, 1] / (cm_test[1, 0] + cm_test[1, 1]) if (cm_test[1, 0] + cm_test[1, 1]) > 0 else 0
    specificity_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1]) if (cm_test[0, 0] + cm_test[0, 1]) > 0 else 0
    accuracy_test = accuracy_score(y_test, predictions_test)
    mcc_test = matthews_corrcoef(y_test, predictions_test)

    # Write selected features
    if 'features' in globals() and len(features) >= max(selected_feature_indices)+1:
        selected_feature_names = [features[i] for i in selected_feature_indices]
        f.write("\nSelected Features:\n")
        f.write(", ".join(selected_feature_names) + "\n")
        f.write(f"Feature count: {best_feature_count}\n")
        
        print("\nSelected Features:")
        print(", ".join(selected_feature_names))
    else:
        f.write("\nSelected Feature Indices:\n")
        f.write(", ".join(map(str, selected_feature_indices)) + "\n")
        f.write(f"Feature count: {best_feature_count}\n")
        
        print("\nSelected Feature Indices:")
        print(", ".join(map(str, selected_feature_indices)))
    print(f"Feature count: {best_feature_count}")
    
    # Write final model performance metrics
    f.write("\nFinal Model Performance on Full Training Set:\n")
    f.write(f"Confusion Matrix:\n{cm_train}\n")
    f.write(f"Sensitivity: {sensitivity_train:.2f}\n")
    f.write(f"Specificity: {specificity_train:.2f}\n")
    f.write(f"Accuracy: {accuracy_train:.2f}\n")
    f.write(f"MCC: {mcc_train:.2f}\n")
    
    print("\nFinal Model Performance on Full Training Set:")
    print(f"Confusion Matrix:\n{cm_train}")
    print(f"Sensitivity: {sensitivity_train:.2f}")
    print(f"Specificity: {specificity_train:.2f}")
    print(f"Accuracy: {accuracy_train:.2f}")
    print(f"MCC: {mcc_train:.2f}")
    
    f.write("\nFinal Model Performance on Test Set:\n")
    f.write(f"Confusion Matrix:\n{cm_test}\n")
    f.write(f"Sensitivity: {sensitivity_test:.2f}\n")
    f.write(f"Specificity: {specificity_test:.2f}\n")
    f.write(f"Accuracy: {accuracy_test:.2f}\n")
    f.write(f"MCC: {mcc_test:.2f}\n")
    
    print("\nFinal Model Performance on Test Set:")
    print(f"Confusion Matrix:\n{cm_test}")
    print(f"Sensitivity: {sensitivity_test:.2f}")
    print(f"Specificity: {specificity_test:.2f}")
    print(f"Accuracy: {accuracy_test:.2f}")
    print(f"MCC: {mcc_test:.2f}")
    
    # Output support vector information
    support_vector_count = best_model.n_support_
    f.write("\nSupport Vector Count:\n")
    f.write(f"Class 0: {support_vector_count[0]}\n")
    f.write(f"Class 1: {support_vector_count[1]}\n")
    f.write(f"Total: {sum(support_vector_count)}\n")
    
    print("\nSupport Vector Count:")
    print(f"Class 0: {support_vector_count[0]}")
    print(f"Class 1: {support_vector_count[1]}")
    print(f"Total: {sum(support_vector_count)}")