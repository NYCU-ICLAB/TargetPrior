import pandas as pd
import numpy as np
import random
import os
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

# --- 1. Data loading and preprocessing ---
file_train = "train_pre.csv"


if os.path.exists(file_train):
    train_data = pd.read_csv(file_train)
    X_train = train_data.iloc[:, 1:-1].values
    y_train = train_data.iloc[:, -1].values
    feature_names = train_data.columns[1:-1].tolist()
    print(f"Loaded data from files. Features: {len(feature_names)}")
else:
    print("Warning: Data files not found.")
    

# Ensure all indices are integers
total_feature_pool = list(range(X_train.shape[1]))

# --- 2. Parameter settings ---
# GA parameters
population_size = 50
generations_per_stage = 1000  # Number of generations per feature count stage (original Gmax) 1000
tournament_size = 2        # Tournament selection size
crossover_prob = 0.8
mutation_prob = 0.05
elitism_size = 1

# Inheritance parameters (IBCGA logic)
r_start = 30  # Starting number of features
r_end = 10    # Ending number of features 10 

# SVM parameter search range
param_bounds = {
    "C": (0.1, 1000.0),
    "gamma": (0.0001, 10.0)
}

# Cross-validation settings
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# --- 3. Core functions ---

def create_individual(num_features):
    """
    Create an individual with randomly selected num_features feature indices
    and random C and gamma
    """
    return {
        "C": random.uniform(*param_bounds["C"]),
        "gamma": random.uniform(*param_bounds["gamma"]),
        "features": random.sample(total_feature_pool, num_features) # Randomly select r features
    }

def evaluate(individual):
    """
    Calculate fitness: 5-fold CV MCC
    """
    selected_indices = individual["features"]
    
    # Extract the corresponding feature subset
    X_train_sub = X_train[:, selected_indices]
    
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X_train_sub, y_train):
        X_tr, X_val = X_train_sub[train_idx], X_train_sub[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        clf = SVC(
            C=individual["C"],
            gamma=individual["gamma"],
            kernel='rbf', 
            probability=False, # Disable probability during training to speed up
            random_state=42
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        
        mcc = matthews_corrcoef(y_val, y_pred)
        if not np.isnan(mcc):
            cv_scores.append(mcc)
            
    return np.mean(cv_scores) if cv_scores else -1.0

def tournament_selection(population, fitness_scores):
    """
    Tournament Selection: Randomly select k individuals and return the best one
    """
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_idx = selected_indices[0]
    best_fit = fitness_scores[best_idx]
    
    for idx in selected_indices[1:]:
        if fitness_scores[idx] > best_fit:
            best_fit = fitness_scores[idx]
            best_idx = idx
            
    return population[best_idx]

def swap_mutation(individual, current_r):
    """
    Swap Mutation: Ensure the number of features remains fixed at current_r
    Remove one from selected features and add one from unselected features
    """
    if random.random() < mutation_prob:
        # SVM parameter mutation
        if random.random() < 0.5:
            individual["C"] = random.uniform(*param_bounds["C"])
        else:
            individual["gamma"] = random.uniform(*param_bounds["gamma"])
            
    if random.random() < mutation_prob:
        # Feature Swap mutation
        current_feats = set(individual["features"])
        available_feats = list(set(total_feature_pool) - current_feats)
        
        if available_feats: # Ensure there are features to swap
            remove_feat = random.choice(list(current_feats))
            add_feat = random.choice(available_feats)
            
            individual["features"].remove(remove_feat)
            individual["features"].append(add_feat)
            
    return individual

def inheritance_step(population):
    """
    Inheritance Step:
    For each individual in the population, randomly remove one feature (from 1 to 0).
    Reduce the number of features from r to r-1.
    """
    new_population = []
    for ind in population:
        new_ind = ind.copy()
        current_feats = new_ind["features"]
        
        if len(current_feats) > 1:
            # Randomly remove one feature
            remove_feat = random.choice(current_feats)
            # Must use copy to avoid modifying the original reference
            new_feats = current_feats.copy()
            new_feats.remove(remove_feat)
            new_ind["features"] = new_feats
        
        new_population.append(new_ind)
    return new_population

# --- 4. Main program loop ---

output_path = "EL_Inheritance_output.txt"
global_best_individual = None
global_best_fitness = -float('inf')

with open(output_path, "w") as f:
    f.write("Starting Inheritance Optimization\n")
    f.write(f"Feature count range: {r_start} -> {r_end}\n")
    
    # Initialize population (r = r_start)
    current_population = [create_individual(r_start) for _ in range(population_size)]
    print(f"Initialized population with {r_start} features.")

    # Outer loop: Inheritance Loop (feature count decreasing)
    for r in range(r_start, r_end - 1, -1):
        print(f"\n=== Current Feature Count: {r} ===")
        f.write(f"\n=== Current Feature Count: {r} ===\n")
        
        # Inner loop: GA Loop (evolve parameters and feature combinations)
        best_ind_stage = None
        best_fit_stage = -float('inf')
        
        for gen in range(generations_per_stage):
            # 1. Evaluation (Parallelized)
            fitness_values = Parallel(n_jobs=-1)(delayed(evaluate)(ind) for ind in current_population)
            
            # Store best solution
            for i, fit in enumerate(fitness_values):
                if fit > best_fit_stage:
                    best_fit_stage = fit
                    best_ind_stage = current_population[i].copy()
                if fit > global_best_fitness:
                    global_best_fitness = fit
                    global_best_individual = current_population[i].copy()
            
            # 2. Elitism
            sorted_indices = np.argsort(fitness_values)[::-1]
            next_population = [current_population[i] for i in sorted_indices[:elitism_size]]
            
            # 3. Evolve next generation
            while len(next_population) < population_size:
                # Selection
                p1 = tournament_selection(current_population, fitness_values)
                p2 = tournament_selection(current_population, fitness_values)
                
                # Crossover (Average SVM parameters, inherit features from P1)
                child = {
                    "C": (p1["C"] + p2["C"]) / 2,
                    "gamma": (p1["gamma"] + p2["gamma"]) / 2,
                    "features": p1["features"].copy() # Simple inheritance: inherit features from P1, rely on Mutation to swap
                }
                
                # Mutation
                child = swap_mutation(child, r)
                next_population.append(child)
            
            current_population = next_population
            
            if (gen + 1) % 10 == 0:
                print(f"  Gen {gen+1}/{generations_per_stage} | Best MCC: {best_fit_stage:.4f}")
                f.write(f"  Gen {gen+1} | Best MCC: {best_fit_stage:.4f}\n")

        print(f"Stage {r} finished. Best MCC: {best_fit_stage:.4f}")
        
        # Inheritance Step: If not the last stage, perform inheritance (r -> r-1)
        if r > r_end:
            print(f"Performing Inheritance: Reducing features from {r} to {r-1}...")
            current_population = inheritance_step(current_population)

    # --- 5. Output final results ---
    print("\nOptimization Complete.")
    f.write("\n" + "="*50 + "\n")
    f.write("Global Best Solution Found:\n")
    
    f.write(f"Features ({len(global_best_individual['features'])}): {global_best_individual['features']}\n")
    f.write(f"SVM C: {global_best_individual['C']}\n")
    f.write(f"SVM gamma: {global_best_individual['gamma']}\n")
    f.write(f"Best CV MCC: {global_best_fitness}\n")
    
    # Convert feature names
    best_feat_names = [feature_names[i] for i in global_best_individual['features']]
    f.write(f"Feature Names: {', '.join(best_feat_names)}\n")
    
    print("\nBest Individual:")
    print(f"Features: {len(global_best_individual['features'])}")
    print(f"C: {global_best_individual['C']:.4f}")
    print(f"Gamma: {global_best_individual['gamma']:.4f}")
    print(f"MCC: {global_best_fitness:.4f}")

    