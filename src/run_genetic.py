import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from data import read_data 

POP_SIZE = 20           
MAX_GEN = 15            
MUTATION_RATE = 0.05    
SEEDS = [31, 153, 2021]

def calculate_fitness(chromosome, X_train_np, y_train, X_val_np, y_val):
    selected_indices = np.where(chromosome == 1)[0]
    
    if len(selected_indices) == 0:
        return 0.0
    
    model = GaussianNB()
    model.fit(X_train_np[:, selected_indices], y_train)
    preds = model.predict(X_val_np[:, selected_indices])
    acc = accuracy_score(y_val, preds)
    
    penalty = 0.001 * len(selected_indices)
    fitness = acc - penalty
    return fitness if fitness > 0 else 0.0

def run_ga_experiment(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    X_train, X_val, y_train, y_val = read_data(seed=seed_value)
    
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    X_train_np = X_train.values
    X_val_np = X_val.values
    num_features = X_train_np.shape[1]
    
    population = np.random.randint(0, 2, size=(POP_SIZE, num_features))
    
    best_fitness_history = []
    diversity_history = [] 
    
    for gen in range(MAX_GEN):
        fitness_scores = np.array([calculate_fitness(ind, X_train_np, y_train, X_val_np, y_val) for ind in population])
        
        best_fitness = np.max(fitness_scores)
        diversity_indicator = np.std(fitness_scores)
        
        best_fitness_history.append(best_fitness)
        diversity_history.append(diversity_indicator)
        
        new_population = []
        for _ in range(POP_SIZE):
            # choose two parents
            i, j = np.random.choice(POP_SIZE, 2, replace=False)
            parent1 = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
            i, j = np.random.choice(POP_SIZE, 2, replace=False)
            parent2 = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
            
            # Crossover
            mask = np.random.randint(0, 2, size=num_features)
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            mut_mask = np.random.rand(num_features) < MUTATION_RATE
            child = np.logical_xor(child, mut_mask).astype(int)
            
            new_population.append(child)
            
        population = np.array(new_population)
        
    return best_fitness_history[-1], diversity_history[-1]

print(f"{'Seed':<10} | {'Best Fitness':<15} | {'Diversity (Std Dev of Fitness)':<30}")
print("-" * 60)

for s in SEEDS:
    best_fit, final_diversity = run_ga_experiment(s)
    print(f"{s:<10} | {best_fit:<15.4f} | {final_diversity:<30.4f}")