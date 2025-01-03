import numpy as np
import pandas as pd
# Example vectors (you can replace these with actual vectors)
df = pd.read_csv("embeddings.csv",header=None)
vec_1 = np.array(df.loc[0])
vec_2 = np.array(df.loc[1])
vec_3 = np.array(df.loc[2])
vec_4 = np.array(df.loc[3])
vec_5 = np.array(df.loc[4])

# Weights for each vector
weights = np.array([0.05, 0.10, 0.80, 0.025, 0.025])

# List of vectors
vectors = [vec_1, vec_2, vec_3, vec_4, vec_5]

# Calculate the weighted sum
final_vec = sum(weight * vec for weight, vec in zip(weights, vectors))
final_vec = ", ".join([str(vec) for vec in final_vec])

print(len(final_vec))
print(final_vec)
