import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("cleaned_train_set.csv")
data['label'] = data['text'].apply(lambda x: 0 if '__label__1' in x else 1)
class_0 = data[data['label'] == 0]
class_1 = data[data['label'] == 1]

# Sample for test dataset
sampled_class_0 = class_0.sample(250, random_state=42)
sampled_class_1 = class_1.sample(250, random_state=42)

# Create test dataset
test_data = pd.concat([sampled_class_0, sampled_class_1])
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data.to_csv("test_set.csv", index=False)

# Exclude test_data from the original data to create a new training dataset
train_data = data.drop(test_data.index)

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data.to_csv("new_train_set.csv", index=False)