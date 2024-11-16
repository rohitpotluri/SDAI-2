import pandas as pd


data = pd.read_csv("cleaned_train_set.csv")
data_class_0 = data[data['label'] == 0].sample(5000, random_state=42)
data_class_1 = data[data['label'] == 1].sample(5000, random_state=42)

# Concatenate both subsets to form a new balanced dataset
balanced_data = pd.concat([data_class_0, data_class_1])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv('Bert_set_final.csv', index=False)

# Verify the dataset
print(balanced_data.head())
print("Total rows:", len(balanced_data))