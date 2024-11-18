print("Missing values in each column:")
print(data.isnull().sum())

duplicate_count = data.duplicated(subset=['review']).sum()
print(f"\nNumber of duplicate reviews: {duplicate_count}")

print("\nReview length statistics (in words):")
print(data['review_length'].describe())

plt.figure(figsize=(10, 5))
plt.hist(data['review_length'], bins=30, color="purple", alpha=0.7)
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

non_ascii_reviews = data['review'].apply(lambda x: any(ord(char) > 127 for char in x))
print(f"\nNumber of reviews with non-ASCII characters: {non_ascii_reviews.sum()}")

print("\nExamples of reviews with non-ASCII characters:")
print(data[non_ascii_reviews].sample(5)['review'])

data = data.drop(columns=['star_rating'])

print(data.head())

data.to_csv("cleaned_train_set.csv", index=False)
from google.colab import files
files.download("cleaned_train_set.csv")