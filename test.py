import pandas as pd

# Load the datasets
empty = pd.read_excel("empty_tweets.xlsx")
filled = pd.read_excel("filled_tweets.xlsx")

# Randomly select 5 entries from the empty dataset
# 623 funny
random_empty_entries = empty.sample(n=5, random_state=4)

# Extract the IDs of the randomly selected rows
selected_ids = random_empty_entries['id'].tolist()

# Filter the filled dataset using the extracted IDs
filled_entries = filled[filled['id'].isin(selected_ids)]

# Ensure that the entries are in the same order as in the random_empty_entries dataset
filled_entries = filled_entries.set_index('id').loc[selected_ids].reset_index()

# Display results
print("5 Random Entries with Empty Locations (Before Prediction):")
print(random_empty_entries)

print("\n5 Random Entries with Filled Locations (After Prediction):")
print(filled_entries)

# Save the results as Excel files
random_empty_entries.to_excel("empty_random.xlsx", index=False)
filled_entries.to_excel("filled_random.xlsx", index=False)
