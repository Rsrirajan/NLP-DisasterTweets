import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from textblob import TextBlob
from sklearn.cluster import KMeans
import scipy.sparse as sp

# Add sentiment features using TextBlob
def extract_sentiment_features(text):
    blob = TextBlob(text)
    return pd.Series({'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity})

# Function to compute region word frequencies
def region_word_frequencies(df):
    """Compute region-specific word frequencies from text."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.85, min_df=5)
    X = vectorizer.fit_transform(df['text'])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Function to add clusters as features
def add_cluster_features(df, num_clusters, train_vectors):
    """Add KMeans cluster labels as features."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(train_vectors)
    return kmeans.predict(train_vectors)

# Load dataset
tweets_df = pd.read_csv("tweets_country_region.csv")

# Drop rows where 'region' is NaN
tweets_df = tweets_df.dropna(subset=['region'])

# Preprocess text and sentiment
tweets_df['text'] = tweets_df['text'].fillna("").astype(str)
sentiment_features = tweets_df['text'].apply(extract_sentiment_features)
tweets_df = pd.concat([tweets_df, sentiment_features], axis=1)

# Split data into training and testing sets
train_df, test_df = train_test_split(
    tweets_df, test_size=0.2, random_state=42, stratify=tweets_df['region']
)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=train_df['region'].unique(),
    y=train_df['region']
)
class_weight_dict = dict(zip(train_df['region'].unique(), class_weights))

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), max_df=0.85, min_df=5)
X_train_tfidf = vectorizer.fit_transform(train_df['text'])
X_test_tfidf = vectorizer.transform(test_df['text'])

# Add region-specific word frequencies (based on training data)
region_frequencies = region_word_frequencies(train_df)

# Add KMeans cluster labels based on training TF-IDF
num_clusters = 5  # Number of clusters
train_clusters = add_cluster_features(train_df, num_clusters, X_train_tfidf)
test_clusters = add_cluster_features(test_df, num_clusters, X_test_tfidf)

# Combine TF-IDF features, sentiment features, and cluster labels
X_train = sp.hstack([
    X_train_tfidf,
    sp.csr_matrix(train_df[['polarity', 'subjectivity']].values),
    sp.csr_matrix(train_clusters.reshape(-1, 1))
])
X_test = sp.hstack([
    X_test_tfidf,
    sp.csr_matrix(test_df[['polarity', 'subjectivity']].values),
    sp.csr_matrix(test_clusters.reshape(-1, 1))
])

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
rf_model.fit(X_train, train_df['region'])

# Predict the region for the test data
test_df['predicted_region'] = rf_model.predict(X_test)

# Evaluate the model using a classification report
print("Classification Report:\n", classification_report(test_df['region'], test_df['predicted_region']))

# Analyze predictions
unique_predicted_classes = test_df['predicted_region'].unique()
predicted_report = {}

for cls in unique_predicted_classes:
    true_positives = ((test_df['region'] == cls) & (test_df['predicted_region'] == cls)).sum()
    false_positives = ((test_df['region'] != cls) & (test_df['predicted_region'] == cls)).sum()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    predicted_report[cls] = {
        'Predicted as': cls,
        'True Positives (TP)': true_positives,
        'False Positives (FP)': false_positives,
        'TP + FP': true_positives + false_positives,
        'Precision': precision
    }

predicted_report_df = pd.DataFrame.from_dict(predicted_report, orient='index')
print("Refined Predicted Region Analysis:\n", predicted_report_df)


final_dataset = pd.read_csv("tweets.csv")


# Preprocess the `final_dataset`
final_dataset['text'] = final_dataset['text'].fillna("").astype(str)

# Identify rows where the 'location' is empty (NaN or null)
empty_location_entries = final_dataset[final_dataset['location'].isnull()]

# Vectorize the text data for empty location entries
empty_location_tfidf = vectorizer.transform(empty_location_entries['text'])

# Add sentiment features to the empty location entries
empty_location_sentiment = empty_location_entries['text'].apply(extract_sentiment_features)
empty_location_sentiment_matrix = sp.csr_matrix(empty_location_sentiment[['polarity', 'subjectivity']].values)

# Generate KMeans cluster labels for the empty location entries
empty_location_clusters = add_cluster_features(
    empty_location_entries, num_clusters, empty_location_tfidf
)
empty_location_clusters_matrix = sp.csr_matrix(empty_location_clusters.reshape(-1, 1))

# Combine TF-IDF, sentiment, and clustering features for prediction
empty_location_features = sp.hstack([
    empty_location_tfidf,
    empty_location_sentiment_matrix,
    empty_location_clusters_matrix
])

# Predict regions for entries with empty locations
empty_location_entries['predicted_region'] = rf_model.predict(empty_location_features)

# Save the empty entries before filling as `empty_tweets.xlsx`
empty_location_entries.to_excel('empty_tweets.xlsx', index=False)

# Fill the missing 'location' column with the predicted regions
filled_dataset = final_dataset.copy()
filled_dataset.loc[filled_dataset['location'].isnull(), 'location'] = empty_location_entries['predicted_region']

# Save the dataset with filled locations as `filled_tweets.xlsx`
filled_dataset.to_excel('filled_tweets.xlsx', index=False)

print("Saved 'empty_tweets.xlsx' and 'filled_tweets.xlsx' with predictions.")
