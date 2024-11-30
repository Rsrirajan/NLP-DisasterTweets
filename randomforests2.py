import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import language_tool_python
import re

# Initialize grammar tool
grammar_tool = language_tool_python.LanguageTool('en-US')

# Function to sanitize and count grammar errors
def count_grammar_errors_safe(text):
    try:
        # Ensure the text is a string
        if not isinstance(text, str) or not text.strip():
            return 0
        # Sanitize text: remove URLs and other problematic patterns
        sanitized_text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
        sanitized_text = re.sub(r"\s+", " ", sanitized_text).strip()  # Normalize whitespace
        if len(sanitized_text) < 5:  # Skip very short texts
            return 0
        matches = grammar_tool.check(sanitized_text)
        return len(matches)
    except Exception as e:
        print(f"Error processing text: {text[:50]}...\n{e}")  # Print partial text for debugging
        return 0

# Load the dataset
tweets_df = pd.read_csv("tweets_country_region.csv")

# Drop rows where 'region' is NaN
tweets_df = tweets_df.dropna(subset=['region'])

# Preprocess text (optional: add more cleaning steps if needed)
tweets_df['text'] = tweets_df['text'].fillna("").astype(str)

# Count grammar errors
tweets_df['grammar_errors'] = tweets_df['text'].apply(count_grammar_errors_safe)

# Split the data into stratified training and testing sets
train_df, test_df = train_test_split(tweets_df, test_size=0.2, random_state=42, stratify=tweets_df['region'])

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=train_df['region'].unique(),
    y=train_df['region']
)
class_weight_dict = dict(zip(train_df['region'].unique(), class_weights))

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), max_df=0.85, min_df=5)
X_train_text = vectorizer.fit_transform(train_df['text'])
X_test_text = vectorizer.transform(test_df['text'])

# Include grammar errors as a feature
X_train = pd.concat([pd.DataFrame(X_train_text.toarray()), train_df[['grammar_errors']].reset_index(drop=True)], axis=1)
X_test = pd.concat([pd.DataFrame(X_test_text.toarray()), test_df[['grammar_errors']].reset_index(drop=True)], axis=1)

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
    # True Positives (TP): Correct predictions for this class
    true_positives = ((test_df['region'] == cls) & (test_df['predicted_region'] == cls)).sum()
    
    # False Positives (FP): Predictions that were assigned to this class but do not actually belong
    false_positives = ((test_df['region'] != cls) & (test_df['predicted_region'] == cls)).sum()
    
    # Precision = TP / (TP + FP)
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
