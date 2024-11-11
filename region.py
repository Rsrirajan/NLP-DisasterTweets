import pandas as pd

# Define file paths
tweets_final_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_final.csv"
continents_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\continents.csv"
output_csv_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_country_region.csv"
output_excel_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_country_region.xlsx"

# Import data
tweets_final_df = pd.read_csv(tweets_final_path)
continents_df = pd.read_csv(continents_path)

# Ensure column names are consistent for merging
continents_df.columns = [col.lower().replace(" ", "_") for col in continents_df.columns]

# Merge based on 'LOCATION' in tweets_final_df and 'country' in continents_df
tweets_country_region_df = tweets_final_df.merge(
    continents_df[['name', 'region', 'sub-region']],
    left_on='LOCATION', right_on='name',
    how='left'
)

# Drop the extra 'country' column resulting from the merge
tweets_country_region_df = tweets_country_region_df.drop(columns=['name'])
tweets_country_region_df.rename(columns={'LOCATION': 'country'})
# Save to CSV and Excel
tweets_country_region_df.to_csv(output_csv_path, index=False)
tweets_country_region_df.to_excel(output_excel_path, index=False)

print("Data with regions and sub-regions saved successfully.")
