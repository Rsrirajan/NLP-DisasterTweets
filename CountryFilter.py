import pandas as pd
from geotext import GeoText
import matplotlib.pyplot as plt

tweets_df = pd.read_csv("tweets.csv")
world_cities_df = pd.read_csv("worldcities.csv")
us_cities_df = pd.read_csv("uscities.csv")
state_names_df = pd.read_csv("state_names.csv")

tweets_df['location'] = tweets_df['location'].fillna('').astype(str)

tweets_df['country'] = ''

us_states = state_names_df['State'].str.lower().tolist()
us_alpha_codes = state_names_df['Alpha'].str.lower().tolist()

def assign_country_from_terms(location, current_country):
    if pd.notna(current_country) and current_country != '':
        return current_country

    location_lower = location.lower()
    location_words = location_lower.split()
    
    if any(word in us_states or word in us_alpha_codes for word in location_words):
        return "United States"
    
    if "usa" in location_lower:
        return "United States"
    if "suisse" in location_lower:
        return "Switzerland"
    if "bharat" in location_lower:
        return "India"
    if "england" in location_lower:
        return "United Kingdom"
    
    return None

tweets_df['country'] = tweets_df.apply(
    lambda row: assign_country_from_terms(row['location'], row['country']), axis=1
)

def is_valid_location(location):
    places = GeoText(location)
    return bool(places.countries or places.cities)

tweets_df['is_valid_location'] = tweets_df['location'].apply(is_valid_location)
filtered_tweets_df = tweets_df[tweets_df['is_valid_location']].drop(columns='is_valid_location')
tweets_df_country = filtered_tweets_df.copy()

def assign_country_direct(location, current_country):
    if pd.notna(current_country) and current_country != '':
        return current_country

    places = GeoText(location)
    if places.countries:
        return places.countries[0]
    return None

tweets_df_country['country'] = tweets_df_country.apply(
    lambda row: assign_country_direct(row['location'], row['country']), axis=1
)

def match_city_to_country(location, current_country):
    if pd.notna(current_country) and current_country != '':
        return current_country

    places = GeoText(location)
    if places.cities:
        city = places.cities[0]
        matching_country = world_cities_df.loc[world_cities_df['city'].str.lower() == city.lower(), 'country']
        if not matching_country.empty:
            return matching_country.values[0]
    return None

tweets_df_country['country'] = tweets_df_country.apply(
    lambda row: match_city_to_country(row['location'], row['country']), axis=1
)

def match_us_city_with_state(location, current_country):
    if pd.notna(current_country) and current_country != '':
        return current_country

    places = GeoText(location)
    for us_city, state in zip(us_cities_df['city'], us_cities_df['state_name']):
        if f"{state}, {us_city}".lower() == location.lower():
            return "United States"

    if places.cities:
        city = places.cities[0]
        if city.lower() in map(str.lower, us_cities_df['city'].unique()):
            return "United States"
    return None

tweets_df_country['country'] = tweets_df_country.apply(
    lambda row: match_us_city_with_state(row['location'], row['country']), axis=1
)

def check_eu(location, current_country):
    if pd.notna(current_country) and current_country != '':
        return current_country
    if "eu" in location.lower():
        return "Europe"
    return None

tweets_df_country['country'] = tweets_df_country.apply(
    lambda row: check_eu(row['location'], row['country']), axis=1
)

output_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_with_country.xlsx"
tweets_df_country.to_excel(output_path, index=False)

tweets_with_country_final = pd.read_excel(r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_with_country_final.xlsx")

tweets_final = tweets_df_country.copy()
tweets_final['LOCATION'] = tweets_with_country_final['country'].values
tweets_final = tweets_final.drop(columns=['location', 'country'])

output_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_final.xlsx"
tweets_final.to_excel(output_path, index=False)


output_path = r"C:\Users\User\Desktop\432 p\NLP-DisasterTweets\tweets_final.csv"
tweets_final.to_csv(output_path, index=False)

print(tweets_final.head())

