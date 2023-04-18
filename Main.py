import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle

credit = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")
# merging credit dataset into Movies
movies = movies.merge(credit, on='title')

# Data Cleaning and removed irrelevant columns
movies = movies[['id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
# checking null values in dataset
movies.isnull().sum()

# dropping the null value if have
movies = movies.loc[movies.dropna().index]
# checking the duplicated values if have
movies.duplicated().sum()

# A function that will convert columns string data into list
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# deploying that function on 'Genres ' column
movies['genres'] = movies['genres'].apply(convert)

# deploying function on 'Keywords' column
movies['keywords'] = movies['keywords'].apply(convert)

# extracting 3 leading role from Data set


def get_hero(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(get_hero)


# extract only "Director" of movie
def get_crew(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# get Director name
def get_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L


movies['crew'] = movies['crew'].apply(get_director)



movies['overview'] = movies['overview'].apply(lambda x: x.split())

# all coulmns have some spaces and should be removed.
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['overview'] = movies['overview'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# merger all columns into tags column
movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']

new_movies_df = movies[['id', 'title', 'tags']].copy()
new_movies_df['tags'] = new_movies_df['tags'].apply(lambda x: " ".join(x))

ps = PorterStemmer()


def stemming(text):
    t = []
    for i in text.split():
        t.append(ps.stem(i))
    return " ".join(t)


new_movies_df['tags'] = new_movies_df['tags'].apply(stemming)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_movies_df['tags']).toarray()

similarity = cosine_similarity(vectors)


# a function to get recommendation
def recommend(movie):
    movie = movie.lower()
    movie_index = new_movies_df[new_movies_df['title'].str.lower() == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        movie_title = new_movies_df.iloc[i[0]].title
        print(movie_title.capitalize())


recommend('batman')




pickle.dump(new_movies_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(new_movies_df.to_dict(), open('movies_dict.pkl', 'wb'))
