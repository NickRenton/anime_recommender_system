import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.sparse import csr_matrix
from IPython.display import clear_output

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
animes = pd.read_csv("dataset/anime.csv")
ratings = pd.read_csv("dataset/rating.csv")


def create_matrix(df):
    users_count = len(df['user_id'].unique())
    animes_count = len(df['anime_id'].unique())

    # Map Ids to indices
    users_mapper = dict(zip(np.unique(df["user_id"]), list(range(users_count))))
    animes_mapper = dict(zip(np.unique(df["anime_id"]), list(range(animes_count))))

    # Map indices to IDs
    users_inv_mapper = dict(zip(list(range(users_count)), np.unique(df["user_id"])))
    animes_inv_mapper = dict(zip(list(range(animes_count)), np.unique(df["anime_id"])))

    user_index = [users_mapper[u] for u in df['user_id']]
    anime_index = [animes_mapper[a] for a in df['anime_id']]

    matrix = csr_matrix((df["rating"], (anime_index, user_index)), shape=(animes_count, users_count))

    return matrix, users_mapper, animes_mapper, users_inv_mapper, animes_inv_mapper


X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)


def find_similar_movies(anime_id, matrix, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[anime_id]
    movie_vec = matrix[movie_ind]
    k += 1
    knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    knn.fit(matrix)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = knn.kneighbors(movie_vec, return_distance=show_distance)
    for index in range(0, k):
        neighbour_ids.append(movie_inv_mapper[neighbour.item(index)])
    neighbour_ids.pop(0)
    return neighbour_ids


def recommend_movies_for_user(user_id, matrix, k=10):
    df1 = ratings[ratings['user_id'] == user_id]

    if df1.empty:
        print(f"User with ID {user_id} does not exist.")
        return

    anime_id = df1[df1['rating'] == max(df1['rating'])]['anime_id'].iloc[0]
    anime_titles = dict(zip(animes['anime_id'], animes['name']))

    similar_anime_ids = find_similar_movies(movie_id, matrix, k)
    anime_title = anime_titles.get(anime_id, "Movie not found")

    if anime_title == "Movie not found":
        print(f"Movie with ID {anime_id} not found.")
        return

    print(f"Since you watched {anime_title}, you might also like:")
    for anime in similar_anime_ids:
        print(movie_titles.get(anime, "Movie not found"))


def stats():
    n_ratings = len(ratings)
    n_animes = len(ratings['anime_id'].unique())
    n_users = len(ratings['user_id'].unique())

    print(f"Number of ratings: {n_ratings}")
    print(f"Number of unique anime_id's: {n_animes}")
    print(f"Number of unique users: {n_users}")
    print(f"Average ratings per user: {round(n_ratings / n_users, 2)}")
    print(f"Average ratings per movie: {round(n_ratings / n_animes, 2)}")

    user_freq = ratings[['user_id', 'anime_id']].groupby(
        'user_id').count().reset_index()
    user_freq.columns = ['user_id', 'n_ratings']
    print(user_freq.head())

    # PLEASE UNCOMMENT ONE BY ONE to build an image

    # Top members by anime
    sns.barplot(data=animes.sort_values('members', ascending=False).head(10), x="name", y="members")
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 1000000, step=100000))
    plt.savefig('top_members.png', bbox_inches='tight')
    clear_output()

    # genres
    # data = animes[['name', 'type']].groupby("type").count()
    # # declaring exploding pie
    # explode = [0, 0, 0, 0.1, 0, 0]
    # # define Seaborn color palette to use
    # palette_color = sns.color_palette('dark')
    # # plotting data on chart
    # plt.pie(data.to_dict()['name'].values(), labels=data.to_dict()['name'].keys(), colors=palette_color,
    #         explode=explode, autopct='%.0f%%')
    # plt.savefig('genres.png', bbox_inches='tight')
    # clear_output()

    # members
    # data = ratings[['rating', 'anime_id']].groupby("rating").count().reset_index()
    # data.columns = ['rating', 'count']
    # data = data.drop(data[data['rating'] == -1].index)
    # sns.barplot(data=data, x="rating", y="count")
    # plt.yticks(np.arange(0, 1800000, step=100000))
    # plt.savefig('members.png', bbox_inches='tight')


if __name__ == '__main__':

    # draw statistics
    stats()

    # find similar anime
    movie_titles = dict(zip(animes['anime_id'], animes['name']))
    movie_id = 44

    similar_ids = find_similar_movies(movie_id, X, k=10)
    movie_title = movie_titles[movie_id]

    print(f"Since you watched {movie_title}")
    for i in similar_ids:
        print(movie_titles[i])
