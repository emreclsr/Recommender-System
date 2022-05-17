#############################################
# HYBRİD RECOMMENDER SYSTEMS
#############################################

# İş Problemi
# ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.


# Veri Seti Hikayesi
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu
# filmlere yapılan derecelendirme puanlarını barındırmaktadır.
#
# movie.csv 3 değişkene sahiptir.
# movieId: Eşsiz film numarası
# title: Film adı
# genres: Tür
#
# rating.csv 4 değişkene sahiptir.
# userId: Eşsiz kullanıcı numarası
# movieId: Eşsiz film numarası
# rating: Kullanıcı tarafından filme verilen puan
# timestamp: Değerlendirme zamanı

#############################################
# USER BASED RECOMMENDATION
#############################################

#############################################
# Görev 1: Veri Hazırlama
#############################################

# Adım 1: movie, rating veri setlerini okutunuz.

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movie = pd.read_csv("week4/recommender_systems/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("week4/recommender_systems/datasets/movie_lens_dataset/rating.csv")

# Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.

df = movie.merge(rating, on='movieId', how="left")

# Adım 3: toplam oy kullanılma sayıı 1000'in altında olan filmlerin isimlerini listede tutunuz ve
# veri setinden çıkartnız.

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index  # mentor sırasında 10000 yaptık 1000 e dönüştür.
common_movies = df[~df["title"].isin(rare_movies)]

# Deneme
# comment_counts = df.groupby(['title']).count()
# common_movies = df[df["title"].isin(comment_counts2[comment_counts2["rating"] > 1000].index)]
# pd.concat([common_movies, common_movies2]).drop_duplicates(keep=False)

# Adım 4: index'te userId'lerin sütunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için
# pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# Adım 5: Yapılan tüm işlemi fonksiyonlaştırınız.


def create_user_movie_df(movie_df_path, rating_df_path, min_comments=1000, columns="title"):
    """
    Creates a pivot table from movie and rating dataframes. Cleans movies with low comment count from dataframe.

    Parameters
    ----------
    movie_df_path: str, path to movie dataframe
    rating_df_path: str, path to rating dataframe
    min_comments: int, minimum number of comments for a movie to be considered, default 1000
    columns: str, column name for pivot table, default "title"

    Returns
    -------
    user_movie_df: pd.DataFrame, pivot table of userId and movieId and rating
    rare_movies: list, list of movies with low comment count
    common_movies: list, list of movies with high comment count


    """
    movie = pd.read_csv(movie_df_path)
    rating = pd.read_csv(rating_df_path)
    df = movie.merge(rating, on='movieId', how="left")

    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= min_comments].index
    common_movies = df[~df["title"].isin(rare_movies)]

    user_movie_df = common_movies.pivot_table(index=["userId"], columns=[columns], values="rating")

    return user_movie_df, list(rare_movies), list(set(common_movies.title)), movie, rating

user_movie_df, rare_movies, \
common_movies, movie, rating = create_user_movie_df("week4/recommender_systems/datasets/movie_lens_dataset/movie.csv",
                                                    "week4/recommender_systems/datasets/movie_lens_dataset/rating.csv",
                                                    1000, "title")


##############################################################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
##############################################################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

random_user = int(pd.Series(user_movie_df.index).sample(1).values)

# Adım 2: Seçilen kullanıcıya air gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız

movies_watched = list(random_user_df.T.dropna().index)

##############################################################################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
##############################################################################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve
# movies_watched_df adında yeni bir dataframe oluşturunuz.

movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan
# user_movie_count adında yeni bir dataframe oluşturunuz.

user_movie_count = movies_watched_df.T.count()

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden
# users_same_movies adında bir liste oluşturunuz.

users_same_movies = list(user_movie_count[user_movie_count >= int(len(movies_watched) * 0.6)].index)

#######################################################################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#######################################################################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların
# id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df = pd.concat([final_df, random_user_df[movies_watched]])

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek
# top_users adında yeni bir dataframe oluşturunuz.

corr_df.index.names = ["UserId_1", "UserId_2"]
corr_df.reset_index(inplace=True)
top_users = corr_df[(corr_df["UserId_1"] == random_user) & (corr_df["corr"] >= 0.65)]

# Adım 4: top_users dataframe’ine rating veri seti ile merge ediniz.

top_users.rename(columns={"UserId_2": "userId"}, inplace=True)
top_users = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

####################################################################################################
# Görev 5:  Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
####################################################################################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında
# yeni bir değişken oluşturunuz.

top_users["weighted_rating"] = top_users["corr"] * top_users["rating"]

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini
# içeren recommendation_df adında yeni bir dataframe oluşturunuz.

recommendation_df = top_users.groupby("movieId").agg({"weighted_rating": "mean"})

# Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve
# weighted rating’e göre sıralayınız.

recommendation = recommendation_df[recommendation_df["weighted_rating"] >= 3.5]. \
                 sort_values(by="weighted_rating", ascending=False)


# Adım 4:  movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.

recommendation.reset_index(inplace=True)
recommendation = recommendation.merge(movie[["movieId", "title"]], how="inner")

print(recommendation["title"].head(5))


#############################################
# ITEM BASED RECOMMENDATION
#############################################

####################################################################################################
# Görev 1:  Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
####################################################################################################

# title üzerinden gitmek hata verdiğinden movieId ler üzerinden yeni bir user_movie_df pivot table oluşturulmuştur.

user_movie_df_movieId, _, _, movie, rating = create_user_movie_df("week4/recommender_systems/datasets/movie_lens_dataset/movie.csv",
                                                         "week4/recommender_systems/datasets/movie_lens_dataset/rating.csv",
                                                         1000, "movieId")

random_user = int(pd.Series(user_movie_df_movieId.index).sample(1).values)

# Adım 1: movie, rating veri setlerini okutunuz.

movie = pd.read_csv("week4/recommender_systems/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("week4/recommender_systems/datasets/movie_lens_dataset/rating.csv")

# Adım 2: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sini alınız.

random_user_rating_five_movie = rating[(rating["userId"] == random_user) & (rating["rating"] == 5)]
# title yerine movieId ile devam edilince gerek kalmadı. Daha sonra sınırlandırılmış veriye film adları eklenecek.
# random_user_rating_five_movie = random_user_rating_five_movie.merge(movie[["movieId", "title"]], how="inner")
random_user_rating_five_movie["timestamp"] = pd.to_datetime(random_user_rating_five_movie["timestamp"])
random_user_rating_five_movie = random_user_rating_five_movie.sort_values(by="timestamp", ascending=False)
random_user_rating_five_last_movie = random_user_rating_five_movie.iloc[0]


# Adım 3: User based recommendation bölümünde oluşan user_movie_df dataframe'ini seçilen film id'sine
# göre filtreleyiniz.

users_rating = user_movie_df_movieId[random_user_rating_five_last_movie.movieId]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

corr_movies = user_movie_df_movieId.corrwith(users_rating).sort_values(ascending=False)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’i öneri olarak veriniz.

recommended_movies = list(corr_movies.iloc[1:6].index)
recommended_movies_name = list(movie[movie["movieId"].isin(recommended_movies)]["title"])








