
import pandas as pd
import plotly.graph_objs as go
#term frequency–inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

movies=pd.read_csv('imdb-movies.csv')
print(movies.info())
print(movies.sample(5))
print(pd.isnull(movies))

mr=movies.drop(columns=['id','imdb_id','budget','revenue','cast','homepage','director','tagline','runtime',
	'production_companies','release_date','release_year','budget_adj','revenue_adj'])
print(movies.head(5)['overview'])

vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

mr['overview'] = mr['overview'].fillna('')
# Keyword-based Vector Space Model
vmtrx = vectorizer.fit_transform(mr['overview'])
print(vmtrx.shape)
sg = sigmoid_kernel(vmtrx, vmtrx)
indices = pd.Series(mr.index, index=mr['original_title']).drop_duplicates()


def recommendations (title, sg=sg):
    ind = indices[title]
    score= list(enumerate(sg[ind]))
    #lambda signifies an anonymous function. In this case, this function takes the single argument x and returns x[1] (i.e. the item at index 1 in x).
    score = sorted (score, key=lambda x: x[1], reverse=True)
    score = score[1:6]
    result = [indx[0] for indx in score]

    return mr['original_title'].iloc[result]
print("Recommended for God Father:")
print(recommendations('The Godfather'))
print("Recommended for One Hundred and One Dalmatians:")
print(recommendations('One Hundred and One Dalmatians'))
print("Recommended for The Lord of the Rings:")

print(recommendations('The Lord of the Rings'))
print("Recommended for Indiana Jones:")
print(recommendations('Raiders of the Lost Ark'))
print("Recommended for The Texas Chainsaw Massacre:")
print(recommendations('The Texas Chainsaw Massacre'))
print("Recommended for Thunderball:")
print(recommendations('Thunderball'))
