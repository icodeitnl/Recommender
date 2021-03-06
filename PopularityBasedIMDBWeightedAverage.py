import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler


# Data Exploration
# Import Data

movies=pd.read_csv("IMDb_movies.csv")

print(movies.info())
print(movies.sample(5))
print(movies.describe())
print(pd.isnull(movies))

ratings=pd.read_csv("IMDb_ratings.csv")

print(ratings.info())
print(ratings.sample(5))
print(ratings.describe())
print(pd.isnull(ratings))

# Merge movie titles to ratings

mr=pd.merge(movies,ratings,on='imdb_title_id')
mr=mr.drop(columns=['votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','allgenders_0age_avg_vote','allgenders_0age_votes','allgenders_18age_avg_vote',
	'allgenders_18age_votes','allgenders_45age_avg_vote','allgenders_45age_votes','males_0age_avg_vote','males_0age_votes','males_18age_avg_vote','males_18age_votes','males_45age_avg_vote',
	'males_45age_votes','females_0age_avg_vote','females_0age_votes','females_0age_votes','females_18age_avg_vote','females_18age_votes','females_45age_avg_vote','females_45age_votes',
	'us_voters_rating','us_voters_votes','non_us_voters_rating','non_us_voters_votes','original_title','year','date_published','duration','country','director','writer','production_company','usa_gross_income','metascore','worlwide_gross_income'])
print(mr.sample(5))
print(mr.info())
mrtrain=mr.drop(columns=['language','genre','imdb_title_id','actors','description','budget','allgenders_30age_avg_vote','allgenders_30age_votes','males_allages_avg_vote','males_allages_votes',
	'males_30age_avg_vote','males_30age_votes','females_allages_avg_vote','females_allages_votes','females_30age_avg_vote','females_30age_votes','top1000_voters_rating','top1000_voters_votes'])
print(mrtrain.sample(5))
print(mrtrain.info())
#Current Rating System at IMDB: W=((R*v)+ (C*m))/(v+m))


R=mrtrain['avg_vote'] #average for the movie as a number from 0 to 10(mean)=(Rating)
v=mrtrain['total_votes'] #number of votes for the movie=(votes)
#C=mrtrain['avg_vote'].mean() #mean for the whole dataset(currently 6.9) 
#print(C) # as C calculated is 5.926, we change the number to 6.9 to be more aligned with IMDB
C=6.9
#m=mrtrain['total_votes'].quantile(0.70) #minimum votes required to be listed in the Top 250 (Currently 3000)
#print(m) #as m calculated is 1334.0  we change the number to 3000
m=3000
mrtrain['calculated_weighted_average_vote']=  (v / (v+m)) * R + (m / (v+m)) * C
print(mrtrain.head(5))
mrtrain=mrtrain.loc[:, mrtrain.columns.intersection(['calculated_weighted_average_vote','weighted_average_vote','title','reviews_from_users'])]
print(mrtrain.head(5))
mrtrain=mrtrain.sort_values('calculated_weighted_average_vote',ascending=False)
print(mrtrain.head(10))
# plot the top 10
fig = go.Figure(
    data=go.Bar(
    	x= mrtrain.query("calculated_weighted_average_vote>=8.88")['calculated_weighted_average_vote'], 
    	y= mrtrain.query("calculated_weighted_average_vote>=8.88")['title'],
        marker={'color': mrtrain.query("calculated_weighted_average_vote>=8.88")['calculated_weighted_average_vote'],
        'colorscale': 'turbid'},
        orientation='h'),
    layout= go.Layout(
        xaxis={"title":'Movie Score'},
        yaxis={"title":'Movie Title'},
        title='Top 10 weighted average')
    )
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro") 
)
fig.show()

mr=mr.loc[:, mr.columns.intersection(['votes', 'title'])]
mr=mr.sort_values('votes',ascending=False)

fig = go.Figure(
    data=go.Bar(
    	x= mr.query("votes>=1421494")['votes'],  
    	y= mr.query("votes>=1421494")['title'],
        marker={'color': mr.query("votes>=1421494")['votes'],
        'colorscale': 'earth'},
        orientation='h'),
    layout= go.Layout(
        xaxis={"title":'Movie Score'},
        yaxis={"title":'Movie Title'},
        title='Top 10 most voted')
    )
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro") 
)
fig.show()

# Top scaled weighted average and most reviewed 

scaling=MinMaxScaler()
data=pd.merge(mr,mrtrain,on='title')
mr_scaled=scaling.fit_transform(data[['calculated_weighted_average_vote','reviews_from_users']])
mr_normalized=pd.DataFrame(mr_scaled,columns=['calculated_weighted_average_vote','reviews_from_users'])
print(mr_normalized.head(5))
mrtrain[['normalized_calculated_weighted_average_vote','normalized_reviews_from_users']]=mr_normalized
print(mrtrain.head(5))
