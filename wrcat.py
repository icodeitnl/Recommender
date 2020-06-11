import pandas as pd
import numpy as np
import plotly.graph_objs as go

#https://www.kaggle.com/jainaashish/orders-merged
import warnings
warnings.simplefilter('ignore')

data=pd.read_csv("ecom.csv", low_memory=False)
print(data.info())

# Recommender Weigted Average and Categories
data=data.drop(columns=['seller_id','order_id','order_status','order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date',
	'order_estimated_delivery_date','customer_id','customer_zip_code_prefix','customer_city','customer_state','review_comment_title','review_creation_date','review_answer_timestamp',
	'payment_sequential','payment_type','payment_installments','payment_value','order_item_id','price','freight_value','seller_zip_code_prefix','seller_city','seller_state'
	,'product_name_lenght','review_comment_message','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm'])

print(data.head(5))
print(data.sort_values(by='product_id', ascending=False))

#Average review for the product
total_score=data.groupby(['product_id'])['review_score'].sum().reset_index().sort_values(by='review_score', ascending=False)
total_rate=total_score.rename({'review_score':'total_score'}, axis=1)
print(total_rate.head(15))

mean_score=data.groupby(['product_id'])['review_score'].mean().reset_index().sort_values(by='review_score', ascending=False)
mean_rate=mean_score.rename({'review_score':'average_score'}, axis=1)
print(mean_rate.head(5))
data_score=pd.merge(data,total_rate[['product_id', 'total_score']], on = 'product_id', how = 'left')
df=pd.merge(data_score,mean_rate[['product_id', 'average_score']],on = 'product_id', how = 'left')

print(df.info())
# Recommender weighted average
R=mean_rate['average_score'] #average for the product as a number from 0 to 5
v=total_rate['total_score'] #number of votes for the item
C=mean_rate['average_score'].mean()#mean for the whole dataset
m=total_rate['total_score'].quantile(0.70) #minimum votes required to be listed in Top Rated

data['weighted_average_rate']=  (v / (v+m)) * R + (m / (v+m)) * C
print(data.head(5))

data=data.sort_values('weighted_average_rate',ascending=False)
print(data.head(10))
fig = go.Figure(
    data=go.Bar(
    	x= df.query("total_score>=1230")['total_score'], 
    	y= df.query("total_score>=1230")['product_id'],
        marker={'color': df.query("total_score>=1230")['total_score'],
        'colorscale': 'earth'},
        orientation='h'),
    layout= go.Layout(
        xaxis={"title":'Total Review Score'},
        yaxis={"title":'Product ID'},
        title='Top Products')
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

#Data Vizualisation

fig = go.Figure(
    data=go.Bar(
    	x= data.query("weighted_average_rate>=4.85")['weighted_average_rate'], 
    	y= data.query("weighted_average_rate>=4.85")['product_category_name'],
        marker={'color': data.query("weighted_average_rate>=4.85")['weighted_average_rate'],
        'colorscale': 'turbid'},
        orientation='h'),
    layout= go.Layout(
        xaxis={"title":'Weighted Average Rate'},
        yaxis={"title":'Product Category'},
        title='Top in Category')
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

#Recommender weighted average and category

def rec (product_category_name):
    pr = df[df['product_category_name'] == product_category_name]
    R=mean_rate['average_score'] 
    v=total_rate['total_score'] 
    C=mean_rate['average_score'].mean()
    m=total_rate['total_score'].quantile(0.70)
    bla=pr[['product_id','total_score','average_score','product_category_name']]
    bla['weighted_average_rate']=  (v / (v+m)) * R + (m / (v+m)) * C
    bla=bla.sort_values('weighted_average_rate',ascending=False)
    return bla
#Passing value to the function
print(rec('eletronicos').head(15))
print(rec('eletronicos').info())

