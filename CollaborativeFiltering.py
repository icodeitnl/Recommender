import pandas as pd
from surprise import Reader, Dataset, KNNBasic

#https://www.kaggle.com/jainaashish/orders-merged
# User-based collaborative filtering using Pearson’s Correlation similarity
parser = Reader()
scores = pd.read_csv('ecom.csv')

scores=scores.drop(columns=['seller_id','order_id','order_status','order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date',
    'order_estimated_delivery_date','customer_zip_code_prefix','customer_city','customer_state','review_comment_title','review_creation_date','review_answer_timestamp',
    'payment_sequential','payment_type','payment_installments','payment_value','order_item_id','price','freight_value','seller_zip_code_prefix','seller_city','seller_state'
    ,'product_name_lenght','customer_id','review_id','product_category_name','review_comment_message','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm'])
print(scores.info())

sort=scores.sort_values(by='product_id', ascending=False).reset_index()
top=sort[:10000]
print(top.head(50))
print(top.info())
top.shape
top[top['product_id'] == 1]

data = Dataset.load_from_df(top[['customer_unique_id', 'product_id', 'review_score']], parser)
sim_options = {'name': 'pearson_baseline','user_based': False,'shrinkage': 0  }
algo = KNNBasic(sim_options=sim_options)
trainset = data.build_full_trainset()
algo.fit(trainset)
#Sample predict for customer_unique_id ('70167d9510c6f54bd29bbffd0f4d9e90')'s rating of product ('ffd4bf4306745865e5692f69bd237893')
print(algo.predict(uid= '70167d9510c6f54bd29bbffd0f4d9e90', iid= 'ffd4bf4306745865e5692f69bd237893', clip=True, verbose=False))
