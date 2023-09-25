import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules,apriori


st.set_page_config(
    page_title="COFFEE SHOP MBA",
    page_icon=":coffee:"
)
# Load Dataset
df = pd.read_excel('Coffe Shop Sales.xlsx')
st.title(':coffee: Coffee Shops Market Basket Analysis :croissant:')

# def get_data(item =''):
#     data = df.copy()
#     filtered = data.loc[
#         (data['item'].str.contains(item)) 
#     ]
#     return filtered if filtered.shape[0] else 'No Result!'

def user_input_features():
    item = st.selectbox('**Choose Item**',
                         df['item'].unique())
    return  item

item = user_input_features()

# data = get_data(item)

def encode(x):
    if x <= 0:
        return 0
    else:
        return 1

# if type(data) != type('No Result!'):
item_count = df[['transaction_number', 'item', 'amount']]
item_count_pivot = item_count.pivot_table(index='transaction_number',columns ='item',values = 'amount',aggfunc='sum').fillna(0)
item_count_pivot = item_count_pivot.applymap(encode)

support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

metric = 'lift'
min_threshold = 1

rules = association_rules(frequent_items, metric=metric, min_threshold=1)[['antecedents','consequents','support','confidence','lift']]
rules.sort_values('confidence', ascending= False, inplace=True)
    
def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) >1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[['antecedents','consequents']].copy()
    
    data['antecedents'] = data['antecedents'].apply(parse_list)
    data['consequents'] = data['consequents'].apply(parse_list)
    
    return list(data.loc[data['antecedents'] == item_antecedents].iloc[0,:])

# if type(data) != type('No Result!'):
st.markdown("Recommendation Result : ")
st.success(f' If the customer buy **{item}**, then the customer tend to buy **{return_item_df(item)[1]}**')
