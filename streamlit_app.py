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
st.write('Market basket analysis is a way to study what items customers often buy together to help businesses make informed decisions about product placement and promotions.')

def user_input_features():
    item = st.selectbox('**Choose Item**', df['item'].unique())
    return item

item = user_input_features()

item_count = df[['transaction_number', 'item', 'amount']]
item_count_pivot = item_count.pivot_table(index='transaction_number', columns='item', values='amount', aggfunc='sum').fillna(0)
item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x > 0 else 0)

support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

metric = 'lift'
min_threshold = 1

rules = association_rules(frequent_items, metric=metric, min_threshold=1)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[['antecedents', 'consequents']].copy()

    data['antecedents'] = data['antecedents'].apply(parse_list)
    data['consequents'] = data['consequents'].apply(parse_list)

    matching_rows = data[data['antecedents'] == item_antecedents]

    if not matching_rows.empty:
        return f' If the customer buys **{item}**, then the customer tends to buy **{matching_rows.iloc[0, 1]}**'
    else:
        return f'No recommendation found for item: **{item}**'

st.markdown("Recommendation Result : ")
st.success(return_item_df(item))

