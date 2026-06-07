# Type of ML problem - Unsupervised
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Import Libraries

import numpy as np #Conducts Numerical Computing Operations

import pandas as pd # Manipulates data as Data Frames
pd.set_option('display.max_columns', 100)# Ensures wide Data Frames aren't truncated when displayed


from matplotlib import pyplot as plt # Foundational plotting library - used to create and customise and render charts

import seaborn as sns # Built on top of matplotlib

import warnings 
warnings.filterwarnings("ignore") #supresses non-critical warnings - removes clutter from output without changing results

from sklearn.preprocessing import StandardScaler # all numerical features have mean of 0 and standard deviation of 1

    # Why do we need this? - Both PCA and KMeans and as both are sensitive to scale of input

from sklearn.decomposition import PCA 
    # Principle Component Analysis - compresses customer features into fewer components to caputure most of  variance
    # Effect on KMeans - makes Kmeans clustering more effective

from sklearn.cluster import KMeans # Groups customers into k clustersbased on similariirties across features (ie: spending, transaction frequency, etc)

from sklearn.metrics import adjusted_rand_score # Evaluates how similar 2 sets of cluster labels are - useful fro comparing clustering results

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2: Load Dataset - international online transactions data from CSV

df_international = pd.read_csv('int_online_tx(2).csv', encoding='latin-1')
df = df_international[df_international['Country'] != 'United Kingdom']
# as we are looking mainly at the company's international sales, we are excluding sales made in the UK 
# why are we only looking at international sales

#df_international.shape - shows (Rows, Columns) in original data set 

df.shape #shows (Rows, Columns) in data set without Uk sales

df.head(10) # First 10 rows of data

#Distribute transactions by country
plt.figure(figsize=(9, 10))
sns.countplot(y='Country', data=df)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3: Clean data at Transaction Level
print("-----------------------------------------------")
print("Missing values:\n", df.isnull().sum()) #print number of missing observations per feature

# When run, 'CustomerID' has missing observations
    # What should be done with the data in this scenario?
        # AIM: to cluster customers to provide a more tailored experience
        # so keeping transactions without a CustomerID is pointless
        # DECISION: Discard observations missing CustomerID

df = df[df.CustomerID.notnull()] # removing observations without customer ID
df['CustomerID'] = df.CustomerID.astype(int) # convert into integer
print("-----------------------------------------------")
print("First 5 CustomerIDs:\n", df.CustomerID.head()) # print first 5 customer IDs

# Create Sales Interaction Feature
df['Sales'] = df.Quantity * df.UnitPrice
print("-----------------------------------------------")
print("First 5 Sales values:\n", df.Sales.head())

# Save Cleaned transaction data to new csv file
df.to_csv('cleaned_transactions.csv', index=None)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4: Adapting for Customer-Level Data

# Converting data from one row per product per invoice, resulting in numerous rows per customer,
# to one row per customer, so customer ID is seen once in the data and all purchases are centralised

# For Invoice Data
invoice_data = df.groupby('CustomerID').agg(total_transactions=('InvoiceNo', 'nunique')) # Aggregate Invoice Data
print("-----------------------------------------------")
print("Invoice data (first 5):\n", invoice_data.head()) # print invoice data for first 5 customers
print("-----------------------------------------------")

# For Product Data
product_data = df.groupby('CustomerID').agg(total_products=('StockCode', 'count'), total_unique_products=('StockCode', 'nunique')) # Aggregate Product Data
print("Product data (first 5):\n", product_data.head()) # print product data for first 5 customers
print("-----------------------------------------------")

# Aggregate sales data from Low scope to High scope - Roll up from Transaction Level to Customer-Level
sales_data = df.groupby('CustomerID').agg(total_sales=('Sales', 'sum'),avg_product_value=('Sales', 'mean')) # combining data so that customers repeated have a combined invoice and so only appear once 
print("Sales data (first 5):\n", sales_data.head()) # print sales data for first 5 customers
print("-----------------------------------------------")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 5: Aggregate for Cart-level data

# Converting data from one row per product per invoice to one row per shopping session, so there are still repeats in Customer ID but not as many as originally

# Why is this useful? 
    # This preserves per-session structure but collapses individual products within each session into a single cart value total
    # Rolling this up to customer level gives us valuable insight into spending behaviour per session rather than just overall spend

# Value to the retailer?
    # Let Customer A: 2 sessions x £200 each - shows occasional customer but bulk buys when they do come
        # Customer B: 20 sessions x £20 each -  shows frequent customer with low spend in each session

    # Without the Cart Level Data, there is no way to differentiate Customer A and B as both have the same total_sales of £400.
    # As a result both customer would be treated the same despite having completely different shopping personalities
    # This information lets the clustering algorithm correctly spilt them to tailor marketing and service

cart_data = df.groupby(['CustomerID', 'InvoiceNo']).agg(cart_value=('Sales', 'sum')) # Aggregate cart-level data (i.e. invoice-level)
print("Cart data (first 20):\n", cart_data.head(20))
print("-----------------------------------------------")


cart_data.reset_index(inplace=True) # Reset index so that Customer ID and Invoice No, returns to being columns from being an index
print("Cart data after reset_index (first 10):\n", cart_data.head(10)) # print first 10 of cart-level data, now that index has been reset
print("-----------------------------------------------")

# Aggregate cart data at customer-level
agg_cart_data = cart_data.groupby('CustomerID').agg(avg_cart_value=('cart_value', 'mean'), min_cart_value=('cart_value', 'min'), max_cart_value=('cart_value', 'max'))
print("Aggregated cart data (first 5):\n", agg_cart_data.head())
print("-----------------------------------------------")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6: Join Customer-level data frames

customer_df = invoice_data.join([product_data, sales_data, agg_cart_data]) # Join together customer-level data (ie: product data, sales data and cart data
print("Customer-level data (first 5):\n", customer_df.head())
print("-----------------------------------------------")

customer_df.to_csv('analytical_base_table.csv') # Save analytical base table as csv, keeping CustomerID as index

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 7: Set up df for Clustering

# What should be done?
    # Add products bought by each customer as features within df
    # Problem: to record what a customer has bought, with 'pd.get_dummies(df.StockCode)', we use a one column per product, resulting thousands of new columns
        # reflects the issue with dimensionality - When number of features vastly exceed meaningful signal, clustering algorithm becomes inefficient
    # Solution: Use Thresholding or PCA methods to cut down number of columns so that data is actually useful, eliminating noise

df = pd.read_csv('cleaned_transactions.csv') # Read cleaned_transactions.csv

item_dummies = pd.get_dummies(df.StockCode) # create the vector of StockCode
print("Item dummies (first 5):\n", item_dummies.head())
print("-----------------------------------------------")

item_dummies['CustomerID'] = df.CustomerID # Add CustomerID to item_dummies
print("Item dummies with CustomerID (first 5):\n", item_dummies.head())
print("-----------------------------------------------")

item_data = item_dummies.groupby('CustomerID').sum() # Create item_data by aggregating at customer level
print("Item data (first 5):\n", item_data.head())
print("-----------------------------------------------")
print("Total number of each item purchased:\n", item_data.sum()) # Total times each item was purchased
item_data.to_csv('item_data.csv') # Save item_data.csv (keeping CustomerID as index)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 8.1 - Method 1: Thresholding

print("Most popular 120 items:\n", item_data.sum().sort_values().tail(120)) # Display most popular 120 items
top_20_items = item_data.sum().sort_values().tail(20).index # Get list of StockCodes for the top 20 most popular items
print("Top items index:\n", top_20_items)

top_20_item_data = item_data[top_20_items] # Keep only features for top 20 items
print("Shape of top item data:", top_20_item_data.shape)
print(top_20_item_data.head())
top_20_item_data.to_csv('threshold_item_data.csv') # Save threshold_item_data.csv

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 8.2 - Method 2: PCA

item_data = pd.read_csv('item_data.csv', index_col=0) # Read item_data.csv w/ CustomerID as index
print("Item data shape:", item_data.shape)

scaler = StandardScaler()
item_data_scaled = scaler.fit_transform(item_data)
print("Scaled item data (first 5 rows):\n", item_data_scaled[:5]) # Initialize and fit StandardScaler

pca = PCA()
pca.fit(item_data_scaled) # Initialise and fit PCA (keep all components first)

PC_items = pca.transform(item_data_scaled)
print("PC items (first 5 rows):\n", PC_items[:5]) # Generate PC features

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_) # Cumulative explained variance

plt.grid()
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.show() # Plot cumulative explained variance

print("Variance captured by first 300 components:", cumulative_explained_variance[300]) # Variance captured with first 300 components

pca = PCA(n_components=300)
PC_items = pca.fit_transform(item_data_scaled)
print("PC items shape:", PC_items.shape) # PCA with 300 components

items_pca = pd.DataFrame(PC_items)
items_pca.columns = ['PC{}'.format(i + 1) for i in range(PC_items.shape[1])]
items_pca.index = item_data.index
print("PCA item data (first 5):\n", items_pca.head()) # Put PC_items into a df

items_pca.to_csv('pca_item_data.csv') # Save pca_item_data.csv w/ CustomerID as index

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 9 - KMeans CLustering

# Import analytical base table
base_df = pd.read_csv('analytical_base_table.csv', index_col=0)

# Import thresholded item features
threshold_item_data = pd.read_csv('threshold_item_data.csv', index_col=0)

# Import PCA item features
pca_item_data = pd.read_csv('pca_item_data.csv', index_col=0)

# Print shape of each dataframe
print("base_df shape:", base_df.shape)
print("threshold_item_data shape:", threshold_item_data.shape)
print("pca_item_data shape:", pca_item_data.shape)

# Join base_df with threshold_item_data
threshold_df = base_df.join(threshold_item_data)
print("Threshold DF (first 5):\n", threshold_df.head())

# Join base_df with pca_item_data
pca_df = base_df.join(pca_item_data)
print("PCA DF (first 5):\n", pca_df.head())

# Scale both dataframes
t_scaler = StandardScaler()
p_scaler = StandardScaler()

threshold_df_scaled = t_scaler.fit_transform(threshold_df)
pca_df_scaled = p_scaler.fit_transform(pca_df)

# KMeans with threshold_df
t_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)
t_kmeans.fit(threshold_df_scaled)
threshold_df['cluster'] = t_kmeans.fit_predict(threshold_df_scaled)

# Scatterplot coloured by cluster (threshold)
sns.lmplot(x='total_sales', y='avg_cart_value', hue='cluster', data=threshold_df, fit_reg=False)
plt.title('KMeans Clusters - Threshold Features')
plt.show()

# KMeans with pca_df
p_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)
p_kmeans.fit(pca_df_scaled)
pca_df['cluster'] = p_kmeans.fit_predict(pca_df_scaled)

# Scatterplot coloured by cluster (PCA)
sns.lmplot(x='total_sales', y='avg_cart_value', hue='cluster', data=pca_df, fit_reg=False)
plt.title('KMeans Clusters - PCA Features')
plt.show()

# Similarity between pca_df clusters and threshold_df clusters
ari_score = adjusted_rand_score(pca_df.cluster, threshold_df.cluster)
print("Adjusted Rand Score (PCA vs Threshold clusters):", ari_score)
