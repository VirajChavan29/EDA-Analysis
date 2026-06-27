#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[27]:


#load the dataset
df= pd.read_csv("googleplaystore.csv",
                  engine="python",
                  on_bad_lines="skip")
print(df.shape)


# In[28]:


df


# In[29]:


print(df)


# In[30]:


df.isnull().sum()


# In[31]:


df = df.drop_duplicates()
print("Duplicates removed successfully.")


# In[32]:


df.describe()


# In[33]:


df.dtypes


# In[34]:


#Histogram
df["Rating"].hist()

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.show()


# In[35]:


#Box Plot
plt.boxplot(df["Rating"].dropna())

plt.title("Box Plot of Rating")

plt.show()


# In[36]:


#Correlation
numeric_df = df.select_dtypes(include=['number'])

numeric_df.corr()


# In[37]:


#Correlation Heatmap
import matplotlib.pyplot as plt

corr = numeric_df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.show()


# In[38]:


#Scatter Plot
plt.scatter(df["Reviews"], df["Rating"])

plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Reviews vs Rating")

plt.show()


# In[39]:


#Top 10 App Categories
df["Category"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")

plt.show()


# In[40]:


#Pie Chart
df["Type"].value_counts().plot(kind="pie", autopct="%1.1f%%")

plt.ylabel("")
plt.title("Free vs Paid Apps")

plt.show()


# In[45]:


print("EDA Summary")
print("--------------------")

print("Rows and Columns :", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nTop Categories")
print(df["Category"].value_counts().head())

print("\nApp Types")
print(df["Type"].value_counts())

print("\nAverage Rating")
print(df["Rating"].mean())


# In[47]:


import plotly.graph_objects as go
fig = go.Figure()


# In[49]:


fig.write_html("EDAanalysis_Plot.html")

