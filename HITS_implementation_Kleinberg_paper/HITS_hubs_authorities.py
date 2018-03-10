# We're going to adapt the classic HITS approach to allow us to find not the most authoritative web pages, but rather to find significant Twitter users. So, instead of viewing the world as web pages with hyperlinks (where pages = nodes, hyperlinks = edges), we're going to construct a graph of Twitter users and their retweets of other Twitter users (so user = node, retweet of another user = edge). Over this Twitter-user graph, we can apply the HITS approach to order the users by their hub-ness and their authority-ness.
# 
import json

original_tweet_dict = {}
retweet_dict = {}
for line in open('HITS.json', 'r'):
    if(json.loads(line)["retweeted_status"]["user"]['id'] != json.loads(line)["user"]['id']):
        if(json.loads(line)["retweeted_status"]["user"]['id'] in original_tweet_dict):
            original_tweet_dict[json.loads(line)["retweeted_status"]["user"]['id']].append(json.loads(line)["user"]['id'])
        else:
            original_tweet_dict[json.loads(line)["retweeted_status"]["user"]['id']] = [json.loads(line)["user"]['id']]
        
        if(json.loads(line)["user"]['id'] in retweet_dict):
            retweet_dict[json.loads(line)["user"]['id']].append(json.loads(line)["retweeted_status"]["user"]['id'])
        else:
            retweet_dict[json.loads(line)["user"]['id']] = [json.loads(line)["retweeted_status"]["user"]['id']]


row_idx = [key for key, value in retweet_dict.items() for item in range(len(value))]
col_idx = [i for ids in retweet_dict.values() for i in ids]

import numpy as np
import scipy.sparse as sp
mat = sp.dok_matrix((max(row_idx)+1, max(col_idx)+1 ))
#print len(set(row_idx))
for key, value in retweet_dict.items():
    for i in value:
        mat[key, i] += 1

print mat


# ### Using pandas to create a webgraph matrix
import pandas as pd

# clean up the example
g = {k: [v for v in vs] for k, vs in retweet_dict.items()}

edges = [(a, b) for a, bs in g.items() for b in bs]

df = pd.DataFrame(edges)

adj_matrix = pd.crosstab(df[0], df[1])

print adj_matrix


# ## HITS Implementation
# 
# Hub Scores
# 
# * user1 - score1
# * user2 - score2
# * ...
# * user10 - score10
# 
# Authority Scores
# 
# * user1 - score1
# * user2 - score2
# * ...
# * user10 - score10
# 
# * Assume all nodes start out with equal scores.

#Creating a unit column matrix for  authority initial values
xdata = [1 for i in adj_matrix.columns.values]
#print len(xdata)
authority_column_score_array = pd.DataFrame(xdata,index=adj_matrix.columns.values)

#print authority_column_score_array

#Creating a unit matrix for hub initial values
xdata = [1 for i in adj_matrix.index.values]
#print len(xdata)
hub_row_score_array = pd.DataFrame(xdata,index=adj_matrix.index.values)
#print hub_row_score_array


# In[17]:


from math import sqrt
k = 1000

#Using matrix multiplcation to compute the hubs and authority scores
for i in range(k):
    #Authority score computation using hub scores
    #Multiplying adj matrix with hub row scores
    authority_column_score_array = np.dot(adj_matrix.T, hub_row_score_array) 
    #Normalizing the values
    xdata = authority_column_score_array/sqrt(np.sum(np.square(authority_column_score_array,dtype=float)))

    #Converting the xdata to pandas dataframe
    authority_column_score_array = pd.DataFrame(xdata,index=adj_matrix.columns.values)
    #printing authority scores
    #print "iter",i
    #print authority_column_score_array



    #Hub score computation using authority scores
    #Multiplying adj matrix with authority column scores
    hub_row_score_array = np.dot(adj_matrix, authority_column_score_array) 
    #Normalizing the values
    xdata = hub_row_score_array/sqrt(np.sum(np.square(hub_row_score_array,dtype=float)))

    #Converting the xdata to pandas dataframe
    hub_row_score_array = pd.DataFrame(xdata,index=adj_matrix.index.values)


print "Hub Scores\n"
top_ten_hubs = hub_row_score_array[0].nlargest(10)
for key, value in top_ten_hubs.iteritems():
    print key,"-", value
    
print "\nAuthority Scores\n"
top_ten_authorities = authority_column_score_array[0].nlargest(10)
for key, value in top_ten_authorities.iteritems():
    print key,"-", value
