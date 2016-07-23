
# coding: utf-8

# # Data Analysis Project (P2), NanoDegree program
# # for the Titanic DataSet
# # Tedros Hagos

# In, this project, I will use the titanic data which Contains demographics and passenger information from 891 of the 2224 . As described from the Kaggle website, the titanic sank after it colided with an iceberg on April, 15, 1912 which resulted in the overall death of 1502 out of 2224 passengers and crew. The question on the project for the titanic dataset is "What factors made people more likely to survive?". Based on this broader question, I will try to refine my questions to answer some of these questions: Is there any age group who got better survival chance? or if there is any survival rate difference between male and female? or if there was any survibal rate difference between classes where the passengers was sitting? Having watched the titanic movie several years back, I have my own sterotype especially on the question who had a better survival chance on that tragic incident. I will try to examine if the titanic data reveals the same or not. 
# 
# N.B. I have used a lot of resources and codes from the kaggle website, given in citation page down.
# 

# In[50]:

import pandas as pd
import numpy as np
titanic_df = pd.read_csv('titanic_data.csv') # just calling our data fram without need to specify the path directory
titanic_df.head() # To have a quick overview of what our data looks like, 
                  # the head() function gives us the first 5 rows of our data, if we want to see a specified # of rows, we have explicitly put it in the bracket.
                  # Naturally, we expect age to be an integer number, but as described on the kaggle webpage and as our data shows, we have 
                  # we have some ages with decimal points (estimated ages and infants less than a year)


# In[51]:

titanic_df.info() # Let's try to see the number of entries in each column and type of data for each column
                  # from the output window depicted below, we can see that there are missing data in some columns 
                 


# Let us investigate more to see the number of missing data for each variable.

# In[52]:

titanic_df.isnull().sum() # As displayed below, we have 3 columns with missing data.
                          # Cabin and age data have very high number of missings so, I will try to treat them separately.


# Let me try to treat the cabin dataset. As we can see from above, 687 out 891 rows are missing for the cabin column. 
# Since it is a string data, it seems difficult to use any statistical formula to estimate what is missing, even though we know the list of possible cabin letters. so I decided to drop the null and only try to display the not null values.

# In[119]:

print "Null cabin count: ", titanic_df["Cabin"].isnull().sum()

not_null_cabin = titanic_df.dropna(subset = ["Cabin"])
print "Total_Passengers_In_Cabin: ", len(titanic_df)
print "not_null_cabin passengers count: ", len(not_null_cabin)
not_null_cabin.head()


# In[66]:

split_cabin = not_null_cabin["Cabin"].str.split(" ") # create a Series containing lists of cabins
split_cabin = split_cabin.apply(pd.Series, 1).stack() # split the lists in multiple rows 
split_cabin.index = split_cabin.index.droplevel(-1) # remove the extra index column
split_cabin.name = "Cabin"

del not_null_cabin["Cabin"] # delete the original column
not_null_cabin = not_null_cabin.join(split_cabin) # insert the new column


# In[67]:

decks = not_null_cabin["Cabin"].str[0]
#decks.name = "Deck"

cabins = not_null_cabin["Cabin"].str[1:]
cabins.name = "Cabin"

del not_null_cabin["Cabin"]
not_null_cabin = pd.concat([not_null_cabin, decks, cabins], join="inner", axis=1)


# In[118]:

not_null_cabin.head()


# In[ ]:

In this section, I will try to answer the questions raised in the introductioin: Is rate of survival different for : different sex or classes


# In[120]:

#Let us also try to see if there was survival chance difference across different classes for male and female sepeately.
survival_sex_class = pd.crosstab(index=titanic_df["Survived"],
                             columns=[titanic_df["Pclass"],
                                      titanic_df["Sex"]],
                             margins=True)
survival_sex_class


# As we can see from the table above, females had better survival rate than male in each class. But the survival rate was also different across different classes. Class 3 had the highest number of passengers and the survival rate was very low for both sexes. Further investigation and data is needed why so. 

# In[86]:

# lets try to see the composition of passengers by sex and how much each survived.
crosstabulation=pd.crosstab(titanic_df.Survived,titanic_df.Sex, margins=True)

crosstabulation


# As we can see from the table displayed, raugly speaking, survival rate was better for females than male.
# I will try to visualize this result more to investigate it further.
# 

# Having seen survival rate difference between different classes and different sex, let us clean our data to get more meaningful explanation of our data. First we will recode the string values "male and female" of sex column to have a numerical value, which gives us ease to do statistical summary. In all our data outputs, the order is female and male, so let us recode the gender to have the nominal values 0 and 1.

# In[99]:

# Before changing the values of sex column, let us have the gender column and we will copy its values later.
titanic_df['Gender'] = 4  # creating a column 'gender"
titanic_df['Gender'] = titanic_df['Sex'].map( lambda x: x[0].upper() ) # copying first letters of sex into gender column
titanic_df['Gender'] = titanic_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int) # mapping exact values of sex into gender column

titanic_df.head()


# Measures of central tendency (mean and mode) are always affected by extreme values. Even though we have numerous missing values for our age, the age bracket varies greatly. I mean age range goes from less than a year upto 80+. So we will use the median age, which is always resistant to extreme values, to estimate the missing age accross the different classes for male and female seperately.

# In[100]:

median_ages = np.zeros((2,3))
median_ages


# In[103]:

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = titanic_df[(titanic_df['Gender'] == i) &                               (titanic_df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages


# Now let's try to treat the missing ages. But instead of filling the missing ages, let us make a carta-carbon copy of age, so that we can keep the originality of the data. 

# In[117]:


titanic_df['AgeFill'] = titanic_df['Age']  # Calculating median age for male and female seperately accross the 3 classes &
for i in range(0, 2):                      # we added the new column 'AgeIsNull" to check if it was initially null
    for j in range(0, 3):
        titanic_df.loc[ (titanic_df.Age.isnull()) & (titanic_df.Gender == i) & (titanic_df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
titanic_df['AgeIsNull'] = pd.isnull(titanic_df.Age).astype(int)
titanic_df[ titanic_df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill','AgeIsNull']].head(10)





# In[115]:

titanic_df.describe()


# from the above summary statistics, we can see that the two columns 'age' and 'AgeFill'are very similar, which suggests our estimate of of missing value is good

# In[94]:

survival_sex_class.describe()


# In[137]:

from scipy.stats import chi2_contingency
chisquare(pd.crosstab(titanic_df.Pclass, titanic_df.Survived))


# In[124]:

new_survived= pd.Categorical(titanic_df["Survived"])
new_survived = new_survived.rename_categories(["Not Survived","Survived"])
#titanic_df[ titanic_df[ ][['Gender','Pclass','Age','AgeFill','new_survived']].head(10)
titanic_df.head()


# In[92]:

titanic_df['Age'].mean()


# In[33]:

titanic_df[ ['Sex', 'Pclass', 'Age'] ]


# In[34]:

titanic_df[titanic_df['Age'] > 60]


# In[35]:

OldAge= titanic_df[titanic_df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
OldAge


# In[36]:

titanic_df.groupby('Survived').sum()['Age'].mean()


# In[37]:

titanic_df[titanic_df['Age'].isnull()][['Sex', 'Pclass', 'Age']]


# In[38]:

for i in range(1,4):
    print i, len(titanic_df[ (titanic_df['Sex'] == 'male') & (titanic_df['Pclass'] == i) ])


# In[39]:

for i in range(1,4):
    print i, len(titanic_df[ (titanic_df['Sex'] == 'male') & (titanic_df['Pclass'] == i) ])


# In[40]:

get_ipython().magic(u'matplotlib inline')


# In[41]:

#Import matplotlib.pyplot as plt
titanic_df ['Age'].dropna ().hist (bins=16, range=(0, 80)


# In[44]:

get_ipython().magic(u'matplotlib inline')
import pylab as P
titanic_df['Age'].hist()
P.show()


# In[45]:


#import pylab as P
titanic_df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()


# In[ ]:



