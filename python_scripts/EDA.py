from typing import final
from modules import * 
from imblearn.under_sampling import RandomUnderSampler


from os import listdir
from os.path import isfile, join


sampled_dataset = 'sample_0.02_33.csv'
path = '/Users/tushar/iisc/sem3/TSE/project/Why-Will-My-Question-Be-Closed/Dataset/' 
df = pd.read_csv(path+sampled_dataset,dtype={'OwnerUserId': 'float64'})

print("Read sampled dataset ", sampled_dataset)

# Keeping only Questions i.e removing Answers, wiki, comments etc
df = df[ df['PostTypeId'] == 1]  

# Making new column to represent if question is closed or not 
df['closed'] = df['ClosedDate'].notnull().astype(int)

# converting to date format to datetime
df['ClosedDate'] = pd.to_datetime( df['ClosedDate'])
df['CreationDate'] = pd.to_datetime( df['CreationDate'])

# Filtering datasets
startDate = '07-01-2013'
df = df[df['CreationDate'] >= startDate ] 


# Getting reasons of closed questions, separating closed and open questions
closed_reasons = pd.read_csv('/Users/tushar/iisc/sem3/TSE/project/ClosedPosts-Type/closed_reasons.csv')
closed_questions =df[ df['closed'] == 1 ]
open_questions = df[ df['closed'] == 0 ]

print("separated dataset of open closed questions")

# Adding new column of comment in open questinos
open_questions['comment'] = 0 




# Inner joining joining dataset with closed question dataset to obtain reason for closing 
merged_closed_questions = pd.merge(closed_questions,closed_reasons,left_on = 'Id',right_on = 'id')


# Removing duplicate closed questions 
merged_closed_questions = merged_closed_questions[~(merged_closed_questions['comment'] == 101.0)]


# Replaing values with proper class labels 
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(np.nan,0)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(102.0,1)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(103.0,2)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(104.0,3)
merged_closed_questions['comment'] = merged_closed_questions['comment'].replace(105.0,4)


# converting to ints
merged_closed_questions['comment'] = merged_closed_questions['comment'].astype(int)


# Removing nans 
merged_closed_questions = merged_closed_questions.replace(np.nan,"") 
open_questions = open_questions.replace(np.nan,"")

# Preparing for same columns for to concat open and closed questions
closed_questions = merged_closed_questions[open_questions.columns]

# concating open and closed questions 

final_df = pd.concat([open_questions, closed_questions])

rus = RandomUnderSampler(random_state=seed_val)

# Re-Balancing binary classes 
X = final_df.drop(['closed'],axis= 1)
y = final_df['closed']
X_resampled, y_resampled = rus.fit_resample(X, y)
balanced_bin = pd.concat([X_resampled,y_resampled], axis = 1)


# Re-balancing multiple classes
X = final_df.drop(['comment'],axis= 1)
y = final_df['comment']
X_resampled, y_resampled = rus.fit_resample(X, y)
balanced_multi = pd.concat([X_resampled,y_resampled], axis = 1)

print("Distibution of binary classes after balancing")
print(balanced_bin['closed'].value_counts())

print("Distibution of multiple classes after balancing")
print(balanced_multi['comment'].value_counts())


balanced_multi['Tags'] = balanced_multi['Tags'].str.replace("<"," ")
balanced_multi['Tags'] = balanced_multi['Tags'].str.replace("<"," ")

balanced_bin['Tags'] = balanced_bin['Tags'].str.replace(">"," ")
balanced_bin['Tags'] = balanced_bin['Tags'].str.replace(">"," ")


balanced_bin['title_body_tags'] = balanced_bin['Title'] + balanced_bin['Body'] + balanced_bin['Tags']
balanced_multi['title_body_tags'] = balanced_multi['Title'] + balanced_multi['Body'] + balanced_multi['Tags']
balanced_bin = balanced_bin[['Id','title_body_tags','closed','comment']]
balanced_multi = balanced_multi[['Id','title_body_tags','closed','comment']]

balanced_bin.to_csv(path + 'balanced_bin.csv',index=False)
balanced_multi.to_csv(path + 'balanced_multi.csv',index=False)
print("Saved final dataset! binary and multi-class")







