from dotenv import load_dotenv
import os
import psycopg2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns




load_dotenv()

conn = psycopg2.connect(
    dbname = os.getenv("DB_NAME"),
    user = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
    host = os.getenv("DB_HOST")
)


##################################################
##################################################
########### viewing all data

### looking at ticket queue for potential classification (paid vs dismissed vs unpaid)
if os.path.exists("data/ticket_queue_counts.csv"):
    count = pd.read_csv("data/ticket_queue_counts.csv")
else:
    query = """SELECT ticket_queue, COUNT(*) as count 
               FROM tickets GROUP BY ticket_queue;"""
    count = pd.read_sql(query, conn)

    total = count["count"].sum()
    count["percentage"] = (count["count"] / total * 100).round(2)


    ### saving since its a longer query 
    count.to_csv("data/ticket_queue_counts.csv", index=False)

#   ticket_queue     count  percentage
# 6         Paid  33633072       61.79
# 5       Notice   8344873       15.33
# 2       Define   7535110       13.84
# 3    Dismissed   4362658        8.02
# 0   Bankruptcy    489326        0.90
# 1        Court     60233        0.11
# 4  Hearing Req      5274        0.01
#                 54430546


### paying given that they recieved a notice/late penalties potentially 
if os.path.exists("data/payment_counts.csv"):
    pay_count = pd.read_csv("data/payment_counts.csv")
else:
    query = """SELECT notice_level, COUNT(*) as count
            FROM tickets
            WHERE ticket_queue = 'Paid'
            GROUP BY notice_level"""
    pay_count = pd.read_sql(query, conn)

    ### saving since its a longer query 
    pay_count.to_csv("data/payment_counts.csv", index=False)

#   notice_level    count
# 0               7588264
# 1         DETR  5524604
# 2          DLS  1427445
# 3         FINL  4292913
# 4         SEIZ  6338303
# 5         VIOL  8461543


##### cleaned in postgres after this 

### checking nulls/invalid values on relevant columns
#    bad_zips  null_area  null_make  null_fine  null_queue
# 0         8    4531936          0          0           0


### double checked no cars are unknown
# ^ many were mispelled or 'rare', grouped as OTHER in sample

### cleaned invalid zipcodes
### grouped target variable for classification




##################################################
##################################################
########### sampling data for more eda & modeling

### around 7% of all data for eda and modeling 
tik = pd.read_sql("SELECT * FROM tickets_sample;", conn)





# fixing lots of mispelled looking makes and numbers 
make_counts = tik['vehicle_make'].value_counts()
rare_makes = make_counts[make_counts < 50].index
tik['vehicle_make'] = tik['vehicle_make'].replace(rare_makes, 'OTHER')



### plots 

### class conts ! imbalanced
class_counts = tik['target'].value_counts()
class_counts.plot(kind='bar')
plt.title('Ticket Status Distribution')
plt.xlabel('Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

### tickets by the hour
tik['hour'].plot.hist(bins = 12)
plt.title("tickets by hour")
plt.show()

### payment rates by area 
area_rates = tik.groupby('community_area_name')['target'].value_counts(normalize = True).unstack()
area_rates['Paid'].sort_values().plot(kind ='barh', figsize =  (10,12))
plt.title('Payment Rate by Community Area')
plt.xlabel('% Paid')
plt.yticks(fontsize = 6)
plt.tight_layout()
plt.show()

### payment rates by year
### rate and raw numbers 
#year_rates = tik.groupby('year')['target'].value_counts(normalize=True).unstack()

year_rates = tik.groupby('year')['target'].value_counts().unstack()
year_rates.plot(kind='bar', figsize=(10,5))
plt.title('Ticket Status by Year')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.legend(title='Status')
plt.tight_layout()
plt.show()

### payment rate by violation type
viol_rates = tik.groupby('violation_description')['target'].value_counts(normalize = True).unstack()
viol_rates['Paid'].sort_values().tail(15).plot(kind = 'barh', figsize = (12,8))
plt.title('Violations by Payment Rate')
plt.xlabel('% Paid')
plt.xlim(0, 1.05)
plt.tight_layout()
plt.show()


### payments after gettig fined vs before
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

tik.groupby('target')['fine_level1_amount'].mean().plot(kind='bar', ax=axes[0])
axes[0].set_title('Avg Original Fine by Status')
axes[0].set_xlabel('Status')
axes[0].set_ylabel('Average Fine ($)')

tik.groupby('target')['fine_level2_amount'].mean().plot(kind='bar', ax=axes[1])
axes[1].set_title('Avg Fine + Penalties by Status')
axes[1].set_xlabel('Status')
axes[1].set_ylabel('Average Fine ($)')

plt.tight_layout()
plt.show()






##################################################
##################################################
########### modeling (decision tree, svm, knn, naive bayes)

### categorical columns need to be encoded 
cat_cols = ['violation_description', 'zipcode', 'license_plate_state',
            'vehicle_make', 'notice_level', 'community_area_name']

tik[cat_cols] = tik[cat_cols].fillna('Unknown')
tik['zipcode'] = tik['zipcode'].fillna('Unknown')


le = LabelEncoder()
for col in cat_cols:
    tik[col] = tik[col].astype(str)
    tik[col] = le.fit_transform(tik[col])


x = tik.drop(columns=['ticket_queue', 'target'])
le_target = LabelEncoder()
y = le_target.fit_transform(tik['target'])
### 0 - dismissed, 1 - paid, 2 - unpaid


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=29)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




##################################################
### models, balanced class weights & using f1 score as metric


### random forest 
rf = RandomForestClassifier(class_weight = 'balanced', random_state = 0)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

rf_f1 = f1_score(y_test, rf_pred, average = 'weighted')
rf_cm = confusion_matrix(y_test, rf_pred)


# feature importance 
importances = pd.Series(rf.feature_importances_, index = x.columns)
importances.sort_values().plot(kind = 'barh', figsize = (8,6))
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


### support vector machine
svm = SGDClassifier(loss = 'hinge', random_state = 0, class_weight = 'balanced')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

svm_f1 = f1_score(y_test, svm_pred, average = 'weighted')
svm_cm = confusion_matrix(y_test, svm_pred)


### k nearest neighbour
knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

knn_f1 = f1_score(y_test, knn_pred, average = 'weighted')
knn_cm = confusion_matrix(y_test, knn_pred)


### naive bayes 
# does not have class weights, so using sample balanced weights
sample_weights = compute_sample_weight('balanced', y_train)
nb = GaussianNB()
nb.fit(x_train, y_train, sample_weight = sample_weights)
nb_pred = nb.predict(x_test)

nb_f1 = f1_score(y_test, nb_pred, average = 'weighted')
nb_cm = confusion_matrix(y_test, nb_pred)





##################################################
### modeling results

### printing out f1 score
print(f"Random Forest F1: {rf_f1:.4f}")
print(f"SVM F1:           {svm_f1:.4f}")
print(f"KNN F1:           {knn_f1:.4f}")
print(f"Naive Bayes F1:   {nb_f1:.4f}")


# Random Forest F1: 0.7178
# SVM F1:           0.5124
# KNN F1:           0.6641
# Naive Bayes F1:   0.5782

# after balancing classes and using weighted f1 score
# Random Forest F1: 0.7160
# SVM F1:           0.5740
# KNN F1:           0.6660
# Naive Bayes F1:   0.6008


### confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()

cms = [rf_cm, svm_cm, knn_cm, nb_cm]
titles = ['Random Forest', 'SVM', 'KNN', 'Naive Bayes']

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot = True, cmap = 'Blues', fmt = 'd', ax = ax,
                xticklabels = le_target.classes_,
                yticklabels = le_target.classes_)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()
