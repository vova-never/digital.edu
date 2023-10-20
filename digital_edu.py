import pandas as pd 
 
df_train = pd.read_csv("train.csv") 
 
 
df_train.drop(["id", "education_status", "life_main", "people_main",  
         "city", "last_seen","occupation_type", "occupation_name",  
         "career_start", "career_end"], axis = 1, inplace=True) 
 
 
def set_age(age: str): 
    try: 
        age = age.split(".") 
        year = int(age[2]) 
    except: 
        year = pd.NA 
     
    return year 
 
df_train["bdate"] = df_train["bdate"].apply(set_age) 
df_train["bdate"].fillna(df_train["bdate"].median(), inplace=True) 
 
 
def set_lang(langs: str): 
    langs = langs.replace("Русский;", "") 
    langs = langs.replace("Русский", "") 
 
    if langs == "False" or langs == "": 
        return ["English"] 
     
    return langs.split(";") 
 
df_train.langs = df_train.langs.apply(set_lang) 
 
 
#TODO: можливо дз, зробити фіктивні зміни 
df_train.education_form.fillna("Full-time", inplace=True) 
 
columns_names = list(pd.get_dummies(df_train.education_form).columns) 
 
df_train[columns_names] = pd.get_dummies(df_train.education_form) 
 
df_train.drop("education_form", axis=1, inplace=True) 
 
#def count_langs(langs: list):
    #return len(langs)

#df_train["langs_num"] = df_train ["langs"].apply(count_langs)

df_train["langs_num"] = df_train ["langs"].apply(lambda langs: len(langs)) 

#df_train.info()

#-----------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df_test = pd.read_csv('test.csv')

df_test.drop(["id", "education_status", "life_main", "people_main",  
         "city", "last_seen","occupation_type", "occupation_name",  
         "career_start", "career_end"], axis = 1, inplace=True) 

df_test["bdate"] = df_train["bdate"].apply(set_age) 
df_test["bdate"].fillna(df_train["bdate"].median(), inplace=True) 

df_test.langs = df_test.langs.apply(set_lang) 

df_test.education_form.fillna("Full-time", inplace=True)  
columns_names = list(pd.get_dummies(df_test.education_form).columns)  
df_test[columns_names] = pd.get_dummies(df_test.education_form)  
df_test.drop("education_form", axis=1, inplace=True) 


df_test["langs_num"] = df_test ["langs"].apply(lambda langs: len(langs))

#---------------------------

X_train = df_train.drop(["result", "langs"], axis = 1)
y_train = df_train.result

X_test = df_test.drop(["result", "langs"], axis=1)
y_test = df_test.result

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100
print(f"Точність:  {accuracy}%")