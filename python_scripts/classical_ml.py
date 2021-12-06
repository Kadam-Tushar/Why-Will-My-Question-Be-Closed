# from modules import * 
# from CustomTextDataset import *


# X_train = torch.load(export_path + "train_" + prob + prefix + "class_emb.pt",map_location=device).cpu()
# y_train = torch.load(export_path + "train_" + prob + prefix +  "class_emb_target.pt",map_location=device).cpu()

# X_test = torch.load(export_path + "test_" + prob + prefix + "class_emb.pt",map_location=device).cpu()
# y_test = torch.load(export_path + "test_" + prob + prefix +  "class_emb_target.pt",map_location=device).cpu()

# X_train= pd.DataFrame(X_train.numpy())
# y_train= pd.DataFrame(y_train.numpy())
# X_test= pd.DataFrame(X_test.numpy())
# y_test= pd.DataFrame(y_test.numpy())

# X_train.to_csv(export_path + "train_" + prob + prefix + "class_emb.csv")
# y_train.to_csv(export_path + "train_" + prob + prefix + "class_emb_target.csv")

# X_test.to_csv(export_path + "test_" + prob + prefix + "class_emb.csv")
# y_test.to_csv(export_path + "test_" + prob + prefix + "class_emb_target.csv")


import cudf, cuml
import pandas as pd 
from cuml.neighbors import KNeighborsClassifier as cuKNeighbors
from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_score
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC

# random forest depth and size
n_estimators = 100
max_depth = 17

export_path = '../Dataset/'  

prefix = ""
num_classes = 2
col = "closed" if num_classes == 2 else "comment"
prob = "bin" if num_classes == 2 else "multi"

X_train = pd.read_csv(export_path + "train_" + prob + prefix + "class_emb.csv",index_col=[0])
# X_train.drop("Unnamed: 0",inplace=True)

y_train = pd.read_csv(export_path + "train_" + prob + prefix + "class_emb_target.csv",index_col=[0])
X_test  = pd.read_csv(export_path + "test_" + prob + prefix + "class_emb.csv",index_col=[0])
y_test  = pd.read_csv(export_path + "test_" + prob + prefix + "class_emb_target.csv",index_col=[0])


model = SVC(kernel='rbf', C=10, gamma=1, cache_size=2000)
# model = LogisticRegression()

# model = cuRF( max_depth = max_depth, n_streams=1,
#               n_estimators = n_estimators,
#               random_state  = 0 )

# model = cuKNeighbors(n_neighbors=357)
# print(X_train.head())
model.fit(X_train, y_train)
print("model trained!")
y_hat = model.predict(X_test)
print("prediction done!")
accuracy = accuracy_score( y_test, y_hat)
f1_macro = f1_score( y_test, y_hat , average = "macro")
f1_micro = f1_score( y_test, y_hat , average = "micro")
recall_macro = recall_score( y_test, y_hat , average = "macro")
recall_micro = recall_score( y_test, y_hat , average = "micro")
precision_macro = precision_score( y_test, y_hat , average = "macro")
precision_micro = precision_score( y_test, y_hat , average = "micro")

print("SVC  performance over GRU word embeddings :")
print( "accuracy: ", accuracy )
print( "f1 macro: ", f1_macro )
print( "f1 micro: ", f1_micro )
print( "recall macro: ", recall_macro )
print( "recall micro: ", recall_micro )
print( "precision macro: ", recall_macro )
print( "recall micro: ", precision_micro )



