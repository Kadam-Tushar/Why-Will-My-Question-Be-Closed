from modules import * 
from CustomTextDataset import *
from models.GRU import *

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


preds = torch.load(export_path + prob + prefix + "class_emb.pt",map_location=device).cpu()
target = torch.load(export_path +prob + prefix +  "class_emb_target.pt",map_location=device).cpu()

preds_tsne_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(preds)
preds_pca_embedded = PCA(n_components=2).fit_transform(preds)
df = pd.DataFrame(preds_tsne_embedded,columns=['x-cord','y-cord'])
df['classes'] = target.numpy().tolist()
sns.scatterplot(data=df, x="x-cord", y="y-cord", hue="classes" , palette = sns.color_palette("tab10",n_colors=num_classes)).set_title('TSNE plot')
plt.legend(['0 - Open', '1 - Off-topic','2 - Unclear','3 - Broad','4 - Openion'])
plt.show()

df = pd.DataFrame(preds_pca_embedded,columns=['x-cord','y-cord'])
df['classes'] = target.numpy().tolist()

sns.scatterplot(data=df, x="x-cord", y="y-cord", hue="classes" ,  palette = sns.color_palette("tab10",n_colors=num_classes) ).set_title('PCA plots')
plt.legend(['0 - Open', '1 - Off-topic','2 - Unclear','3 - Broad','4 - Openion'])

plt.show()