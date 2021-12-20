#k-means聚类
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans#导入kmeans算法
#读取标准化之后的数据
airline_scale = np.load('../data/airline_scale.npz')['arr_0']
k=5
# print(airline_scale)
#构建模型，随机种子设为123
kmeans_model = KMeans(n_clusters=k,n_jobs=4,random_state=123)
fit_kmeans =kmeans_model.fit(airline_scale)#模型训练


#查看聚类结果
kmeans_cc=kmeans_model.cluster_centers_#聚类中心
print('各聚类中心为：\n',kmeans_model.labels_)#样本的类别标签
kmeans_labels=kmeans_model.labels_
print('各样本的类别标签为：\n',kmeans_labels)
r1=pd.Series(kmeans_model.labels_).value_counts()#统计不同类别样本的数目
print('最终每个类别的数目为：\n',r1)\
#输出聚类分群的结果
cluster_center =pd.DataFrame(kmeans_model.cluster_centers_,\
                             columns=['ZL','ZR','ZF','ZM','ZC'])#将聚类中心放在数据框中
cluster_center.index=pd.DataFrame(kmeans_model.labels_).\
    drop_duplicates().iloc[:,0]
print(cluster_center)

import matplotlib.pyplot as plt
# 客户分群雷达图
labels = ['ZL','ZR','ZF','ZM','ZC']
legen = ['客户群'+str(i+1)for i in cluster_center.index]#客户群命名，作为雷达图的图例
lstype = ['-','--',(0,(3,5,1,5,1,5)),':','-.']
kinds = list(cluster_center.iloc[:,0:])
#由于雷达图要保证数据闭合，因此在添加L列，并转换为np.ndarray
cluster_center=pd.concat([cluster_center,cluster_center[['ZL']]],axis=1)
centers= np.array(cluster_center.iloc[:,0:])

#分割圆周周长，并让其闭合
n=len(labels)
angle = np.linspace(0,2*np.pi,n,endpoint=False)
angle = np.concatenate((angle,[angle[0]]))
print(centers)
print(angle)
print(kinds)
#绘图
fig = plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,polar=True)
#以极坐标的形式绘图形
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#画线
for i in range(len(kinds)):
    ax.plot(angle,centers[i],linestyle=lstype[i],linewidth=2,label=kinds[i])
#添加属性标签
ax.set_thetagrids(angle[:-1]*180/np.pi,labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.show()
plt.close()