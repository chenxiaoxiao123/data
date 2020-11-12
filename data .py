import numpy as np 
import pandas as pd 
import plotly.graph_objects as go

def ZDT1(x):
    f1 = x[0] #obbjective 1
    g = 1 + 9 * np.sum(x[1:D] / (D-1))
    h = 1 - np.sqrt(f1/g)
    f2 = g * h #objective 2
    return [f1,f2]

N = 50
D = 30
D_lower = np.zeros((1,D))
D_upper = np.ones((1,D))
M = 2

X = pd.DataFrame(np.random.uniform(low=D_lower, high=D_upper, size=(N,D)))
X.head(5) #Show only the first 5 solution
# print(X )

Y = np.empty((0,2))
print(Y)

for n in range(N):
    y = ZDT1(X.iloc[n])
    Y = np.vstack([Y,y])

Y = pd.DataFrame(Y,columns=['f1','f2']) #convert to DataFrame
Y.head(5) #Shows only first 5 sets of objective values
print(Y.head(5))
# print(Y)

#数据可视化
# fig = go.Figure(layout=dict(xaxis=dict(title='f1'),yaxis=dict(title='f2')))

# for index , row in Y.iterrows():
#     fig.add_scatter(x=[row.f1],y=[row.f2],name=f'solution{index+1}',mode='markers')

# fig.show()

fig = go.Figure(layout=dict(xaxis=dict(title='decision variables',range=[1,D]),yaxis=dict(title='value')))


for  index , row in X.iterrows():
    fig.add_scatter(x=X.columns.values+1,y=row, name=f'solution{index + 1}')

fig.show()