import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

## recreate a DataFrame that looks how a gridsearch would look
features = list(range(1,7))
estimators = [25,50,75,100,125,150,175,200]
depth = list(range(1,6))
grid = []
for x in features:
    for y in estimators:
        for z in depth:
            grid.append([x,y,z])
results = pd.DataFrame(data=np.array(grid), columns=["parametr1","parametr2","parametr3"])

## let's assume your mean_test_score gradually increases with some noise
np.random.seed(42)
mean_scores = np.linspace(0.17, 0.42, len(results)) + np.random.normal(0, 0.01, len(results))
results["mean_scores"] = mean_scores

print(results['parametr1'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(results['parametr1'],results['parametr2'],results['parametr3'])

ax.set_xlabel("pierwszy hiperparametr")

ax.set_ylabel("drugi hiperparametr")

ax.set_zlabel("trzeci hiperparametr")

plt.show()

# fig = px.scatter_3d(results, x='max_features', y='n_estimators', z='max_depth', color='mean_scores')
# fig.update_layout(
#     title="Hyperparameter tuning",
#     autosize=True, width=700, height=700,
#     margin = dict(l=65, r=50, b=65, t=90))
# fig.show()
