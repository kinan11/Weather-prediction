import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

features = list(range(1, 7))
estimators = [25, 50, 75, 100, 125, 150, 175, 200]
depth = list(range(1, 6))
grid = []
for x in features:
    for y in estimators:
        for z in depth:
            grid.append([x, y, z])
results = pd.DataFrame(data=np.array(grid), columns=["parametr1", "parametr2", "parametr3"])

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
