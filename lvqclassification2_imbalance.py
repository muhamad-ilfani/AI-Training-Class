#print(__doc__)

from LVQClassifier import LVQClassifier as LVQ
# we create clusters with 1000 and 100 points
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)

# LVQ parameter
epochs = 10
# LVQ1, no bias correction, uniform random initial state
clf = LVQ(n_components=30,alpha=0.1,epochs=epochs,
          initial_state='Uniform',bias_decrease_rate=1.0)
clf.fit(X, y)
X_LVQ = clf.weights
y_LVQ = clf.label_weights
title = 'LVQ with 30 comp. alpha=0.2 epochs=10'

plt.figure(figsize=(9, 7))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.scatter(X_LVQ[:, 0], X_LVQ[:, 1], c=y_LVQ,
               cmap=plt.cm.coolwarm, s=50, marker='^', edgecolors='k')
plt.title(title)

# create grid to evaluate model
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z,  colors='k', levels=[0], alpha=0.5, 
               linestyles=['-'])
plt.show()