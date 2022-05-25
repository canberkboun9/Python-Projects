import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('mnist_train.csv')
label = df['label']
df.drop('label', axis = 1, inplace = True)

# Part 1
# Plotting 10 digits
digits = np.unique(label)
y = np.array(label)
X = np.array(df)
fig, axs = plt.subplots(10, 10, figsize=(20,20))
for i,d in enumerate(digits):
    for j in range(10):
        axs[i,j].imshow(X[y == d][j].reshape(28, 28), cmap = "gray_r")
        axs[i,j].axis('off')
plt.tight_layout()
plt.show()

# Part 2
# Plotting the mean image.
mean_img = df.mean()
mean_img = np.array(mean_img).reshape(28, 28)
plt.imshow(mean_img, interpolation = None, cmap = 'gray_r')
plt.show()

df = df - df.mean()  # centering the data
cov_matrix = df.cov()

# plotting the eigenvectors and eigenvalues that are the most significant 100 and 50 ones.
cov_matrix = np.array(cov_matrix)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
eigen_vectors = np.array(eigen_vectors, dtype='float')
eigen_values = np.array(eigen_values[:50])
figure = plt.figure(figsize=(15,9))
values_axis = np.sort(np.arange(len(eigen_values)) + 1)
plt.plot(values_axis, eigen_values, 'ro-', linewidth=1)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue') 
plt.show()

eigen_vectors = eigen_vectors.T  # take transpose due to np.linalg.eig

fig, axs = plt.subplots(10, 10, figsize=(20,20))
counter = 0
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(eigen_vectors[counter].reshape((28,28)), cmap = "gray_r")
        axs[i,j].axis('off')
        counter += 1
plt.tight_layout()
plt.show()

# Part 3
df_test = pd.read_csv('mnist_test.csv')
label = df_test['label']
df_test.drop('label', axis = 1, inplace = True)
df_test = df_test - df_test.mean()  # centering the data
cov_matrix = df_test.cov()

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
eigen_vectors = np.array(eigen_vectors, dtype='float')
eigen_values = np.array(eigen_values, dtype='float')

eigen_vectors = eigen_vectors.T

# Projection of our data onto the principal components
final_df_test = np.matmul(eigen_vectors[:2], df_test.T)
# took transpose of our data to achieve dimensions to comply for matrix multiplication

final_df_testT = np.vstack((final_df_test, label)).T
dataFrame = pd.DataFrame(final_df_testT, columns = ['pca_1', 'pca_2', 'label'])
dataFrame

sns.FacetGrid(dataFrame, hue = 'label', size = 15).map(sns.scatterplot, 'pca_1', 'pca_2').add_legend()
plt.show()

# Part 4
df_test = pd.read_csv('mnist_test.csv')
label = df_test['label']
df_test.drop('label', axis = 1, inplace = True)
tsne = TSNE(n_components = 2, random_state=0)
tsne_res = tsne.fit_transform(df_test)

plt.figure(figsize=(16,10))
sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full')
plt.show()

# Part 6
#Reconstruction Part.
df_test = pd.read_csv('mnist_test.csv')
label = df_test['label']
df_test.drop('label', axis = 1, inplace = True)
X = np.array(df_test)
image = X[128].reshape(28,28) # plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray_r')
plt.show()

images = []
# Explained Variance Ratio
sum_of_eigen_values = int(sum(eigen_values))
percentages = []
for i in range(2,783,10):
    percentage = 100*eigen_values[i]/sum_of_eigen_values
    percentages.append(percentage)
    
percentage = np.array(percentages)

cumul_percentages = []
cumul_percentage = 0
star_point = 0
for i in range(0,784):
    percentage = 100*eigen_values[i]/sum_of_eigen_values
    cumul_percentage += percentage
    cumul_percentages.append(cumul_percentage)
    if cumul_percentage < 95.01:
        star_point = i+2
    
mean = np.array(df_test.mean())
for i in range (2,783,10):
    final_df_test = np.matmul(eigen_vectors[:i], X[128])
    # reconstruction
    rec_ = np.matmul(final_df_test, eigen_vectors[:i])
    rec_ = rec_ + mean
    images.append(rec_) 

print("for 95% explained variance ratio, number of eigen vectors:", star_point)
figure = plt.figure(figsize=(15,9))
x_axis = list(range(0,784))
plt.plot(x_axis, cumul_percentages, 'ro-', linewidth=1)
plt.title('Explained Variance Ratio Cumulative')
plt.xlabel('Number of eigenvectors')
plt.ylabel('Percentage') 
plt.show()

fig, axs = plt.subplots(9, 9, figsize=(20,20))
counter = 0
for i in range(9):
    for j in range(9):
        axs[i,j].imshow(images[counter].reshape(28, 28), cmap = "gray_r")
        axs[i,j].axis('off')
        counter += 1
        if counter == 78:
            break
plt.tight_layout()
plt.show()

figure = plt.figure(figsize=(15,9))
x_axis = list(range(2,783,10))
plt.plot(x_axis, percentages, 'ro-', linewidth=1)

plt.title('Explained Variance Ratio For each')
plt.xlabel('Number of eigenvectors')
plt.ylabel('Percentage') 
plt.show()

# for non digit image
df_test = pd.read_csv('fashion-mnist_test.csv')
label = df_test['label']
df_test.drop('label', axis = 1, inplace = True)
df_test = df_test - df_test.mean()  # centering the data
cov_matrix = df_test.cov()

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
eigen_vectors = np.array(eigen_vectors, dtype='float')
eigen_values = np.array(eigen_values, dtype='float')

eigen_vectors = eigen_vectors.T

#Reconstruction Part for non digit.
df_test = pd.read_csv('fashion-mnist_test.csv')
label = df_test['label']
df_test.drop('label', axis = 1, inplace = True)
X = np.array(df_test)
image = X[243].reshape(28,28) # plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray_r')
plt.show()
image.shape

images = []
# Explained Variance Ratio
sum_of_eigen_values = int(sum(eigen_values))
percentages = []
for i in range(2,783,10):
    percentage = 100*eigen_values[i]/sum_of_eigen_values
    percentages.append(percentage)
    
percentage = np.array(percentages)

cumul_percentages = []
cumul_percentage = 0
star_point = 0
for i in range(0,784):
    percentage = 100*eigen_values[i]/sum_of_eigen_values
    cumul_percentage += percentage
    cumul_percentages.append(cumul_percentage)
    if cumul_percentage < 95.01:
        star_point = i+2
    
mean = np.array(df_test.mean())
for i in range (2,783,10):
    final_df_test = np.matmul(eigen_vectors[:i], X[243])
    # reconstruction
    rec_ = np.matmul(final_df_test, eigen_vectors[:i])
    rec_ = rec_ + mean
    images.append(rec_) 

print("for 95% explained variance ratio, number of eigen vectors:", star_point)

fig, axs = plt.subplots(9, 9, figsize=(20,20))
counter = 0
for i in range(9):
    for j in range(9):
        axs[i,j].imshow(images[counter].reshape(28, 28), cmap = "gray_r")
        axs[i,j].axis('off')
        counter += 1
        if counter == 78:
            break
plt.tight_layout()
plt.show()
