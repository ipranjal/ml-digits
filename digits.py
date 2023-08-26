# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.

# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# We will define utils here :

def preprocess_data(data):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets

def split_data(X,y,test_size=0.5,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params,model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf


# 1. Data Loading
digits = datasets.load_digits()

# 2. Data Visualization
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

data = digits.images
# 3. Data splitting
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3);

# 4. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 5. Train the data
model = train_model(X_train, y_train, model_params={'gamma': 0.001}, model_type='svm')



# Predict the value of the digit on the test subset
# 6. Model Predictino
predicted = model.predict(X_test)

# Below we visualize the first 4 test samples and show their predicted digit value in the title.


# 7. Quantitative sanity check 
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# classification_report builds a text report showing the main classification metrics.

#8. Model Evaluation
print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# We can also plot a confusion matrix of the true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
