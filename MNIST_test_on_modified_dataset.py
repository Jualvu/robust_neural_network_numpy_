import numpy as np
import pandas as pd
import os
print(os.getcwd())
from Layer_Dense import Layer_Dense 
from CategoricalCrossentropyLoss import CategoricalCrossentropyLoss
from Layer_Activation import Layer_Activation
import matplotlib.pyplot as plt
from Altering_Dataset import augment_image

#import mnist dataset
mnist_dataset_test = pd.read_csv('mnist_dataset/mnist_train.csv')


#separate features and labels for testing
X = mnist_dataset_test.iloc[:100, 1:]
y = mnist_dataset_test.iloc[:100, 0:1]


#print X and y
print("X dataset shape: " + str(X.shape))
print("y dataset shape: " + str(y.shape))
print("\n")



dense1 = Layer_Dense(n_input_neurons=784, n_output_neurons=128)
activation1 = Layer_Activation(type='relu')
dense2 = Layer_Dense(n_input_neurons=dense1.output_neurons_size, n_output_neurons=64)
activation2 = Layer_Activation(type='relu')
dense3 = Layer_Dense(n_input_neurons=dense2.output_neurons_size, n_output_neurons=10)
activation3 = Layer_Activation(type='softmax') # activation prime is None for softmax because we will use crossentropyLoss


#LOAD WEIGHTS AND BIASES
dense1.load_weights("values/w_layer1.npy")
dense1.load_biases("values/b_layer1.npy")
dense2.load_weights("values/w_layer2.npy")
dense2.load_biases("values/b_layer2.npy")
dense3.load_weights("values/w_layer3.npy")
dense3.load_biases("values/b_layer3.npy")





layers= [
    dense1,
    activation1,
    dense2,
    activation2,
    dense3,
    activation3
]



X = X.to_numpy()
y = y.to_numpy()




# MODIFYING DATASET

X_modified = np.array([augment_image(x_sample.reshape(28,28)) for x_sample in X])

# PLOT ORIGINAL VS MODIFIED IMAGES TO VISUALIZE

# for x_sample_modified, original in zip(X_modified, X):
    
#     # Plot modified x
#     plt.imshow(x_sample_modified, cmap='gray')
#     plt.axis('off')
#     plt.show()

#     # Plot original x
#     plt.imshow(original.reshape(28,28), cmap='gray')
#     plt.axis('off')
#     plt.show()









#print X and y
print("X dataset numpy shape: " + str(X.shape))



#Normalize the X values
X_modified = X_modified / 255.0

print("x.shape")
print(X_modified.shape)
print("y.shape")
print(y.shape)



#test the model
print("\nTESTING\n")

correct_predictions = 0
error = 0
sample = 1








#calculate predictions on each m example
for x_sample, y_true in zip(X_modified, y):
    print(f"\nsample {sample}:")
    sample += 1
    # FORWARD PROP
    #initiliaze output values with the m sample from the x tranining data
    output = x_sample.reshape(1, -1) # normalize
    print("X dataset numpy normalized shape: " + str(output.shape))

    for layer in layers:
        #keep propagating forward the output values from layer to layer
        output = layer.forward(inputs=output)

    #one hot encoding y_true
    y_true_one_hot_encoded = np.zeros(output.shape)
    y_true_one_hot_encoded[0, y_true] = 1.0
    
    #with the output we can calculate CategoricalCrossentropyLoss
    loss = CategoricalCrossentropyLoss.forward(y_prediction=output, y_true=y_true_one_hot_encoded)
    error += loss

    if np.argmax(output) == y_true:
        correct_predictions += 1
    # Code to see wrong predicted cases
    # else:
    #     # Pick one sample
    #     image = x.reshape(28, 28)  # reshape the flat image

    #     # Plot it
    #     plt.imshow(image, cmap='gray')
    #     plt.title(f'True: {y_true} | Pred: {np.argmax(output)}')
    #     plt.axis('off')
    #     plt.show()

    # Code to see the full 0-9 classes predictions
    # print("full output:\n")
    # i = 0
    # for num in output[0]:
    #     print(f"i: {i}  ---> {num:.4f}")
    #     i += 1

    # code to check the sum of predictions == 1
    # print("sum:")
    # print(np.sum(output[0]))



    print(f"\nPredicted num: {np.argmax(output)}")
    print(f"Actual num: {y_true}")

    # VISUALIZE EACH PREDICTION

    # # Pick one sample
    # image = x_sample.reshape(28, 28)  # reshape the flat image

    # # Plot corrected sample
    # plt.imshow(image, cmap='gray')
    # plt.title(f'True: {y_true} | Pred: {np.argmax(output)}')
    # plt.axis('off')
    # plt.show()

    


average_loss = error / len(X)
accuracy = correct_predictions / len(X)
print(f"Loss = {error:.4f}, Accuracy = {accuracy:.4f}")
