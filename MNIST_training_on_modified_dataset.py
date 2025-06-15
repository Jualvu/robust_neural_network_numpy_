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



# Load different datasets of the 15 corrupted versions of MNIST from the MNIST-C dataset


# brightness corrupted version

X_bright_c = np.load('mnist_c/brightness/train_images.npy')
Y_bright_c = np.load('mnist_c/brightness/train_labels.npy')

#reshape
X_bright_c = X_bright_c.reshape(60000, 784)

print(f'X brightness corrupted dataset shape: {X_bright_c.shape}')
print(f'Y brightness corrupted dataset shape: {Y_bright_c.shape}')

#plot first sample
# plt.imshow(X_bright_c[0].reshape(28,28), cmap='gray')
# plt.axis('off')
# plt.title('brightness')
# plt.show()



# # canny_edges corrupted version

# X_canny_edges_c = np.load('mnist_c/canny_edges/train_images.npy')
# Y_canny_edges_c = np.load('mnist_c/canny_edges/train_labels.npy')

# #reshape
# X_canny_edges_c = X_canny_edges_c.reshape(60000, 784)

# print(f'X canny edges corrupted dataset shape: {X_canny_edges_c.shape}')
# print(f'Y canny edges corrupted dataset shape: {Y_canny_edges_c.shape}')


# #plot first sample
# # plt.imshow(X_canny_edges_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('canny_edges')
# # plt.show()





# # dotted_line corrupted version

# X_dotted_line_c = np.load('mnist_c/dotted_line/train_images.npy')
# Y_dotted_line_c = np.load('mnist_c/dotted_line/train_labels.npy')

# #reshape
# X_dotted_line_c = X_dotted_line_c.reshape(60000, 784)

# print(f'X dotted line corrupted dataset shape: {X_dotted_line_c.shape}')
# print(f'Y dotted line corrupted dataset shape: {Y_dotted_line_c.shape}')


# #plot first sample
# # plt.imshow(X_dotted_line_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('dotted_line')
# # plt.show()




# # fog corrupted version

# X_fog_c = np.load('mnist_c/fog/train_images.npy')
# Y_fog_c = np.load('mnist_c/fog/train_labels.npy')

# #reshape
# X_fog_c = X_fog_c.reshape(60000, 784)

# print(f'X fog corrupted dataset shape: {X_fog_c.shape}')
# print(f'Y fog corrupted dataset shape: {Y_fog_c.shape}')


# #plot first sample
# # plt.imshow(X_fog_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('fog')
# # plt.show()




# # glass_blur corrupted version

# X_glass_blur_c = np.load('mnist_c/glass_blur/train_images.npy')
# Y_glass_blur_c = np.load('mnist_c/glass_blur/train_labels.npy')

# #reshape
# X_glass_blur_c = X_glass_blur_c.reshape(60000, 784)

# print(f'X glass_blur corrupted dataset shape: {X_glass_blur_c.shape}')
# print(f'Y glass_blur corrupted dataset shape: {Y_glass_blur_c.shape}')


# #plot first sample
# # plt.imshow(X_glass_blur_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('glass_blur')
# # plt.show()



# # identity corrupted version

# X_identity_c = np.load('mnist_c/identity/train_images.npy')
# Y_identity_c = np.load('mnist_c/identity/train_labels.npy')

# #reshape
# X_identity_c = X_identity_c.reshape(60000, 784)

# print(f'X identity corrupted dataset shape: {X_identity_c.shape}')
# print(f'Y identity corrupted dataset shape: {Y_identity_c.shape}')


# #plot first sample
# # plt.imshow(X_identity_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('identity')
# # plt.show()






# # impulse_noise corrupted version

# X_impulse_noise_c = np.load('mnist_c/impulse_noise/train_images.npy')
# Y_impulse_noise_c = np.load('mnist_c/impulse_noise/train_labels.npy')

# #reshape
# X_impulse_noise_c = X_impulse_noise_c.reshape(60000, 784)

# print(f'X impulse_noise corrupted dataset shape: {X_impulse_noise_c.shape}')
# print(f'Y impulse_noise corrupted dataset shape: {Y_impulse_noise_c.shape}')


# #plot first sample
# # plt.imshow(X_impulse_noise_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('impulse_noise')
# # plt.show()




# # motion_blur corrupted version

# X_motion_blur_c = np.load('mnist_c/motion_blur/train_images.npy')
# Y_motion_blur_c = np.load('mnist_c/motion_blur/train_labels.npy')

# #reshape
# X_motion_blur_c = X_motion_blur_c.reshape(60000, 784)

# print(f'X impumotion_blurlse_noise corrupted dataset shape: {X_motion_blur_c.shape}')
# print(f'Y motion_blur corrupted dataset shape: {Y_motion_blur_c.shape}')


# # #plot first sample
# # plt.imshow(X_motion_blur_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('motion_blur')
# # plt.show()





# rotate corrupted version

X_rotate_c = np.load('mnist_c/rotate/train_images.npy')
Y_rotate_c = np.load('mnist_c/rotate/train_labels.npy')

#reshape
X_rotate_c = X_rotate_c.reshape(60000, 784)

print(f'X rotate corrupted dataset shape: {X_rotate_c.shape}')
print(f'Y rotate corrupted dataset shape: {Y_rotate_c.shape}')


#plot first sample
# plt.imshow(X_rotate_c[0].reshape(28,28), cmap='gray')
# plt.axis('off')
# plt.title('rotate')
# plt.show()






# # scale corrupted version

# X_scale_c = np.load('mnist_c/scale/train_images.npy')
# Y_scale_c = np.load('mnist_c/scale/train_labels.npy')

# #reshape
# X_scale_c = X_scale_c.reshape(60000, 784)

# print(f'X scale corrupted dataset shape: {X_scale_c.shape}')
# print(f'Y scale corrupted dataset shape: {Y_scale_c.shape}')


# #plot first sample
# # plt.imshow(X_scale_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('scale')
# # plt.show()





# # shear corrupted version

# X_shear_c = np.load('mnist_c/shear/train_images.npy')
# Y_shear_c = np.load('mnist_c/shear/train_labels.npy')

# #reshape
# X_shear_c = X_shear_c.reshape(60000, 784)

# print(f'X shear corrupted dataset shape: {X_shear_c.shape}')
# print(f'Y shear corrupted dataset shape: {Y_shear_c.shape}')


# #plot first sample
# # plt.imshow(X_shear_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('shear')
# # plt.show()





# # shot_noise corrupted version

# X_shot_noise_c = np.load('mnist_c/shot_noise/train_images.npy')
# Y_shot_noise_c = np.load('mnist_c/shot_noise/train_labels.npy')

# #reshape
# X_shot_noise_c = X_shot_noise_c.reshape(60000, 784)

# print(f'X shot_noise corrupted dataset shape: {X_shot_noise_c.shape}')
# print(f'Y shot_noise corrupted dataset shape: {Y_shot_noise_c.shape}')


# #plot first sample
# # plt.imshow(X_shot_noise_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('shot_noise')
# # plt.show()






# # spatter corrupted version

# X_spatter_c = np.load('mnist_c/spatter/train_images.npy')
# Y_spatter_c = np.load('mnist_c/spatter/train_labels.npy')

# #reshape
# X_spatter_c = X_spatter_c.reshape(60000, 784)

# print(f'X spatter corrupted dataset shape: {X_spatter_c.shape}')
# print(f'Y spatter corrupted dataset shape: {Y_spatter_c.shape}')


# # #plot first sample
# # plt.imshow(X_spatter_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('spatter')
# # plt.show()






# # stripe corrupted version

# X_stripe_c = np.load('mnist_c/stripe/train_images.npy')
# Y_stripe_c = np.load('mnist_c/stripe/train_labels.npy')

# #reshape
# X_stripe_c = X_stripe_c.reshape(60000, 784)

# print(f'X stripe corrupted dataset shape: {X_stripe_c.shape}')
# print(f'Y stripe corrupted dataset shape: {X_stripe_c.shape}')


# #plot first sample
# # plt.imshow(X_stripe_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('stripe')
# # plt.show()






# # zigzag corrupted version

# X_zigzag_c = np.load('mnist_c/zigzag/train_images.npy')
# Y_zigzag_c = np.load('mnist_c/zigzag/train_labels.npy')

# #reshape
# X_zigzag_c = X_zigzag_c.reshape(60000, 784)

# print(f'X zigzag corrupted dataset shape: {X_zigzag_c.shape}')
# print(f'Y zigzag corrupted dataset shape: {Y_zigzag_c.shape}')


# #plot first sample
# # plt.imshow(X_zigzag_c[0].reshape(28,28), cmap='gray')
# # plt.axis('off')
# # plt.title('zigzag')
# # plt.show()






# REGULAR MNIST DATASET
#separate features and labels for testing
X = mnist_dataset_test.iloc[:, 1:]
y = mnist_dataset_test.iloc[:, 0:1]


#print X and y
print("X dataset shape: " + str(X.shape))
print("y dataset shape: " + str(y.shape))
print("\n")



dense1 = Layer_Dense(n_input_neurons=784, n_output_neurons=512)
activation1 = Layer_Activation(type='relu')
dense2 = Layer_Dense(n_input_neurons=512, n_output_neurons=256)
activation2 = Layer_Activation(type='relu')
dense3 = Layer_Dense(n_input_neurons=256, n_output_neurons=128)
activation3 = Layer_Activation(type='relu')
dense4 = Layer_Dense(n_input_neurons=128, n_output_neurons=10)
activation4 = Layer_Activation(type='softmax') # activation prime is None for softmax because we will use crossentropyLoss


#LOAD WEIGHTS AND BIASES
dense1.load_weights("values/w_layer1.npy")
dense1.load_biases("values/b_layer1.npy")
dense2.load_weights("values/w_layer2.npy")
dense2.load_biases("values/b_layer2.npy")
dense3.load_weights("values/w_layer3.npy")
dense3.load_biases("values/b_layer3.npy")
dense4.load_weights("values/w_layer4.npy")
dense4.load_biases("values/b_layer4.npy")





layers= [
    dense1,
    activation1,
    dense2,
    activation2,
    dense3,
    activation3,
    dense4,
    activation4
]



X = X.to_numpy()
y = y.to_numpy()




# MANUALLY MODIFYING DATASET

# X_modified = np.array([augment_image(x_sample.reshape(28,28)) for x_sample in X])

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








#Change batch sizes if needed
X_rotate_c = X_rotate_c[50000:60000]
Y_rotate_c = Y_rotate_c[50000:60000]


#Change batch sizes if needed
X_bright_c = X_bright_c[2000:4000]
Y_bright_c = Y_bright_c[2000:4000]


#Normalize the X values
# X_modified = X_modified / 255.0
X = X / 255.0
X_rotate_c = X_rotate_c / 255.0
X_bright_c = X_bright_c / 255.0








# dataset to use in training
# X_to_use = X_rotate_c
# Y_to_use = Y_rotate_c
X_to_use = X_bright_c
Y_to_use = Y_bright_c

# Final shapes to use for training
print("Final x.shape")
print(X_to_use.shape)
print("Final y.shape")
print(Y_to_use.shape)

#train the model
print("\nTRAINING\n")



epochs = 5
learning_rate = 0.000005


for i in range(epochs):

    error = 0
    correct_predictions = 0
    # print(len(X))

    #calculate predictions on each m example
    for x_sample, y_true in zip(X_to_use, Y_to_use):
        
        # FORWARD PROP
        #initiliaze output values with the m sample from the x tranining data
        output = x_sample.reshape(1, -1) # normalize
        # print("X dataset numpy normalized shape: " + str(output.shape))

        for layer in layers:
            #keep propagating forward the output values from layer to layer
            output = layer.forward(inputs=output)

        #one hot encoding y_true
        y_true_one_hot_encoded = np.zeros(output.shape)
        y_true_one_hot_encoded[0, y_true] = 1.0
        
        
        # check if prediction is correct
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



        # print(f"\nPredicted num: {np.argmax(output)}")
        # print(f"Actual num: {y_true}")

        # VISUALIZE EACH PREDICTION

        # # Pick one sample
        # image = x_sample.reshape(28, 28)  # reshape the flat image

        # # Plot corrected sample
        # plt.imshow(image, cmap='gray')
        # plt.title(f'True: {y_true} | Pred: {np.argmax(output)}')
        # plt.axis('off')
        # plt.show()

        #with the output we can calculate CategoricalCrossentropyLoss
        loss = CategoricalCrossentropyLoss.forward(y_prediction=output, 
                                                   y_true=y_true_one_hot_encoded, 
                                                   layers_weights=[layer.weights for layer in layers if isinstance(layer, Layer_Dense)],
                                                   lambda_=0.001)
        error += loss


        # print("loss: ")
        # print(loss)


        # print("\nerror:")
        # print(error)
        # print("\nPredicted num: ")
        # print(np.argmax(output))
        # print("\nActual num: ")
        # print(y)

        # print("\nfull output:\n")
        # i = 0
        # for num in output[0]:
        #     print("i: " + str(i) + "  ---> " + str(num))
        #     i += 1

        # print("sum:")
        # print(np.sum(output[0]))



        # BACK PROP LEARNING

        #start backpropagation by calculating dE_dY with output


        dE_dY = output - y_true_one_hot_encoded  # softmax + cross-entropy simplification

        # i = 0
        for layer in reversed(layers):
            # print("layer : " + str(i))
            # i += 1
            dE_dY = layer.backward(output_gradient=dE_dY, learning_rate=learning_rate)

        


    average_loss = error / len(X_to_use)
    accuracy = correct_predictions / len(X_to_use)
    print(f"\nEpochs: {i+1}, Loss = {error:.4f}, Accuracy = {accuracy:.4f}\n")

# Code to save weights and biases after training


# Functions to save weigths and biases
def save_weights(file_name, weights):
    np.save(file_name, weights)

def save_biases(file_name, biases):
    np.save(file_name, biases)

answer = input("Save weights and biases?: Yes[y] No[n]\n")

if answer == 'y':
    print('\nSaving weights and biases...')
    j = 4
    for layer in reversed(layers):
            if type(layer) == Layer_Dense:
                # save weights
                save_weights(file_name=f"values/w_layer{j}.npy", weights=layer.weights)
                save_biases(file_name=f"values/b_layer{j}.npy", biases=layer.biases)
                j -= 1
else:
    print('\nNot Saving weights and biases...')
