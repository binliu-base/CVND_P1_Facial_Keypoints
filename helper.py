import numpy as np
import matplotlib.pyplot as plt

def show_all_keypoints(image, predicted_key_pts):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=40, marker='.', c='m')

def visualize_output(test_image, test_outputs, i):

    plt.figure(figsize=(20,10))
    ax = plt.subplot(1, 2, i+1)

    # un-transform the predicted key_pts data
    predicted_key_pts = test_outputs.data
    predicted_key_pts = predicted_key_pts.cpu().numpy()
    # undo normalization of keypoints      
    predicted_key_pts = predicted_key_pts*50.0+100

    # call show_all_keypoints
    show_all_keypoints(np.squeeze(test_image), predicted_key_pts[0])
    plt.axis('on')