import numpy as np
import matplotlib.pyplot as plt

def visualize_output(figure, test_image, test_output, i):

    s = test_image.shape[0]/2
    # un-transform the predicted key_pts data
    predicted_key_pts = test_output.data
    # predicted_key_pts = predicted_key_pts.cpu().numpy()
    predicted_key_pts = predicted_key_pts.cpu().numpy()    
    # undo normalization of keypoints      
    predicted_key_pts = (predicted_key_pts + 0.999)*s

    # image is grayscale
    plt.imshow(test_image, cmap='gray')
    plt.scatter(predicted_key_pts[0][:, 0], predicted_key_pts[0][:, 1], s=40, marker='.', c='m')
    