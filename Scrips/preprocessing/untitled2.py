import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Function to load the TensorFlow model
def load_model(model_path):
    """
    Load a TensorFlow model from the given path.
    
    Args:
        model_path (str): Path to the saved model.
    
    Returns:
        model: The loaded TensorFlow object detection model.
    """
    model = tf.saved_model.load(model_path)
    return model

# Function to preprocess the image
def preprocess_image(image_path, target_size, normalize=False):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The target height and width as a tuple.
        normalize (bool): Whether to normalize image pixel values to [0, 1].
    
    Returns:
        np.array: Preprocessed image.
    """
    try:
        image = Image.open(image_path)
    except IOError as e:
        print(f"Error opening image: {e}")
        return None
    
    image = image.resize(target_size)  # Resize image to the target size
    image_np = np.array(image)         # Convert to numpy array
    
    if normalize:
        image_np = image_np / 255.0     # Normalize pixel values to [0, 1]
        image_np = (image_np * 255).astype(np.uint8)  # Convert back to uint8
    
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

# Function to run inference on a single image
def run_inference_for_single_image(image, model):
    """
    Run inference on a single image.
    
    Args:
        image (np.array): The input image.
        model: The loaded TensorFlow object detection model.
    
    Returns:
        dict: The output dictionary from the model.
    """
    # Convert image to tensor with the correct dtype
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    
    # Run inference
    output_dict = model(input_tensor)
    
    # Output is a dictionary with keys 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'
    return output_dict

# Function to extract and process results from the model
def extract_results(output_dict):
    """
    Extract detection results from the output dictionary.
    
    Args:
        output_dict (dict): The output dictionary from the model.
    
    Returns:
        dict: A dictionary with detection results.
    """
    output_dict = {key:value.numpy() for key, value in output_dict.items()}
    return {
        'detection_boxes': output_dict['detection_boxes'][0],
        'detection_scores': output_dict['detection_scores'][0],
        'detection_classes': output_dict['detection_classes'][0],
        'num_detections': int(output_dict['num_detections'][0])
    }

# Function to visualize results
def visualize_results(image_path, boxes, scores, classes, class_names):
    """
    Visualize the detection results on the image.
    
    Args:
        image_path (str): Path to the image file.
        boxes (np.array): Array of bounding boxes.
        scores (np.array): Array of confidence scores.
        classes (np.array): Array of class labels.
        class_names (list): List of class names.
    """
    try:
        image = Image.open(image_path)
    except IOError as e:
        print(f"Error opening image: {e}")
        return
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    ax = plt.gca()
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            xmin *= image.width
            xmax *= image.width
            ymin *= image.height
            ymax *= image.height
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='blue')
            ax.add_patch(rect)
            
            class_index = int(classes[i])
            if class_index < len(class_names):  # Check if index is within bounds
                class_name = class_names[class_index]
            else:
                class_name = 'CompanyName'  # Default class name if out of bounds
            
            plt.text(xmin, ymin, f'{class_name}: {scores[i]:.2f}', color='red', fontsize=12)
    
    plt.show()

# Main function to run the complete process
def main():
    # Paths to your model directory and image file
    model_path = 'E:/Dataset/final_invoice/workspace/training_demo/exported-models/my_model/saved_model'
    image_path = 'E:/Dataset/final_invoice/workspace/training_demo/images/train_images/Invoice_21_blur.jpg'  # Make sure this is a valid image file
    target_size = (640, 640)  # Example size, adjust according to your model
    class_names = ['CompanyName', 'CompanyAddress', 'CustomerAddress', 'Total', 'InvoiceNumber', 'Total']  # Replace with your class names

    # Load the model
    model = load_model(model_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path, target_size, normalize=True)
    if preprocessed_image is None:
        print("Image preprocessing failed.")
        return

    # Run inference
    output_dict = run_inference_for_single_image(preprocessed_image, model)

    # Extract and process results
    results = extract_results(output_dict)

    # Visualize the results
    visualize_results(image_path, results['detection_boxes'], results['detection_scores'], results['detection_classes'], class_names)

if __name__ == "__main__":
    main()
