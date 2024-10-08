import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt


# Function to read an image from file
"""def read_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not read the image from {file_path}.")
    return image"""

def read_image(file_path):
    image = Image.open(file_path)
    if image is None:
        print(f"Error: Could not read the image from {file_path}.")
    return np.array(image)

# Image preprocessing
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    unsharp_masked_image = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)
    #bilateral_filtered_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)


    return unsharp_masked_image


# Sobel edge detection for both X and Y
def sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobelx, sobely

def extract_horizontal_portion(binary_image, start_y, end_y):
    # Ensure binary_image is a binary image (values are 0 or 255)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Get the height and width of the image
    height, width = binary_image.shape[:2]

    # Create a mask with the desired horizontal portion
    mask = np.zeros_like(binary_image)
    mask[start_y:end_y, :] = 255

    # Apply the mask to the binary image
    horizontal_portion = cv2.bitwise_and(binary_image, mask)

    return horizontal_portion





# Count yarns using K-means clustering
def count_yarns(image):
    # Convert image to 1D array
    data = image.reshape((-1, 1))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    # Count number of yarns based on cluster centers
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    return counts[0] if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1] else counts[1]

# Main function
def main():
    # Specify the path to your image file
    image_path = r"C:\Users\Dell\PycharmProjects\pythonProject1\image.jpg"  # Replace with the actual path

    # Read image from file
    image = read_image(image_path)

    if image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        i = Image.fromarray(preprocessed_image)
        i.show()

        # Sobel edge detection for both X and Y

        sobelx, sobely = sobel_edge_detection(preprocessed_image)
        x = Image.fromarray(sobelx)
        x.show()
        """y = Image.fromarray(sobely)
        y.show()"""

        kernel = np.ones((4,4), np.uint8)

        # Apply morphological opening (erosion followed by dilation) to reduce noise
        opened_image = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kernel)

        # Display the results
        #cv2.imshow('Original Sobelx', sobelx)
        cv2.imshow('Opened Image', opened_image)
        #cv2.waitKey(5)
        cv2.destroyAllWindows()

        for j in range(0, 3):
            # Define the start and end y-coordinates for the horizontal portion
            start_y = 0 + j * 50
            end_y = 10 + j * 50

            # Extract the horizontal portion
            result = extract_horizontal_portion(opened_image, start_y, end_y)

            image_array = []
            image_array.append(result)



        """for j in image_array:
            img = Image.fromarray(j)
            img.show()
            cv2.waitKey(0)

            
        cv2.destroyAllWindows()"""

        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis('off')  # Turn off axis labels
        plt.show()
             # Display the result
            #cv2.imshow('Horizontal Portion', result)
            #cv2.waitKey(0)

        #yarn_count_horizontal = count_yarns(opened_image)
        #print(f"Yarn Count (Horizontal): {yarn_count_horizontal}")
        #cv2.destroyAllWindows()


        # Count yarns horizontally
        #yarn_count_horizontal = count_yarns(opened_image)

        # Count yarns vertically
       # yarn_count_vertical = count_yarns(sobely)

       # print(f"Yarn Count (Horizontal): {yarn_count_horizontal}")
        #print(f"Yarn Count (Vertical): {yarn_count_vertical}")"""

if __name__ == "__main__":
    main()