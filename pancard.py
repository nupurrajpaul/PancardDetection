from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
# import PIL
import requests


# Open image and display
original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg', stream=True).raw)
tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png', stream=True).raw) 

# The file format of the source file.
print("Original image format : ",original.format) 
print("Tampered image format : ",tampered.format)

# Image size, in pixels. The size is given as a 2-tuple (width, height).
print("Original image size : ",original.size) 
print("Tampered image size : ",tampered.size) 

# Resize Image
original = original.resize((250, 160))
print(original.size)
original.save("sample_data\original.jpg")#Save image
tampered = tampered.resize((250,160))
print(tampered.size)
tampered.save("sample_data\\tampered.png")#Saves image

# Change image type if required from png to jpg
tampered = Image.open("sample_data\\tampered.png")

# tampered.save("sample_data\\tampered.jpg")#can do png to jpg
# # Display original image
# plt.imshow(original)
# plt.show()
# Display user given image
# tampered
# # load the two input images
original = cv2.imread('sample_data\original.jpg')
tampered = cv2.imread('sample_data\\tampered.png')
# # Convert the images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
# # Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
# # Calculating threshold and contours 
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# # loop over the contours
for c in cnts:
    # applying contours on image
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #Diplay original image with contour
print('Original Format Image')

im=Image.fromarray(original)
im.show()
# #Diplay tampered image with contour
print('Tampered Image')
im2=Image.fromarray(tampered)
im2.show()
# #Diplay difference image with black
print('Different Image')
im3=Image.fromarray(diff)
im3.show()
# #Display threshold image with white
print('Threshold Image')
im4=Image.fromarray(thresh)
im4.show()