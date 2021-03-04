import cv2

img = cv2.imread('images/devon.jpg')
print(f'shape={img.shape}')

cv2.imshow('Image', img)

blue = img[:, :, 0]     # blue channel
green = img[:, :, 1]    # green channel
red = img[:, :, 2]      # red channel

cv2.imshow('Blue Channel', blue)
cv2.imshow('Green Channel', green)
cv2.imshow('Red Channel', red)

cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()
