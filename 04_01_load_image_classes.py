import cv2

img = cv2.imread('images/typewriter.jpg')
# print(img.shape)

all_rows = open('model/synset_words.txt').read().strip().split("\n")

# read only classes skipping the id field
img_classes = [r[r.find(' ') + 1:] for r in all_rows]

# print image classes
for i, img_class in enumerate(img_classes):
    if i == 4:
        break
    print(i, img_class)

# read and show image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
