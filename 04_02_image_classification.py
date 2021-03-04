import numpy as np
import cv2

img = cv2.imread('images/typewriter.jpg')
print(img.shape)

all_classes = open('model/synset_words.txt').read().strip().split("\n")
# strip off class id
classes = [r[r.find(' ') + 1:] for r in all_classes]

net = cv2.dnn.readNetFromCaffe('model/bvlc_googlenet.prototxt', 'model/bvlc_googlenet.caffemodel')

blob = cv2.dnn.blobFromImage(img, 1, (224, 224))

net.setInput(blob)

# get prediction for match with all classes in synset
output = net.forward()
# print the probabilities for class matches
print(output)

# sort probabilities in ascending order and get top 5 predictions
idx = np.argsort(output[0])[::-1][:5]

for i, id in enumerate(idx, 1):
    print(f'{i}. {all_classes[id]}: Probability {output[0][id] * 100:.3} %')

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
