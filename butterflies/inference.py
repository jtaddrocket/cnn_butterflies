import torch
import cv2
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

from model import CNNModel

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
    default='../butterflies/input/test/GOLD BANDED/2.jpg',
    help='path to the input image')
args = vars(parser.parse_args())

# the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# list containing all the class labels
labels = [
    'adonis', 'american snoot', 'an 88', 'banded peacock', 'beckers white', 
    'black hairstreak', 'cabbage white', 'chestnut', 'clodius parnassian', 
    'clouded sulphur', 'copper tail', 'crecent', 'crimson patch', 
    'eastern coma', 'gold banded', 'great eggfly', 'grey hairstreak', 
    'indra swallow', 'julia', 'large marble', 'malachite', 'mangrove skipper',
    'metalmark', 'monarch', 'morning cloak', 'orange oakleaf', 'orange tip', 
    'orchard swallow', 'painted lady', 'paper kite', 'peacock', 'pine white',
    'pipevine swallow', 'purple hairstreak', 'question mark', 'red admiral',
    'red spotted purple', 'scarce swallow', 'silver spot skipper', 
    'sixspot burnet', 'skipper', 'sootywing', 'southern dogface', 
    'straited queen', 'two barred flasher', 'ulyses', 'viceroy', 
    'wood satyr', 'yellow swallow tail', 'zebra long wing'
    ]

# initialize the model and load the trained weights
model = CNNModel().to(device)
checkpoint = torch.load('outputs/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# read and preprocess the image
image = cv2.imread(args['input'])
# get the ground truth class
gt_class = args['input'].split('/')[-2]
orig_image = image.copy()
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]
cv2.putText(orig_image, 
    f"GT: {gt_class}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, (0, 255, 0), 2, cv2.LINE_AA
)
cv2.putText(orig_image, 
    f"Pred: {pred_class}",
    (10, 55),
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, (0, 0, 255), 2, cv2.LINE_AA
)
print(f"GT: {gt_class}, pred: {pred_class}")

# Display the image using matplotlib
plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
plt.title(f"GT: {gt_class}, Pred: {pred_class}")
plt.axis('off')
plt.show()

# Also display the image using cv2_imshow for Google Colab
cv2_imshow(orig_image)

cv2.imwrite(f"../butterflies/outputs/{gt_class}{args['input'].split('/')[-1].split('.')[0]}.png", orig_image)
