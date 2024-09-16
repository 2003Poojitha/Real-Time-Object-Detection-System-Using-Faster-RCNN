import torch
import cvzone
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Unable to load image at {file_path}. Please check the file path and integrity.")
            return

        image = cv2.resize(image, (640, 480))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_transform = transforms.Compose([transforms.ToTensor()])
        img = image_transform(image_rgb)
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
            bbox, scores, labels = pred['boxes'], pred['scores'], pred['labels']
            conf = torch.sum(scores > 0.70).item()
            for i in range(int(conf)):
                x1, y1, x2, y2 = bbox[i].numpy().astype('int')
                classname_index = labels[i].item()
                if classname_index < len(classnames):
                    class_detected = classnames[classname_index]
                else:
                    class_detected = 'Unknown'

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cvzone.putTextRect(image, class_detected, [x1 + 8, y1 - 12], scale=2, border=2)
        
        cv2.imwrite('output.png', image)
        display_image('output.png')

def display_image(image_path):
    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

root = tk.Tk()
root.title("Object Detection")

frame = tk.Frame(root)
frame.pack(expand=True)

panel = tk.Label(frame)
panel.pack(pady=20)

btn = tk.Button(frame, text="Load Image", command=load_image)
btn.pack(pady=20)

root.mainloop()
