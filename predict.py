import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import models
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['pretrained_model'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    im = Image.open(image)
    width, height = im.size
    image_coordinates = [width, height]
    max_element = max(image_coordinates)
    max_index = image_coordinates.index(max_element)
    
    if max_index == 0:
        min_index = 1
    else:
        min_index = 0
        
    aspect_ratio = image_coordinates[max_index] / image_coordinates[min_index]
    
    new_image_coordinates = [0, 0]
    new_image_coordinates[min_index] = 256
    new_image_coordinates[max_index] = int(256 * aspect_ratio)
    
    im = im.resize(new_image_coordinates)
    
    left_margin = (im.width - 224) / 2
    upper_margin = (im.height - 224) / 2
    right_margin = left_margin + 224
    lower_margin = upper_margin + 224
    im = im.crop((left_margin, upper_margin, right_margin, lower_margin))
    
    np_image = np.array(im)
    np_image_norm = ((np_image / 255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image_norm = np_image_norm.transpose((2, 0, 1))
    
    return np_image_norm

def predict(image_path, model_path, device, topk):
    image_tensor = torch.from_numpy(process_image(image_path))
    image_tensor = image_tensor.unsqueeze_(0).float()
    with torch.no_grad():
        model = load_checkpoint(model_path)
        model.eval()
        model, image_tensor = model.to(device), image_tensor.to(device)
        output = model(image_tensor)
        top_ps, top_class = torch.exp(output).topk(topk, dim = 1)
        top_ps = top_ps.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        idx_to_class = {idx:classes for classes, idx in model.class_to_idx.items()}
        top_class = [idx_to_class[top_class[i]] for i in range(top_class.size)]
        return top_ps, top_class

def display_predict(im_path, model_path, cat_to_name, device, topk):
    probs, classes = predict(im_path, model_path, device, topk)
    named_classes = [cat_to_name[i] for i in classes]
    
    for prob, class_name in zip(probs, named_classes):
        print("{}: {:.5f}".format(class_name, prob))
        
        
def input_args():
    parser = argparse.ArgumentParser(description = 'Getting inputs for prediction')
    parser.add_argument('path_to_image', help = 'Path to input image for prediction')
    parser.add_argument('checkpoint', help = 'Model checkpoint path to use for prediction')
    parser.add_argument('--topk', help = 'Top k most likely classes')
    parser.add_argument('--category_names', help = 'Path to json containing mapping from category to names')
    parser.add_argument('--gpu', help = 'Whether to use gpu for inference', action = 'store_true')
    
    return parser.parse_args()

def main():
    in_args = input_args()
    
    topk  = 5 if in_args.topk is None else int(in_args.topk)
    category_names = 'cat_to_name.json' if in_args.category_names is None else in_args.category_names
    gpu = False if in_args.gpu is None else True
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    display_predict(in_args.path_to_image, in_args.checkpoint, cat_to_name, device, topk)
    
if __name__ == '__main__':
    main()
    