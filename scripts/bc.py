import gradio as gr
from clipseg.clipseg import CLIPDensePredT
from pathlib import Path
from PIL import  Image, ImageFilter, ImageOps
import os
import tqdm
from torchvision import transforms
import cv2
import torch
import numpy as np
from modules import script_callbacks

def get_folder(file):
    folder = os.path.dirname(file.name)
    return folder

def on_ui_tabs():
    with gr.Blocks() as ui:
        with gr.Row():
            clipmode = gr.Radio(choices=["clip", "mask"], value="clip", label="mode")
            options = gr.CheckboxGroup(label = "option", choices=["crop", "crop with bg"], type = "value", value = "crop")
        with gr.Row():
            texts = gr.Textbox(label="words")
        with gr.Row():
            threshold = gr.Number(label = "threshold",value = 100,step = 1)
            blur = gr.Number(label = "blur",value = 0,step = 0.1)
            smooth= gr.Number(label = "smoothing",value = 0,step = 1)
        input_dir = gr.Textbox(label="Input directory")
        output_dir = gr.Textbox(label="Output directory")
        with gr.Row():
            batch = gr.Button(value="batch", variant='primary')
            single =  gr.Button(value="single", variant='primary')
        with gr.Row():
            with gr.Column(variant='panel'):
                in_image = gr.Image(label="Source", source="upload", interactive=True, type="pil")
            with gr.Column(variant='panel'):
                out_image = gr.Image(label="Result", interactive=False, type="pil")

        single.click(fn=singlefn,inputs = [in_image,texts,threshold,blur,clipmode,smooth,options],outputs =[out_image])
        batch.click(fn=batchfn,inputs = [input_dir,output_dir,texts,threshold,blur,clipmode,smooth,options],outputs =[])

    return [(ui, "Batch Clipper", "Batch Clipper")]

script_callbacks.on_ui_tabs(on_ui_tabs)

def singlefn(img,texts,threshold,blur,clipmode,smooth,options):
    return single(img,texts,threshold,blur,smooth,clipmode,options)

def single(img,texts,threshold,blur,smooth,clipmode,options):
    if not img:
        return
    texts = texts.split(",")
    masks = clipseg(img,texts,threshold,blur,smooth)
    origimage = np.array(img)
    mask = np.zeros_like(masks[0])
    for m in masks:
        mask += np.array(m)
    if "clip" in clipmode:
        outimage = np.where(mask[:,:,np.newaxis]==0, 255,origimage)
    else:
        outimage = np.where(mask[:,:,np.newaxis]==0, origimage,255)
    outimage = Image.fromarray(outimage)

    if "crop" in options:
        outimage = getcrop(outimage)
    elif "crop with bg" in options:
        outimage = getcrop_withbg(origimage, outimage)

    return outimage

def batchfn(input_dir, output_dir, texts,threshold,blur,clipmode,smooth,options):
    if not input_dir:
        raise ValueError("Please input Input_dir")
    if not os.path.exists(input_dir):
        raise ValueError("Input_dir not found")
    if not output_dir:
        raise ValueError("Please input Output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm.tqdm(os.listdir(input_dir)):
        src_path = os.path.join(input_dir, filename)
        stem, ext = os.path.splitext(filename)
        dst_path = os.path.join(output_dir, f"{stem}.png")
        img = Image.open(src_path)
        out = single(img,texts,threshold,blur,smooth,clipmode,options)
        out.save(dst_path,quality=95)
    
def clipseg(image,texts,threshold,blur,smooth):
    model = CLIPDensePredT(version='ViT-B/16',reduce_dim=64,complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load(clipsegdealer(), map_location=torch.device('cuda')), strict=False)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),transforms.Resize((512, 512)),])
    masks =[]
    imaget = transform(image).unsqueeze(0)
    with torch.no_grad():
        resions = model(imaget.repeat(len(texts), 1, 1, 1), texts)[0]
        resions = [r2m(resions[i][0],int(threshold),blur,int(smooth)) for i in range(len(resions))]
    masks.extend([r.resize((image.width, image.height)) for r in resions])
    return masks

def r2m(reasion, threshold, radius,filter):
    mask = torch.sigmoid(reasion).cpu()
    mask = (mask.numpy() * 256).astype(np.uint8)
    if filter > 0 : 
        kernel = np.ones((filter,filter),np.float32)/(filter*filter)
        mask = cv2.filter2D(mask,-1,kernel)
    mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]

    mask = Image.fromarray(mask)
    if radius > 0 :
        mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
        return mask.point(lambda x: 255 * (x > 0))
    return mask

def clipsegdealer():
    filename = Path("./models/clipseg/rd64-uni-refined.pth")
    if not os.path.exists(filename):
        filename.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading clipseg model weights...")
        with open(filename, 'wb') as fout:
            response = requests.get("https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni-refined.pth", stream=True)
            response.raise_for_status()
            for block in response.iter_content(4096):
                fout.write(block)
    return filename

def getcrop(img):
    inverted = ImageOps.invert(img)
    crop = inverted.getbbox()
    img = img.crop(crop)
    return img

def getcrop_withbg(img, img_for_bbox):
    inverted = ImageOps.invert(img_for_bbox)
    crop = inverted.getbbox()
    img = Image.fromarray(np.uint8(img))
    img = img.crop(crop)
    return img
