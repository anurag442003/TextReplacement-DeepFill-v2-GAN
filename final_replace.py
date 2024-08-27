import argparse
import cv2
import numpy as np
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import uuid
import datetime


def detect_text_and_create_mask(image_path, search_text):
    client = vision.ImageAnnotatorClient()#api init

    with open(image_path, "rb") as image_file:
        content = image_file.read()#Read the image

    image = vision.Image(content=content)
    response = client.text_detection(image=image)#response from vision
    texts = response.text_annotations

    if not texts:
        print("No text detected.")
        return None, None

    bounding_boxes = []#store bb vertices of text
    for text in texts[1:]:
        if text.description.strip().lower() == search_text.lower():
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            bounding_boxes.append(vertices)
            print(f"Found '{search_text}' at: {vertices}")

    if not bounding_boxes:
        print(f"Text '{search_text}' not found.")
        return None, None

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)#mask creation

    for box in bounding_boxes:
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)#Draw polygon using bb, fill white
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_path = f'masks/mask_{timestamp}.png'
    cv2.imwrite(mask_path, mask)
    print(f"Mask created and saved as '{mask_path}'")

    return mask_path, bounding_boxes

def inpaint_with_deepfill_v2(image_path, mask_path, checkpoint, output_path='output.png'):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h // grid * grid, :w // grid * grid].unsqueeze(0)
    mask = mask[0:1, :h // grid * grid, :w // grid * grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
    generator_state_dict = torch.load(checkpoint)['G']#load generator model 
    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator

    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)

    image = (image * 2 - 1.).to(device)#map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32, device=device)#mask-1, unmasked-0

    image_masked = image * (1. - mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    image_inpainted = image * (1. - mask) + x_stage2 * mask

    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    inpainted_output_path = f'inpaints/inpainted_output_{timestamp}.png'
    img_out.save(inpainted_output_path)

    print(f"Inpainted image saved as '{inpainted_output_path}'")

    return inpainted_output_path



def render_replacement_text(image_path, bounding_boxes, replace_text):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for box in bounding_boxes:
            min_x = min(box, key=lambda p: p[0])[0]
            min_y = min(box, key=lambda p: p[1])[1]
            max_y = max(box, key=lambda p: p[1])[1]
            max_x = max(box, key=lambda p: p[0])[0]

            text_position = (min_x, min_y)
            bounding_box_height = max_y - min_y
            bounding_box_width = max_x - min_x
            #print(bounding_box_height,bounding_box_width)

            
            try:
                font = ImageFont.truetype("arial.ttf", bounding_box_height)#font size fits the bb
            except IOError:
                font = ImageFont.load_default()

            text_size = draw.textsize(replace_text, font=font)
            text_x = min_x + (bounding_box_width - text_size[0]) // 2
            text_y = min_y + (bounding_box_height - text_size[1]) // 2

            draw.text((text_x, text_y), replace_text, fill="black", font=font)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_image_path = f'output/Output_{timestamp}.png'
        img.save(final_image_path)
        print(f"Final image with replaced text saved as '{final_image_path}'")

    return final_image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Inpainting and Replacement Pipeline')
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--search_text", type=str, required=True, help="Text to be replaced")
    parser.add_argument("--replace_text", type=str, required=True, help="Replacement text")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output", type=str, default="output.png", help="Path for the output file")
    
    args = parser.parse_args()
    # print(args)

    mask_path, bounding_boxes = detect_text_and_create_mask(args.image, args.search_text)#mask create call

    if mask_path and bounding_boxes:
 
        inpainted_image_path = inpaint_with_deepfill_v2(args.image, mask_path, args.checkpoint, args.output)# inapaint call
        print(f"Inpainted image is saved at: {inpainted_image_path}")


        final_image_path = render_replacement_text(inpainted_image_path, bounding_boxes, args.replace_text)# call to put new text

