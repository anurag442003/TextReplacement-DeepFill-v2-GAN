#TextReplacement-DeepFill-v2-GAN

This project enables automated text detection, removal using a GAN (DeepFill v2), and replacement with custom text in images. It leverages Google Cloud Vision API for text detection and a DeepFill v2-based GAN for inpainting the removed regions.

##🔧 Features

Detects specific text in an image using OCR (Google Vision API).

Creates a mask for the detected text regions.

Uses a pre-trained GAN model (DeepFill v2) to remove the text by inpainting the masked regions.

Replaces the removed text with custom text, rendered in the same location.

## Instructions: 
1. enable google vision api
### Credentials
$env:GOOGLE_APPLICATION_CREDENTIALS=path_to_credentials.json

## Execution:
python final_replace.py --image [path_to_input_image] --search_text [text_to_be_replaced] --replace_text [new_text] --checkpoint pretrained/states_pt_places2.pth --output [path_to_output_image]
