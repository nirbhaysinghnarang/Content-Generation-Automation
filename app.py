from flask import Flask, request, send_file, jsonify, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
import requests
from PIL import Image, ImageFont, ImageDraw
from moviepy.editor import ImageSequenceClip, AudioFileClip, afx, concatenate_videoclips
import numpy as np
import cv2
import io
import os
import random
from gc import collect

app = Flask(__name__)

system_prompt = 'You are a helpful assistant.'

style_prompt = 'The characters should be drawn from the time of the Hindu epic the Mahabharata. Use a style reminiscent of traditional painting techniques to create an image that is suitable for illustrating Hindu religious or mythological narratives. The bottom 1/3 of the image should depict the ground, air or some other unimportant item.'

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)


def dalle(bearer_token, prompt, style_prompt=style_prompt):
    headers = {
      'Authorization': bearer_token
    }
    payload = {
        "model": "dall-e-3",
        "prompt": prompt + ' ' + style_prompt,
        "n": 1,
        "size": "1024x1024"
    }

    response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        for img_data in data['data']:
            img_bytes = requests.get(img_data['url']).content
            img = Image.open(io.BytesIO(img_bytes))
            return img
    else:
        print(f"Failed to generate images: {response.text}")


def chatgpt(bearer_token, prompt, system_prompt=system_prompt):
    headers = {
        'Authorization': bearer_token,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content'].strip()  # Extract and return the text from the response
    else:
        print(f"Failed to generate text: {response.text}")
        return None
    

def generate_short_content(bearer_token, prompt, filename, num_slides, gen_images):
    stories = []
    for i in range(num_slides):
        story = generate_story(bearer_token, prompt)
        print('story ',story)
        stories.append(story)
    if gen_images:
        images = generate_images(bearer_token, prompt, num_slides)
    else:
        images = load_images(num_slides)
    create_slideshow(images, stories, filename)


def generate_zoom_pan_frames(np_img, num_frames):
    height, width = np_img.shape[:2]
    zoom_scale = 2.0  # End with a 2x zoom
    start_x, start_y = 0, 0
    end_x, end_y = int(width * 0.3), int(height * 0.3)

    for i in range(num_frames):
        scale = 1 + (zoom_scale - 1) * (i / (num_frames - 1))
        scaled_frame = cv2.resize(np_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_height, scaled_width = scaled_frame.shape[:2]
        current_x = int(start_x + (end_x - start_x) * (i / (num_frames - 1)))
        current_y = int(start_y + (end_y - start_y) * (i / (num_frames - 1)))
        x1 = max(0, min(current_x, scaled_width - width))
        y1 = max(0, min(current_y, scaled_height - height))
        cropped_frame = scaled_frame[y1:y1 + height, x1:x1 + width]
        yield cropped_frame


def generate_story(bearer_token, prompt):
    return chatgpt(bearer_token, prompt, 'You are the Hindu god Krishna. Please respond to the users question with wisdom and compassion with a direct quotation from the Mahabharata in Sanskrit of not more than 200 Devanigiri characters, along with the English translation, separated by a new line, with no quotation marks or other surrounding text. Do not include any new lines in the English translation.')


def get_random_files(directory_path):
    # List all files in the directory and filter out non-image files if necessary
    files = [f for f in os.listdir(directory_path) if f.endswith(('.webp', '.mp3'))]
    
    # Shuffle the list of files to ensure random order
    random.seed()
    random.shuffle(files)

    return files

def load_images(num_images=1):
    # Define the directory path where images are stored
    #directory_path = './resources/images'
    directory_path = './output/images'
    files = get_random_files(directory_path)
    
    # Load the specified number of images without duplicates
    images = []
    for filename in files[:num_images]:
        # Form the full file path
        file_path = os.path.join(directory_path, filename)
        # Open and load the image
        with open(file_path, 'rb') as file:
            img = Image.open(file)
            img.load()  # Load the image data to ensure it's all read
            images.append(img)
    
    return images
    

# Function to generate images using OpenAI's DALL-E API
def generate_images(bearer_token, story, num_images=1):
    images = []
    for i in range(num_images):
        prompt = chatgpt(bearer_token, f"Based on the following prompt, create an image prompt suitable for an image generation tool like dall-e. The prompt should highlight the divine presence of Lord Krisna. Respond with the image prompt itself without any introductory or extraneous text. Prompt: {story}", "You are a helpful assistant that converts the text prompt into an image prompt suitable for an image generation tool like dall-e")
        print('img prompt ',prompt)
        img = dalle(bearer_token, prompt)
        images.append(img)
        unique_id = str(uuid.uuid4())
        img.save(f"./output/images/{unique_id}.webp")

    return images


def wrap_text(text, font, max_width):
    lines = []
    # Split the text into lines based on newline characters
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        words = paragraph.split()
        current_line = ''
        
        while words:
            # Check if adding the next word would exceed the max width
            if font.getsize(current_line + words[0] + ' ')[0] <= max_width:
                current_line += words.pop(0) + ' '
            else:
                lines.append(current_line.strip())
                current_line = words.pop(0) + ' '

        # Add any remaining text
        if current_line:
            lines.append(current_line.strip())
    
    return lines


def put_text_on_image(img, text, font_scale=1, text_color=(0, 0, 0), bg_color=(255, 255, 255, 0)):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Use a Unicode compatible font
    font_path = "./resources/NotoSansDevanagari-Bold.ttf"
    font = ImageFont.truetype(font_path, int(32 * font_scale))

    # Text wrapping
    wrapped_text = wrap_text(text, font, img_pil.width - 20)
    # Calculate line height based on the tallest character 'Mg' and add padding
    line_height = font.getsize('Mg')[1] + 10

    # Adjust text_y_start to ensure all text fits in the image
    total_text_height = line_height * len(wrapped_text)
    text_y_start = img_pil.height - total_text_height - 20  # Subtract 10 pixels for the gap

    # Check if the text height exceeds the image height
    if total_text_height > img_pil.height:
        print("Warning: Text exceeds image height, consider resizing the image or reducing font size.")

    # Draw a rectangle background and text for each line
    for i, line in enumerate(wrapped_text):
        bg_rect_top_left = (0, text_y_start + i * line_height - 5)
        bg_rect_bottom_right = (img_pil.width, text_y_start + (i + 1) * line_height + 5)
        draw.rectangle([bg_rect_top_left, bg_rect_bottom_right], fill=bg_color)

        text_width, _ = draw.textsize(line, font=font)
        text_x = (img_pil.width - text_width) / 2  # Center the text horizontally
        draw.text((text_x, text_y_start + i * line_height), line, font=font, fill=text_color)

    # Convert back to numpy array
    return np.array(img_pil)


def create_slideshow(images, stories, filename, duration=10, fadein_duration=2, fadeout_duration=2, fps=24):
    clips = []
    for img, story in zip(images, stories):
        np_img = np.array(img)
        for frame in generate_zoom_pan_frames(np_img, int(duration * fps / len(images))):
            processed_frame = put_text_on_image(frame, story)
            clips.append(processed_frame)
            del frame  # Free memory of each frame after processing
        del np_img  # Free memory of the numpy image
        collect()  # Force garbage collection

    # Create video from frames
    clip = ImageSequenceClip(clips, fps=fps)
    audio_path = './resources/audio/sample_audio.mp3'  # Example path
    audio = AudioFileClip(audio_path).audio_loop(duration).set_duration(clip.duration).audio_fadein(fadein_duration).audio_fadeout(fadeout_duration)
    clip.set_audio(audio)
    clip.write_videofile(filename, fps=24, codec='libx264', audio_codec='aac')
    audio.close()  # Ensure resources are cleared
    del clips  # Ensure all frames are deleted
    collect()  # Final garbage collection


@app.route('/generate_video', methods=['POST'])
def generate_video():
    # Get bearer token from headers
    bearer_token = request.headers.get('Authorization')

    if not bearer_token:
        return jsonify({"error": "Authorization token is missing"}), 401

    # Get JSON data for prompt and story
    data = request.get_json()
    prompt = data.get('prompt', "")
    num_slides = data.get('num_slides', 1)
    gen_images = data.get('gen_images', False)

    if not prompt:
        return jsonify({"error": "Prompt must be provided"}), 400

    print('prompt ',prompt)

    # Generate a unique filename for the video
    unique_id = str(uuid.uuid4())
    filename = f"./output/videos/{unique_id}.mp4"

    # Generate video based on the bearer token, prompt, and story
    generate_short_content(bearer_token, prompt, filename, num_slides, gen_images)

    # Check if the video file has been created and exists

    # Generate URL for the created video
    # video_url = url_for('get_video', video_id=unique_id, _external=True)
    video_url = f"https://{request.host}/videos/{unique_id}.mp4"
    return jsonify({"video_url": video_url})

'''
@app.route('/video/<video_id>')
def get_video(video_id):
    video_path = f"./output/videos/{video_id}.mp4"
    try:
        return send_file(video_path, as_attachment=True, download_name=video_path)
    except Exception as e:
        return video_path
'''

@app.route('/')
def home():
    return "Welcome to the Video Generator API!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6000))  
    app.run(host='0.0.0.0', port=port)
