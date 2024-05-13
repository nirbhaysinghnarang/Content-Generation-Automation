from flask import Flask, request, jsonify, send_file, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
import requests
from PIL import Image, ImageFont, ImageDraw
from moviepy.editor import ImageSequenceClip, AudioFileClip, afx, concatenate_videoclips
import numpy as np
import cv2
import io
import os
import json
import random
from gc import collect

app = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)


def dalle(bearer_token, prompt):
    headers = {
      'Authorization': bearer_token
    }
    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
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
        app.logger.error(f"Failed to generate images: {response.text}")


def chatgpt(bearer_token, prompt, system_prompt="You are a helpful assistant"):
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
        app.logger.error(f"Failed to generate text: {response.text}")
        return None
    

def get_random_files(directory_path):
    # List all files in the directory and filter out non-image files if necessary
    files = [f for f in os.listdir(directory_path) if f.endswith(('.webp', '.mp3'))]
    
    # Shuffle the list of files to ensure random order
    random.seed()
    random.shuffle(files)

    return files

def load_images(num_images=1):
    # Define the directory path where images are stored
    #directory_path = './output/images'
    directory_path = './resources/images'
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
def generate_images(bearer_token, stories, img_prompt, style_prompt):
    images = []
    for story in stories:
        prompt = chatgpt(bearer_token, f"Based on the following text, create an image prompt suitable for an image generation tool like dall-e. {img_prompt} Respond with the image prompt itself without any introductory or extraneous text. {style_prompt} Text: {story}")
        app.logger.info('img prompt: ' + prompt)
        img = dalle(bearer_token, prompt)
        images.append(img)
        unique_id = str(uuid.uuid4())
        img.save(f"./output/images/{unique_id}.webp")

    return images

# Function to generate stories using chatGPT
def generate_stories(bearer_token, prompt, system_prompt, num_slides, num_chars):
    '''
    stories = []
    for i in range(num_slides):
        story = chatgpt(bearer_token, prompt, system_prompt)
        app.logger.info('story: ' + story)
        stories.append(story)
    '''
    chatgpt_prompt = f"{prompt}. Generate {num_slides} responses, each of not more than ${num_chars} characters each. Respond with a JSON array of simple string literals, surrounded by square brackets, suitable to be parsed into JSON, with no backticks, quotation marks or other surrounding text."
    stories = chatgpt(bearer_token, chatgpt_prompt, system_prompt)
    app.logger.info('stories: ' + stories)
    stories = json.loads(stories)
    return stories


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


def put_text_on_image(img, text, web_url="", phone_number="", font_size=32, text_color=(0, 0, 0, 200), bg_color=(255, 255, 255, 128)):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil, 'RGBA')

    # Use a Unicode compatible font
    font_path = "./resources/NotoSansDevanagari-Bold.ttf"
    font = ImageFont.truetype(font_path, int(16))

    # top right
    watermark_text = phone_number
    wm_text_width, wm_text_height = font.getsize(watermark_text)
    wm_text_x = img_pil.width - wm_text_width - 10
    wm_text_y = 10
    draw.text((wm_text_x, wm_text_y), watermark_text, font=font, fill=(255, 255, 255))

    # top left
    watermark_text = web_url
    wm_text_x = 10
    wm_text_y = 10
    draw.text((wm_text_x, wm_text_y), watermark_text, font=font, fill=(255, 255, 255))

    # Reset font size
    font = ImageFont.truetype(font_path, int(font_size))
    
    # Text wrapping
    wrapped_text = wrap_text(text, font, img_pil.width - 20)
    # Calculate line height based on the tallest character 'Mg' and add padding
    line_height = font.getsize('Mg')[1] + 10

    # Adjust text_y_start to ensure all text fits in the image
    total_text_height = line_height * len(wrapped_text)
    text_y_start = img_pil.height - total_text_height - 20  # Subtract 10 pixels for the gap

    # Check if the text height exceeds the image height
    if total_text_height > img_pil.height:
        app.logger.error("Warning: Text exceeds image height, consider resizing the image or reducing font size.")

    # Draw a single rectangle background for all lines of text
    bg_rect_top_left = (0, text_y_start - 5)
    bg_rect_bottom_right = (img_pil.width, text_y_start + total_text_height + 5)
    draw.rectangle([bg_rect_top_left, bg_rect_bottom_right], fill=bg_color)

    # Draw text for each line
    for i, line in enumerate(wrapped_text):
        text_width, _ = draw.textsize(line, font=font)
        text_x = (img_pil.width - text_width) / 2  # Center the text horizontally
        draw.text((text_x, text_y_start + i * line_height), line, font=font, fill=text_color)

    # Convert back to numpy array
    return np.array(img_pil)


def ease_in_out(x):
    if x < 0.5:
        return 2 * x * x
    else:
        return -1 + (4 - 2 * x) * x


def generate_zoom_pan_frames(np_img, num_frames):
    height, width = np_img.shape[:2]
    zoom_scale = 2.0  # End with a 2x zoom
    start_x, start_y = 0, 0
    end_x, end_y = int(width * 0.3), int(height * 0.3)

    for i in range(num_frames):
        t = i / (num_frames - 1)  # Normalized time from 0 to 1
        eased_t = ease_in_out(t)  # Apply easing function

        scale = 1 + (zoom_scale - 1) * eased_t
        scaled_frame = cv2.resize(np_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_height, scaled_width = scaled_frame.shape[:2]

        current_x = int(start_x + (end_x - start_x) * eased_t)
        current_y = int(start_y + (end_y - start_y) * eased_t)
        x1 = max(0, min(current_x, scaled_width - width))
        y1 = max(0, min(current_y, scaled_height - height))

        cropped_frame = scaled_frame[y1:y1 + height, x1:x1 + width]
        yield cropped_frame


def create_slideshow(images, stories, audio, filename, web_url, phone_number, zoom_pan=True, slide_duration=10, fadein_duration=2, fadeout_duration=2, fps=24):
    clips = []
    for img, story in zip(images, stories):
        np_img = np.array(img)
        if (zoom_pan):
            for frame in generate_zoom_pan_frames(np_img, int(slide_duration * fps)):
                processed_frame = put_text_on_image(frame, story, web_url, phone_number)
                clips.append(processed_frame)
                del frame  
        else:
            clips.append(put_text_on_image(np_img, story))
            fps = 1/slide_duration
        del np_img  
        collect()

    # Create video from frames
    clip = ImageSequenceClip(clips, fps=fps)

    # Load the background audio
    if audio:
        audio = afx.audio_loop(audio, slide_duration * len(images))
        audio = audio.set_duration(clip.duration)
        audio = audio.audio_fadein(fadein_duration).audio_fadeout(fadeout_duration)
        clip = clip.set_audio(audio)

    # Write the video file
    clip.write_videofile(filename, fps=24, codec='libx264', audio_codec='aac')
    
    # Cleanup
    # audio.close()  
    del clips  
    collect()  


def generate_short_content(bearer_token, prompt, stories, system_prompt, img_prompt, style_prompt, web_url, phone_number, gen_images, zoom_pan, add_music, num_slides, num_chars, slide_duration, filename):
    
    if not len(stories):
        stories = generate_stories(bearer_token, prompt, system_prompt, num_slides, num_chars)
    
    if gen_images:
        images = generate_images(bearer_token, stories, img_prompt, style_prompt)
    else:
        images = load_images(len(stories))
    
    audio = None
    if add_music:
        directory_path = './resources/audio'
        files = get_random_files(directory_path)
        audio = AudioFileClip(os.path.join(directory_path, files[0]))
    
    create_slideshow(images, stories, audio, filename, web_url, phone_number, zoom_pan, slide_duration)
    return stories


@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        # Get bearer token from headers
        bearer_token = request.headers.get('Authorization')

        if not bearer_token:
            return jsonify({"error": "Authorization token is missing"}), 401

        # Get JSON data for prompt and story
        data = request.get_json()
        prompt = data.get('prompt', "")
        stories = data.get('stories', [])
        num_slides = data.get('num_slides', 1)
        gen_images = data.get('gen_images', False)
        zoom_pan = data.get('zoom_pan', True)
        num_chars = data.get('num_chars', 300)
        slide_duration = data.get('slide_duration', 10)
        add_music = data.get('add_music', True)
        system_prompt = data.get('system_prompt', "You are a helpful assistant.")
        img_prompt = data.get('img_prompt', "")
        style_prompt = data.get('style_prompt', "")
        web_url = data.get('web_url', "mahabot.in")
        phone_number = data.get('phone_number', "+919944044840")

        if not prompt and not len(stories):
            return jsonify({"error": "Prompt or stories must be provided"}), 400

        # Generate a unique filename for the video
        unique_id = str(uuid.uuid4())
        filename = f"./output/videos/{unique_id}.mp4"

        # Generate video based on the bearer token, prompt and/or stories
        stories = generate_short_content(bearer_token, prompt, stories, system_prompt, img_prompt, style_prompt, web_url, phone_number, gen_images, zoom_pan, add_music, num_slides, num_chars, slide_duration, filename)

        # Check if the video file has been created and exists

        # Generate URL for the created video
        # video_url = url_for('get_video', video_id=unique_id, _external=True)
        video_url = f"https://{request.host}/videos/{unique_id}.mp4"
        return jsonify({"video_url": video_url, "stories": stories})
    
    except Exception as e:
        # Log the exception, could be more detailed depending on the logging setup
        app.logger.error(f"Failed to generate video: {str(e)}")
        # Return a generic error message and a 500 Internal Server Error status code
        return jsonify({"error": "An error occurred while processing your request"}), 500

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
    app.run(debug=True, host='0.0.0.0', port=port)
