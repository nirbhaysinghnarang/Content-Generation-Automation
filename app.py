from flask import Flask, request, send_file, jsonify, url_for
import uuid
import requests
from PIL import Image
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, ImageSequenceClip, AudioFileClip
import io
import numpy as np
import os
import cv2
from PIL import Image, ImageFont, ImageDraw

app = Flask(__name__)

def generate_short_content(bearer_token, prompt, story, filename):

    images = generate_images(bearer_token, prompt, num_images=3)
    create_slideshow(images, story, filename)

def wrap_text(text, font, max_width):
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and font.getsize(line + words[0])[0] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line)
    return lines


# Function to generate images using OpenAI's DALL-E API
def generate_images(bearer_token, prompt, num_images=3):
    images = []
    headers = {
      'Authorization': bearer_token

    }
    payload = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "model": "dall-e-3"
    }

    for i in range(num_images):
      response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, json=payload)
      if response.status_code == 200:
          data = response.json()
          for img_data in data['data']:
              img_bytes = requests.get(img_data['url']).content
              img = Image.open(io.BytesIO(img_bytes))
              images.append(img)
      else:
          print(f"Failed to generate images: {response.text}")
    return images


def put_text_on_image(img, text, font_scale=2, font_thickness=3, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Use a Unicode compatible font
    font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"  # Adjusted path for bold font
    font = ImageFont.truetype(font_path, int(32 * font_scale))  # Adjust font size as needed

    # Text wrapping
    wrapped_text = wrap_text(text, font, img_pil.width - 20)  # Adjust wrapping to the width of the image minus some padding
    line_height = font.getsize('hg')[1] + 10  # Calculate line height based on font size

    text_y_start = img_pil.height - (line_height * len(wrapped_text)) - 100  # Start drawing text from this Y-coordinate

    # Draw a rectangle background for each line of text
    for i, line in enumerate(wrapped_text):
        line_width, line_height = font.getsize(line)
        bg_rect_top_left = (0, text_y_start + i * line_height - 5)
        bg_rect_bottom_right = (img_pil.width, text_y_start + (i + 1) * line_height + 5)
        draw.rectangle([bg_rect_top_left, bg_rect_bottom_right], fill=bg_color)

        text_width, text_height = draw.textsize(line, font=font)
        text_x = (img_pil.width - text_width) / 2  # Center the text horizontally
        draw.text((text_x, text_y_start + i * line_height), line, font=font, fill=text_color)

    # Convert back to numpy array
    return np.array(img_pil)

def create_slideshow(images, story, filename, duration=3, fadein_duration=1, fadeout_duration=1):
    # Prepare clips list
    clips = []

    for img in images:
        # Convert PIL image to numpy array
        np_img = np.array(img)

        # Add text onto the image using OpenCV
        np_img_with_text = put_text_on_image(np_img, story)

        # Append the modified image to the clips list
        clips.append(np_img_with_text)

    # Create a clip from image sequences
    clip = ImageSequenceClip(clips, fps=1/duration)

    # Load the background audio
    audio = AudioFileClip('017941_unknown-54945.mp3')

    audio = audio.audio_fadein(fadein_duration).audio_fadeout(fadeout_duration)

    # Set the audio of the video clip. If the audio is longer than the video, it will be trimmed
    # If the audio is shorter, it will loop
    clip = clip.set_audio(audio.set_duration(clip.duration))

    # Write the clip to a file
    clip.write_videofile(filename, fps=24, codec='libx264', audio_codec='aac')



@app.route('/generate_video', methods=['POST'])
def generate_video():
    # Get bearer token from headers
    bearer_token = request.headers.get('Authorization')

    # Generate a unique filename for the video
    unique_id = str(uuid.uuid4())
    video_filename = f"{unique_id}.mp4"



    if not bearer_token:
        return jsonify({"error": "Authorization token is missing"}), 401

    # Get JSON data for prompt and story
    data = request.json
    prompt = data.get('prompt', "")
    story = data.get('story', "")

    print('bearer_token ',bearer_token)
    print('story ',story)
    print('prompt ',prompt)

    # Check for missing prompt or story
    if not prompt or not story:
        return jsonify({"error": "Both prompt and story must be provided"}), 400



    # Generate video based on the bearer token, prompt, and story
    generate_short_content(bearer_token, prompt, story, video_filename)

    # Check if the video file has been created and exists

    # Generate URL for the created video
    video_url = url_for('get_video', video_id=unique_id, _external=True)
    return jsonify({"video_url": video_url})

@app.route('/video/<video_id>')
def get_video(video_id):
    video_path = f"{video_id}.mp4"
    try:
        return send_file(video_path, as_attachment=True, attachment_filename=video_path)
    except Exception as e:
        return video_path


@app.route('/')
def home():
    return "Welcome to the Video Generator API!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
