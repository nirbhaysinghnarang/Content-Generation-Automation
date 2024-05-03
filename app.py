from flask import Flask, request, send_file, jsonify
import requests
from PIL import Image
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, ImageSequenceClip, AudioFileClip
import io
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

app = Flask(__name__)

# Main function to generate a short content video
def generate_short_content(bearer_token, prompt, story, video_path):
    # Generate images based on the prompt
    images = generate_images(bearer_token, prompt, num_images=3)

    # Create a slideshow video from the images
    create_slideshow(images, story)


# Function to generate images using OpenAI's DALL-E API
def generate_images(bearer_token, prompt, num_images=3):
    images = []
    headers = {
      'Authorization': bearer_token

    }
    payload = {
        "prompt": prompt,
        "n": num_images,
        "size": "1024x1024"
    }

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

def put_text_on_image(img, text, font_scale=1, font_thickness=2, text_color=(0, 0, 0)):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Use a Unicode compatible font, ensure it's bold if needed
    font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"  # Adjusted path for bold font
    font = ImageFont.truetype(font_path, int(32 * font_scale))  # Adjust font size as needed

    # Calculate the text size to position it at the center
    text_size = draw.textsize(text, font=font)
    text_x = (img_pil.width - text_size[0]) / 2  # Center the text horizontally
    text_y = img_pil.height - text_size[1] - 10  # Position the text at the bottom, with a small margin

    # Apply the text onto the image
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Convert back to numpy array
    return np.array(img_pil)


def create_slideshow(images, story, duration=3):
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

    # # Load the background audio
    # audio = AudioFileClip('/content/017941_unknown-54945.mp3')

    # # Set the audio of the video clip. If the audio is longer than the video, it will be trimmed
    # # If the audio is shorter, it will loop
    # clip = clip.set_audio(audio.set_duration(clip.duration))

    # Write the clip to a file
    clip.write_videofile("story_with_subtitles_with_music.mp4", fps=24, codec='libx264', audio_codec='aac')



@app.route('/generate_video', methods=['POST'])
def generate_video():
    # Get bearer token from headers
    bearer_token = request.headers.get('Authorization')

    
    
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

    video_path = "/story_with_subtitles_with_music.mp4"

    try:
        # Assuming generate_short_content uses bearer_token, prompt, story, and video_path
        status = generate_short_content(bearer_token, prompt, story, video_path)
        if status:
            return send_file(video_path, attachment_filename='video.mp4')
        else:
            return jsonify({"error": "Failed to generate video"}), 500
    except Exception as e:
        return str(e), 500

@app.route('/')
def home():
    return "Welcome to the Video Generator API!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
