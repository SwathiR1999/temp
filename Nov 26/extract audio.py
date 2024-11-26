from moviepy.editor import VideoFileClip

def extract_audio(video_file, output_audio_file):
    """
    Extracts audio from an MP4 video file and saves it as a .wav file.

    :param video_file: Path to the input MP4 video file.
    :param output_audio_file: Path to save the extracted audio in .wav format.
    """
    try:
        # Load the video file
        video = VideoFileClip(video_file)
        
        # Extract the audio
        audio = video.audio
        
        if audio is not None:
            # Write the audio to a .wav file
            audio.write_audiofile(output_audio_file)
            print(f"Audio successfully extracted and saved to {output_audio_file}")
        else:
            print("No audio found in the video.")
        
        # Close the video and audio clips
        audio.close()
        video.close()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
video_path = "input_video.mp4"  # Replace with your MP4 file path
audio_path = "output_audio.wav"  # Desired output path for the .wav file
extract_audio(video_path, audio_path)
