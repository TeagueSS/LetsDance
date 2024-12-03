import os
import subprocess

# Base download directory
base_download_dir = "/Volumes/Samsung/CS450"

# Function to convert video to MP4
def convert_to_mp4(directory):
    # Scan the directory for video files
    for root, _, files in os.walk(directory):
        for file in files:
            # Full path of the file
            file_path = os.path.join(root, file)

            # Skip files that are already MP4 or MP3
            if file.endswith(".mp4") or file.endswith(".mp3"):
                print(f"Skipping file: {file_path}")
                continue

            # Determine the output file path
            output_file = os.path.splitext(file_path)[0] + ".mp4"

            # Convert to MP4 using ffmpeg
            print(f"Converting {file_path} to {output_file}")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-i", file_path, "-c:v", "libx264", "-c:a", "aac",
                        "-strict", "experimental", output_file
                    ],
                    check=True
                )
                print(f"Converted successfully: {output_file}")

                # Optionally delete the original file
                os.remove(file_path)
                print(f"Deleted original file: {file_path}")

            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_path}: {e}")

# Run the conversion
convert_to_mp4(base_download_dir)
