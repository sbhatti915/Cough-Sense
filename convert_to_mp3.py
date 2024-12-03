import os
from pydub import AudioSegment

def convert_to_mp3(input_dir, output_dir):
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List .webm and .ogg files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(('.webm', '.ogg'))]
    if not files:
        print(f"No .webm or .ogg files found in the input directory '{input_dir}'.")
        return
    
    for file in files:
        input_file = os.path.join(input_dir, file)
        base_name, ext = os.path.splitext(file)
        output_file = os.path.join(output_dir, f"{base_name}.mp3")
        
        if os.path.exists(output_file):
            print(f"Skipping {file}: {output_file} already exists.")
            continue
        
        print(f"Converting {file} to {output_file}...")
        try:
            # Load and convert the audio file
            audio = AudioSegment.from_file(input_file, format=ext.lstrip('.'))
            audio.export(output_file, format="mp3", bitrate="192k")
            print(f"Conversion complete: {output_file}")
        except Exception as e:
            print(f"Failed to convert {file}. Error: {e}")
    
    print("All conversions are complete.")

if __name__ == "__main__":
    input_dir = '/home/sameer/coughsense/public_dataset'
    output_dir = '/home/sameer/coughsense/data'
    convert_to_mp3(input_dir, output_dir)