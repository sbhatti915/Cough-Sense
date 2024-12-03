from pydub import AudioSegment
import numpy as np

# Load the audio file
audio_file_path = 'your_audio_file.mp3'  # Replace with your audio file path
audio = AudioSegment.from_file(audio_file_path)

# Convert to mono for processing if needed
audio = audio.set_channels(1)

# Define a function to find the most energetic 1-second segment
def find_most_energetic_segment(audio, segment_duration=1000):
    step = segment_duration // 2
    max_energy = 0
    best_segment = None

    for start in range(0, len(audio) - segment_duration + 1, step):
        segment = audio[start:start + segment_duration]
        energy = np.sum(np.array(segment.get_array_of_samples()) ** 2)
        if energy > max_energy:
            max_energy = energy
            best_segment = segment

    return best_segment

def save_segment(segment, output_path, format):
    segment.export(output_path, format=format)
    print(f"Segment saved to: {output_path}")
    

# Extract the most energetic 1-second segment
important_segment = find_most_energetic_segment(audio)

# Save the most important segment
output_path = 'important_segment.mp3'  # Replace with your desired output path
important_segment.export(output_path, format='mp3')

print(f"Important segment saved to: {output_path}")