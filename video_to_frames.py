import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Construct the filename for the frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")

        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Finished extracting frames.")

def main():

    video_path = input("Enter the path to the video file: ")
    output_folder = input("Enter the folder to save the frames: ")

    extract_frames(video_path, output_folder)

if __name__ == "__main__":
    main()
