import os

import cv2

# SETTINGS
class_id = 1  # 0=normal, 1=shaking, 2=hitting
video_path = os.path.join('/media/nvidia/E8C3-FB24/VA Only Zip/C/4C-Cam003.mp4')
crop_times = 0  # for the same video, make sure don't duplicate with
output_folder = "bb-dataset-cropped-upper/images_new"
start_time = 0  # seconds

# Variables to store mouse cursor position and click coordinates
mouse_x = 0
mouse_y = 0
click_coordinates = []

# Extract the filename
video_name = os.path.basename(video_path).split('.')[0]


# Mouse event handler
def mouse_event(event, x, y, flags, param):
    global crop_times

    if event == cv2.EVENT_LBUTTONUP:
        click_coordinates.append((x, y))
        if len(click_coordinates) == 2:
            # Crop the past 4 frames based on the two mouse clicks
            x1, y1 = click_coordinates[0]
            x2, y2 = click_coordinates[1]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            if width * 2 < height:  # if w-h ratio is smaller than 1:2, crop the upper half
                height //= 2
                center_y -= (height // 2 - height // 10)

            side_length = max(width, height)
            half_side_length = side_length // 2
            x_min = center_x - half_side_length
            x_max = center_x + half_side_length
            y_min = center_y - half_side_length
            y_max = center_y + half_side_length

            save_folder = os.path.join(
                output_folder, f'cls{class_id}_vid{video_name}_ppl{crop_times}')
            while os.path.exists(save_folder):
                crop_times += 1
                save_folder = os.path.join(
                    output_folder, f'cls{class_id}_vid{video_name}_ppl{crop_times}')
            os.makedirs(save_folder)

            for i in range(max(0, len(past_frames) - 8), len(past_frames)):
                image = past_frames[i]
                cropped_image = image[y_min:y_max, x_min:x_max]

                save_path = os.path.join(save_folder, f'img_{i + 1:03}.jpg')
                if cv2.imwrite(save_path, cropped_image):
                    print(f'Image saved to {save_path}')
                else:
                    print(f'Failed to save {save_path}')
            click_coordinates.clear()
            crop_times += 1
    elif event == cv2.EVENT_MOUSEMOVE:
        # Update the mouse cursor position
        global mouse_x, mouse_y
        mouse_x = x
        mouse_y = y


# Read the video file
cap = cv2.VideoCapture(video_path)

# starts from particular frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 25)

# Check if video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Create a window to display the video
cv2.namedWindow('Video')

# Set the mouse event handler
cv2.setMouseCallback('Video', mouse_event)

# List to store past frames
past_frames = []

# Read the current frame
ret, frame = cap.read()
if ret:
    img = frame.copy()
    # Store the current frame in the past frames list
    past_frames.append(img)
else:
    exit(1)

while True:
    # Draw lines representing the mouse cursor
    # Horizontal line
    cv2.line(frame, (0, mouse_y),
             (frame.shape[1], mouse_y), (0, 255, 0), 1)
    # Vertical line
    cv2.line(frame, (mouse_x, 0),
             (mouse_x, frame.shape[0]), (0, 255, 0), 1)

    # Display the frame
    cv2.imshow('Video', frame)

    # Check for 'q' key press to exit or 'n' key press to go to the next frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        # Read the current frame
        ret, frame = cap.read()
        if ret:
            img = frame.copy()

            # Store the current frame in the past frames list
            past_frames.append(img)

            # Pop frames if there are more than 4 frames in the past_frames list
            if len(past_frames) > 8:
                past_frames.pop(0)
        else:
            break
    frame = img.copy()

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
