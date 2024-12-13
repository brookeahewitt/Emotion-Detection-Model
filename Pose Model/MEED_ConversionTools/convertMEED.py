import os
import shutil
import argparse
import cv2
from IPython.core.interactiveshell import InteractiveShell
import sys


def move_data_to_target_folder(paths, target_folder, actorname):
    # Create target subdirectories for video and pose if they don't exist
    video_folder = os.path.join(target_folder, 'videos')
    pose_folder = os.path.join(target_folder, 'pose')

    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(pose_folder, exist_ok=True)


    for folder_path in paths:
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_path} as it is not a valid folder.")
            continue
        
        # Walk through each folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Get the file extension
                _, ext = os.path.splitext(file)
                    
                if ext.lower() == '.png':
                    base_filename = file.lower()  # Lowercase for case-insensitive comparison
                    if base_filename.startswith('left'):
                        target_folder_video = os.path.join(video_folder, 'left'+actorname)
                    elif base_filename.startswith('right'):
                        target_folder_video = os.path.join(video_folder, 'right'+actorname)
                    elif base_filename.startswith('front'):
                        target_folder_video = os.path.join(video_folder, 'front'+actorname)
                    else:
                        target_folder_video = video_folder  # Default to the main video folder if no prefix match

                    # Ensure the appropriate subfolder exists
                    os.makedirs(target_folder_video, exist_ok=True)
                    target_path = os.path.join(target_folder_video, file)

                # Handle .json files
                elif ext.lower() == '.json':
                    base_filename = file.lower()  # Lowercase for case-insensitive comparison
                    if base_filename.startswith('left'):
                        target_folder_json = os.path.join(pose_folder, 'left' + actorname + '_json')
                    elif base_filename.startswith('right'):
                        target_folder_json = os.path.join(pose_folder, 'right' + actorname + '_json')
                    elif base_filename.startswith('front'):
                        target_folder_json = os.path.join(pose_folder, 'front' + actorname + '_json')
                    else:
                        target_folder_json = os.path.join(pose_folder, actorname + '_json')  # Default with '_json' appended

                    # Ensure the appropriate subfolder exists
                    os.makedirs(target_folder_json, exist_ok=True)
                    target_path = os.path.join(target_folder_json, file)

                else:
                    print(f"Skipping {file_path} as it is not a png or json file.")
                    continue
            
                try:
                    shutil.move(file_path, target_path)  # Use shutil.copy() if you prefer to copy instead of move
                    print(f"Moved {file_path} to {target_path}")
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")

def create_video_from_folder(image_folder, output_video_path, frame_rate=30):
    # Get a list of all .png files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Sort the image files by filename (you can adjust the sorting logic)
    images.sort()

    # Check if there are any images to process
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Read the first image to get the dimensions for the video
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Loop through the images and add them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer and close the file
    video.release()
    cv2.destroyAllWindows()

    print(f"Video created at {output_video_path}")



def create_videos(base_directory, frame_rate=30):
    print(base_directory)
    for folder_name in os.listdir(base_directory):
        image_folder = os.path.join(base_directory, folder_name)
        print(image_folder)
        if os.path.isdir(image_folder):
            # Define the output video path based on the folder name
            output_video_path = os.path.join(base_directory, f"{folder_name}.mp4")
            print(f"Making video for{output_video_path}")
            create_video_from_folder(image_folder, output_video_path, frame_rate)



def create_3D_pose(pose_creation_directory):
    try:
        original_directory = os.getcwd()
        print(f"Changing directory to {pose_creation_directory} to create 3D Pose...")
        os.chdir(pose_creation_directory)  

        # Create an instance of the interactive shell
        shell = InteractiveShell.instance()

        # A set of commands to execute
        commands = [
            "from Pose2Sim import Pose2Sim",
            "Pose2Sim.synchronization()",
            "Pose2Sim.triangulation()",
        ]

        for command in commands:
            shell.run_cell(command)
        
        shell.exit_now()
        print("IPython commands finished.")
        print("3D Pose Created!") 
        files = os.listdir(pose_creation_directory+"videos")
        os.chdir(original_directory)
    except Exception as e:
        print(f"Error while changing directory or creating 3D pose: {e}")
        #sys.exit(1)


def move_files(source_path, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        print(f"Target directory '{target_dir}' does not exist. Creating it.")
        os.makedirs(target_dir)
    
    # Move the source to the target
    try:
        if os.path.isfile(source_path):
            shutil.move(source_path, target_dir)
            print(f"Moved file '{source_path}' to '{target_dir}'")
        
        elif os.path.isdir(source_path):
            # Join the target directory with the source's directory name
            destination_path = os.path.join(target_dir, os.path.basename(source_path))
            shutil.move(source_path, destination_path)
            print(f"Moved directory '{source_path}' to '{destination_path}'")

    except Exception as e:
        print(f"Error occurred while moving: {e}")




# Main function to walk through folders and process matching folders
def process_folders(source, target, actor):

    for left_dir in os.listdir(source):
        if left_dir.startswith('left_'):
            left_dir_path = os.path.join(source, left_dir)
            print(f"Processing {left_dir_path}...")
            for name in os.listdir(left_dir_path):
                if name == '.DS_Store':
                    continue
                matching_paths = []
                base = name.split('_')[1]
                
                left_path = os.path.join(left_dir_path, name)
                if left_path not in matching_paths:
                    matching_paths.append(left_path)

                # Check the right folder
                right_dir_path = os.path.join(source, 'right_' + actor)
                if os.path.isdir(right_dir_path):
                    for right_name in os.listdir(right_dir_path):
                        if right_name.split('_')[1] == base:
                            right_path = os.path.join(right_dir_path, right_name)
                            if right_path not in matching_paths:
                                matching_paths.append(right_path)  # Append directly

                # Check the front folder
                front_dir_path = os.path.join(source, 'front_' + actor)
                if os.path.isdir(front_dir_path):
                    for front_name in os.listdir(front_dir_path):
                        if front_name.split('_')[1] == base:
                            front_path = os.path.join(front_dir_path, front_name)
                            if front_path not in matching_paths:
                                matching_paths.append(front_path)
                
                print(matching_paths)

                if len(matching_paths) == 3:
                    move_data_to_target_folder(matching_paths,target,base) #Moves all data to appropriate place for OpenSim Process

                    print("Creating Videos")
                    target_video_path = os.path.join(target, 'videos') 
                    create_videos(target_video_path)

                    print("Creating 3D Pose")
                    create_3D_pose(target)
                    print("Resetting OpenSim folder structure")

                    #Move video, poses, pose-3d, sync-pose back
                    source_basename_path = os.path.join(source,base)
                    source_video_path = os.path.join(source_basename_path, 'videos')
                    move_files(target_video_path,source_video_path) 
                    source_pose_path = os.path.join(source_basename_path,'pose')
                    target_pose_path = os.path.join(target,'pose')
                    move_files(target_pose_path,source_pose_path)
                    source_3dpose_path = os.path.join(source_basename_path,'pose-3d')
                    target_3dpose_path = os.path.join(target,'pose-3d')
                    move_files(target_3dpose_path,source_3dpose_path)
                    source_3dpose_path = os.path.join(source_basename_path,'pose-sync')
                    target_3dpose_path = os.path.join(target,'pose-sync"')
                    move_files(target_3dpose_path,source_3dpose_path)


                    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert MEED into 3D Poses")
    parser.add_argument('data_folder', type=str, help="The root directory with all data for an actor.")
    parser.add_argument('openpose_folder', type=str, help="The directory for OpenPose is being processed.")
    parser.add_argument('actor',type=str, help= "Actor being processed")

    args = parser.parse_args()
    current_dir = os.getcwd()
    d_folder = os.path.join(current_dir, args.data_folder)
    p_folder = os.path.join(current_dir, args.openpose_folder)

    # Process the folders with the provided source and target paths
    process_folders(d_folder, p_folder,args.actor)

    print("Done!")
