import os
import shutil
import sys

def find_data_yaml(directory):
    """Find all data.yaml files in the directory and its subdirectories."""
    yaml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "data.yaml":
                yaml_files.append(os.path.join(root, file))
    return yaml_files

if __name__ == "__main__":
    # Path to the yolov10 directory
    yolov10_dir = os.path.join(os.getcwd(), "yolov10")
    
    if not os.path.exists(yolov10_dir):
        print(f"Error: {yolov10_dir} does not exist")
        sys.exit(1)
    
    # Find all data.yaml files
    yaml_files = find_data_yaml(yolov10_dir)
    
    if not yaml_files:
        print("No data.yaml files found")
        sys.exit(1)
    
    print(f"Found {len(yaml_files)} data.yaml files:")
    for i, file_path in enumerate(yaml_files):
        print(f"{i+1}. {file_path}")
        
        # Read and print the content of each file
        with open(file_path, 'r') as f:
            content = f.read()
            print(f"\nContent of {file_path}:\n{content}\n")
            
        # Copy the file to the current directory with a numbered suffix
        dest_path = f"data_yaml_{i+1}.yaml"
        shutil.copy(file_path, dest_path)
        print(f"Copied to {dest_path}")