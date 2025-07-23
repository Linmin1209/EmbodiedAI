import os
import csv
import shutil
from pathlib import Path

def copy_successful_tasks(csv_file_path, source_base_dir, target_base_dir):
    """
    Reads the CSV file containing successful tasks and copies the corresponding
    directories to a new location.
    
    Args:
        csv_file_path (str): Path to the CSV file containing successful tasks
        source_base_dir (str): Base directory of the source datasets
        target_base_dir (str): Base directory where to copy the successful tasks
    """
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return
    
    # Create target base directory if it doesn't exist
    os.makedirs(target_base_dir, exist_ok=True)
    
    successful_copies = 0
    failed_copies = 0
    
    print(f"Reading successful tasks from: {csv_file_path}")
    print(f"Source base directory: {source_base_dir}")
    print(f"Target base directory: {target_base_dir}")
    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            task_id = row['id']
            task_path = row['task_path']
            
            try:
                # Get the parent directory of the task_path
                # task_path format: /data1/linmin/EmbodiedAI/resource/datasets/all_task/104862681_172226874/7/processed_config.json
                # We want to copy: /data1/linmin/EmbodiedAI/resource/datasets/all_task/104862681_172226874/7
                
                task_path_obj = Path(task_path)
                parent_dir = task_path_obj.parent  # This gets the directory containing processed_config.json
                
                # Verify the parent directory exists
                if not parent_dir.exists():
                    print(f"Warning: Parent directory not found for task {task_id}: {parent_dir}")
                    failed_copies += 1
                    continue
                
                # Create the relative path from source base directory
                try:
                    relative_path = parent_dir.relative_to(Path(source_base_dir))
                except ValueError:
                    print(f"Warning: Task path {parent_dir} is not under source base directory {source_base_dir}")
                    failed_copies += 1
                    continue
                
                # Create target path
                target_path = Path(target_base_dir) / relative_path
                
                # Create target directory structure
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the directory
                if target_path.exists():
                    print(f"Warning: Target directory already exists for task {task_id}: {target_path}")
                    # Remove existing directory and copy fresh
                    shutil.rmtree(target_path)
                
                shutil.copytree(parent_dir, target_path)
                print(f"Successfully copied task {task_id}: {parent_dir} -> {target_path}")
                successful_copies += 1
                
            except Exception as e:
                print(f"Error copying task {task_id}: {e}")
                failed_copies += 1
    
    print(f"\nCopy operation completed:")
    print(f"Successfully copied: {successful_copies} tasks")
    print(f"Failed to copy: {failed_copies} tasks")
    print(f"Total tasks processed: {successful_copies + failed_copies}")

if __name__ == "__main__":
    csv_file_path = "/data1/linmin/EmbodiedAI/tests/successful_tasks.csv"
    source_base_dir = "/data1/linmin/EmbodiedAI/resource/datasets/all_task"
    target_base_dir = "/data1/linmin/EmbodiedAI/resource/datasets/all_val_task"
    
    copy_successful_tasks(csv_file_path, source_base_dir, target_base_dir) 