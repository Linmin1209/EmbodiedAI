import os
import re
import csv
import sys
import ast

# To make imports work, run this script from the project root directory.
# For example: python scripts/analyze_logs.py
from simulator.core.dataset import DatasetLoader 

def analyze_logs(log_dir, output_csv_path):
    """
    Parses log files to find successful tasks, retrieves their configurations,
    and saves the information to a CSV file.

    Args:
        log_dir (str): The directory containing the log files.
        output_csv_path (str): The path to save the output CSV file.
    """
    print("Initializing DatasetLoader...")
    try:
        loader = DatasetLoader(
            root_dir="/data1/linmin/EmbodiedAI/resource/datasets/all_task",
            scene_path="/data1/linmin/NPC/hssd_test_scene",
            robot_path="/data1/linmin/EmbodiedAI/resource/robots/stretch/stretch_pos.usd",
            headless=True
        )
        print(f"DatasetLoader initialized. Found {len(loader)} total configs.")
    except Exception as e:
        print(f"Error initializing DatasetLoader: {e}")
        print("Please ensure you are running this script from the project's root directory.")
        return

    successful_tasks = []

    print(f"Processing log files from: {log_dir}")
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at '{log_dir}'")
        return
        
    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith(".log"):
            log_path = os.path.join(log_dir, filename)

            last_line = ""
            with open(log_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                last_line = lines[-1].strip()
            
            if not last_line:
                continue

            try:
                # Use regex to capture the Info and Done parts of the log line
                match = re.search(r"Info: (.*), Done: (.*)", last_line)
                if not match:
                    continue

                info_str = match.group(1)
                done_str = match.group(2)
                
                info_list = ast.literal_eval(info_str)
                done_list = ast.literal_eval(done_str)

                if not info_list:
                    continue

                info_dict = info_list[0]
                stage_success = info_dict.get('stage_success')
                done_val = done_list[0] if done_list else False

                if done_val and stage_success == [[True, True]]:
                    id_match = re.search(r"log_(\d+)\.log", filename)
                    if id_match:
                        task_id = int(id_match.group(1))

                        if task_id < len(loader):
                            cfg = loader[task_id]
                            task_path = cfg.task.task_path
                            successful_tasks.append({'id': task_id, 'task_path': task_path})
                        else:
                            print(f"Warning: Task ID {task_id} from {filename} is out of bounds for DatasetLoader (size: {len(loader)}).")

            except (ValueError, SyntaxError, IndexError):
                # We can ignore lines that don't match the expected format.
                continue

    if successful_tasks:
        print(f"\nFound {len(successful_tasks)} successful tasks in total.")
        successful_tasks.sort(key=lambda x: x['id'])
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'task_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(successful_tasks)
        print(f"Saved successful tasks to {output_csv_path}")
    else:
        print("\nNo successful tasks found.")

if __name__ == "__main__":
    log_directory = "/data1/linmin/EmbodiedAI/eval_data_logs"
    csv_output_file = "successful_tasks.csv" 
    
    analyze_logs(log_directory, csv_output_file) 