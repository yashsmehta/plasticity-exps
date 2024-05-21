import csv
import os

def delete_incorrect_lines(file_path):
    """
    Deletes lines from a CSV file that do not have the correct number of entries, considering quoted fields.
    :param file_path: Path to the CSV file.
    """
    num_lines_deleted = 0
    with open(file_path, mode='r', newline='') as read_file:
        reader = csv.reader(read_file)
        rows = list(reader)
    
    if not rows:
        return 0

    num_columns = len(rows[0])

    with open(file_path + '.tmp', mode='w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(rows[0])

        for row in rows[1:]:
            if len(row) == num_columns:
                writer.writerow(row)
            else:
                num_lines_deleted += 1

    os.remove(file_path)
    os.rename(file_path + '.tmp', file_path)
    
    return num_lines_deleted


if __name__ == "__main__":
    directory_path = 'logs/simdata/recoverability/mlp/'
    for filename in os.listdir(directory_path):
        if filename.startswith('exp_') and filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            deleted_count = delete_incorrect_lines(file_path)
            print(f"Deleted {deleted_count} lines from {filename}.")
