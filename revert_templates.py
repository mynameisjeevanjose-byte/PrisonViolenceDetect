import os

basedir = os.path.abspath(os.path.dirname(__file__))
templates_dir = os.path.join(basedir, 'templates')

if not os.path.exists(templates_dir):
    print(f"Error: Could not find templates folder at {templates_dir}")
    exit()

renamed_count = 0
for filename in os.listdir(templates_dir):
    if 'jailer' in filename.lower() and filename.lower().endswith('.html'):
        old_path = os.path.join(templates_dir, filename)
        new_filename = filename.lower().replace('jailer', 'jailor')
        new_path = os.path.join(templates_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"Reverted: {filename} -> {new_filename}")
        renamed_count += 1

print(f"Done! Reverted {renamed_count} files.")