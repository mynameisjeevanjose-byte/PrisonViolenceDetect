import os

basedir = os.path.abspath(os.path.dirname(__file__))
templates_dir = os.path.join(basedir, 'templates')

if not os.path.exists(templates_dir):
    print(f"Error: Could not find templates folder at {templates_dir}")
    exit()

renamed_count = 0
for filename in os.listdir(templates_dir):
    if 'jailor' in filename and filename.endswith('.html'):
        old_path = os.path.join(templates_dir, filename)
        new_filename = filename.replace('jailor', 'jailer')
        new_path = os.path.join(templates_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
        renamed_count += 1

print(f"Done! Renamed {renamed_count} files.")