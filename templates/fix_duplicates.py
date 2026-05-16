import os

def clean():
    # Files that should only exist in templates/, not in root
    files = ['jailor_attendance.html', 'superintendent_attendance.html']
    base = os.path.abspath(os.path.dirname(__file__))
    
    for f in files:
        path = os.path.join(base, f)
        if os.path.exists(path):
            print(f"Removing duplicate file from root: {path}")
            os.remove(path)
        else:
            print(f"File not found in root (Good): {path}")

if __name__ == "__main__":
    clean()