import subprocess
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to get relative paths
def get_relative_path(relative_path):
    return os.path.join(BASE_DIR, relative_path).replace('/', '\\')

# Path to the Streamlit app
app_path = get_relative_path('app.py')

# Open the Streamlit app programmatically
subprocess.Popen(['streamlit', 'run', app_path])