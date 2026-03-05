import sys
import os

# Now inside backend/, so we add the parent directory to sys.path
# so that `import backend.xxx` continues to work.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
