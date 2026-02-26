import sys
import os

# Manually point to your environment's library folder
lib_path = "/home/ruminator/miniconda3/envs/binf/lib/python3.11/site-packages"
if lib_path not in sys.path:
    sys.path.append(lib_path)

try:
    import MACS3
    print("MACS3 Manual Import: SUCCESS")
except ImportError:
    # Check if the folder even exists on disk
    if os.path.exists(os.path.join(lib_path, "macs3")):
        print("MACS3 exists on disk but failed to import (likely a dependency issue).")
    else:
        print("MACS3 is physically missing from the library folder.")