"""
Understat Data Pipeline Runner
"""

import subprocess
import sys

# Run scripts in order
subprocess.run([sys.executable, "gatherDataGeneral.py"])
subprocess.run([sys.executable, "gatherDataTeam.py"])
subprocess.run([sys.executable, "gatherDataPlayer.py"])
subprocess.run([sys.executable, "cleanDataPlayer.py"])

print("\nPipeline complete!")