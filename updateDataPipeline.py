import asyncio
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return True if successful."""
    print(f"\n{'='*50}")
    print(f" {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error in {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f" Script {script_name} not found")
        return False

def main():
    """Run the complete update pipeline."""
    print(f" Starting Update Data Pipeline")
    print(f" Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_steps = [
        ("updateData.py", "Fetching current season data from Understat API"),
        ("transformTeamData.py", "Transforming and processing team data"),
        ("getTeamEloV2.py", "Computing updated Elo ratings with V2 model")
    ]
    
    success_count = 0
    
    for script, description in pipeline_steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n Pipeline failed at step: {description}")
            print(f" Completed steps: {success_count}/{len(pipeline_steps)}")
            break
    
    if success_count == len(pipeline_steps):
        print(f"\n Pipeline completed successfully!")
        print(f" All {len(pipeline_steps)} steps completed")
        print(f"\n Updated files should now be available:")
        print("  - understat_data/{league}/Team_Results/*.csv (transformed data)")
        print("  - understat_data/{league}/team_elos_v2.csv (updated ratings)")
        print("  - understat_data/{league}/Team_Results/team_elos_timeseries.csv")
    else:
        print(f"\n Pipeline incomplete: {success_count}/{len(pipeline_steps)} steps completed")
    
    print(f"\n Pipeline finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()