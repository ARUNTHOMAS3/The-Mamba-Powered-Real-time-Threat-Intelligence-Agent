"""
SAFE Clean Outputs Script

Safely removes only generated outputs from whitelisted directories.
Includes path validation to prevent accidental deletion of important files.
"""

import os
import glob

# CRITICAL: Whitelist of directories that can be cleaned
ALLOWED_DIRS = ["outputs", "logs"]

def is_safe_to_delete(filepath):
    """
    Check if a file is safe to delete based on directory whitelist.
    
    Args:
        filepath (str): Path to check
        
    Returns:
        bool: True if safe to delete, False otherwise
    """
    # Normalize path
    filepath = os.path.normpath(filepath)
    
    # Check if file is in an allowed directory
    for allowed_dir in ALLOWED_DIRS:
        if filepath.startswith(os.path.normpath(allowed_dir) + os.sep):
            return True
    
    return False

def clean_directory(directory, patterns, description):
    """
    Clean files matching patterns from a directory.
    
    Args:
        directory (str): Directory path
        patterns (list): List of glob patterns to match
        description (str): Description of what's being cleaned
    """
    # CRITICAL SAFETY CHECK
    dir_basename = os.path.basename(os.path.normpath(directory))
    assert dir_basename in ALLOWED_DIRS, \
        f"❌ SAFETY VIOLATION: Directory '{directory}' is not in whitelist {ALLOWED_DIRS}"
    
    os.makedirs(directory, exist_ok=True)
    
    files_deleted = 0
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        files = glob.glob(full_pattern)
        
        for file in files:
            # Double-check safety
            if not is_safe_to_delete(file):
                print(f"  [SKIP] File outside whitelist: {file}")
                continue
                
            try:
                os.remove(file)
                print(f"  [OK] Deleted: {file}")
                files_deleted += 1
            except Exception as e:
                print(f"  [FAIL] Failed to delete {file}: {e}")
    
    if files_deleted == 0:
        print(f"  (No {description} files found)")
    else:
        print(f"  Total deleted: {files_deleted} file(s)")

def main():
    print("\n" + "="*60)
    print("SAFE CLEAN OUTPUTS - Preparing for Fresh Experiment")
    print("="*60 + "\n")
    
    print(f"✓ Safety: Only files in {ALLOWED_DIRS} can be deleted\n")
    
    # Outputs directory
    print("[1/2] Cleaning outputs/ directory...")
    clean_directory(
        directory="outputs",
        patterns=["*.pth", "*.pt", "*.json", "*.pkl", "*.csv", "*.txt"],
        description="output"
    )
    
    # Logs directory (if exists)
    print("\n[2/2] Cleaning logs/ directory...")
    clean_directory(
        directory="logs",
        patterns=["*"],
        description="log"
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] Cleanup Complete - Ready for fresh experiment")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Confirm before proceeding
    print("\nThis will delete model checkpoints, metrics, and logs from WHITELISTED directories only.")
    print(f"Whitelisted directories: {ALLOWED_DIRS}")
    response = input("Continue? [y/N]: ").strip().lower()
    
    if response == 'y' or response == 'yes':
        main()
    else:
        print("Cleanup cancelled.")
