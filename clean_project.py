"""
Clean Project - Safe cleanup script for Mamba vs LSTM experiment

This script removes old outputs, processed data, and optionally unused datasets
while preserving raw CICIDS2017 data and all code files.

Usage:
    python clean_project.py                    # Full cleanup with confirmation
    python clean_project.py --dry-run          # Preview without deleting
    python clean_project.py --include-unused   # Also delete unused datasets
"""

import os
import glob
import argparse
from datetime import datetime

# Safety: Never delete these patterns
PROTECTED_PATTERNS = [
    "data/raw/CICIDS2017/*.csv",
    "**/*.py",
    "**/*.md",
    "**/*.txt",  # Except cleanup_log.txt which we manage
]

# Patterns to clean
CLEANUP_TARGETS = {
    "outputs": [
        "outputs/*.pth",
        "outputs/*.pt",
        "outputs/*.json",
        "outputs/*.pkl",
        "outputs/*.csv",
    ],
    "processed_data": [
        "data/processed/*",
    ],
    "unused_datasets": [  # Only deleted with --include-unused
        "data/raw/UNSW-NB15/*",
        "data/raw/old_datasets/*",
    ]
}

def is_protected(filepath):
    """Check if file is protected from deletion."""
    # Always protect CICIDS2017 CSVs
    if "CICIDS2017" in filepath and filepath.endswith(".csv"):
        return True
    
    # Protect code and documentation
    if filepath.endswith((".py", ".md")) and not filepath.endswith("cleanup_log.txt"):
        return True
    
    return False

def collect_files_to_delete(include_unused=False):
    """Collect all files that will be deleted."""
    # CRITICAL: Whitelist of allowed directories
    ALLOWED_TARGETS = ["outputs", "data/processed"]
    if include_unused:
        ALLOWED_TARGETS.append("data/unused_raw")
    
    files_to_delete = []
    
    # Collect outputs and processed data (always)
    for category in ["outputs", "processed_data"]:
        for pattern in CLEANUP_TARGETS[category]:
            for filepath in glob.glob(pattern, recursive=True):
                # CRITICAL SAFETY CHECK: Ensure file is in allowed directory
                normalized_path = os.path.normpath(filepath)
                is_allowed = any(normalized_path.startswith(os.path.normpath(allowed)) 
                                for allowed in ALLOWED_TARGETS)
                
                if not is_allowed:
                    print(f"⚠️  SKIPPING (not in whitelist): {filepath}")
                    continue
                
                if os.path.isfile(filepath) and not is_protected(filepath):
                    files_to_delete.append(filepath)
    
    # Collect unused datasets (optional)
    if include_unused:
        for pattern in CLEANUP_TARGETS["unused_datasets"]:
            for filepath in glob.glob(pattern, recursive=True):
                # CRITICAL SAFETY CHECK
                normalized_path = os.path.normpath(filepath)
                is_allowed = any(normalized_path.startswith(os.path.normpath(allowed)) 
                                for allowed in ALLOWED_TARGETS)
                
                if not is_allowed:
                    print(f"⚠️  SKIPPING (not in whitelist): {filepath}")
                    continue
                
                if os.path.isfile(filepath) and not is_protected(filepath):
                    files_to_delete.append(filepath)
    
    return sorted(files_to_delete)

def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def print_summary(files, dry_run=False):
    """Print summary of files to be deleted."""
    if not files:
        print("\n✓ No files to delete. Project is already clean.\n")
        return False
    
    total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    
    print("\n" + "="*70)
    if dry_run:
        print("DRY RUN - Files that WOULD be deleted:")
    else:
        print("Files to be deleted:")
    print("="*70)
    
    # Group by category
    outputs = [f for f in files if f.startswith("outputs/")]
    processed = [f for f in files if f.startswith("data/processed/")]
    unused = [f for f in files if f.startswith("data/raw/") and not "CICIDS2017" in f]
    
    if outputs:
        print(f"\n[Outputs] ({len(outputs)} files)")
        for f in outputs[:10]:  # Show first 10
            size = format_size(os.path.getsize(f)) if os.path.exists(f) else "?"
            print(f"  - {f} ({size})")
        if len(outputs) > 10:
            print(f"  ... and {len(outputs) - 10} more files")
    
    if processed:
        print(f"\n[Processed Data] ({len(processed)} files)")
        for f in processed[:10]:
            size = format_size(os.path.getsize(f)) if os.path.exists(f) else "?"
            print(f"  - {f} ({size})")
        if len(processed) > 10:
            print(f"  ... and {len(processed) - 10} more files")
    
    if unused:
        print(f"\n[Unused Datasets] ({len(unused)} files)")
        for f in unused[:10]:
            size = format_size(os.path.getsize(f)) if os.path.exists(f) else "?"
            print(f"  - {f} ({size})")
        if len(unused) > 10:
            print(f"  ... and {len(unused) - 10} more files")
    
    print("\n" + "="*70)
    print(f"Total: {len(files)} files, {format_size(total_size)}")
    print("="*70 + "\n")
    
    return True

def log_cleanup(files, log_path="outputs/cleanup_log.txt"):
    """Log deleted files to cleanup log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Cleanup: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Deleted {len(files)} files:\n\n")
        for filepath in files:
            f.write(f"  - {filepath}\n")
        f.write("\n")

def delete_files(files):
    """Delete files and return count of successful deletions."""
    deleted_count = 0
    errors = []
    
    for filepath in files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_count += 1
        except Exception as e:
            errors.append((filepath, str(e)))
    
    if errors:
        print(f"\n⚠️  Errors during deletion:")
        for filepath, error in errors:
            print(f"  - {filepath}: {error}")
    
    return deleted_count

def confirm_deletion():
    """Ask user for confirmation before deleting."""
    while True:
        response = input("Are you sure you want to permanently delete these files? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        else:
            print("Please enter 'yes' or 'no'")

def main():
    parser = argparse.ArgumentParser(
        description="Clean old outputs and datasets from the project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--include-unused",
        action="store_true",
        help="Also delete unused datasets (UNSW-NB15, old_datasets)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with caution)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PROJECT CLEANUP - Mamba vs LSTM Experiment")
    print("="*70)
    
    # Safety reminder
    print("\n✓ CICIDS2017 CSVs are protected and will NOT be deleted")
    print("✓ All Python and Markdown files are protected")
    
    # Collect files
    files_to_delete = collect_files_to_delete(include_unused=args.include_unused)
    
    # Print summary
    has_files = print_summary(files_to_delete, dry_run=args.dry_run)
    
    if not has_files:
        return
    
    # Dry run - exit after showing summary
    if args.dry_run:
        print("This was a DRY RUN. No files were deleted.")
        print("Run without --dry-run to actually delete files.\n")
        return
    
    # Confirm deletion
    if not args.yes:
        if not confirm_deletion():
            print("\n✗ Cleanup cancelled by user.\n")
            return
    
    # Delete files
    print("\nDeleting files...")
    deleted_count = delete_files(files_to_delete)
    
    # Log cleanup
    log_cleanup(files_to_delete)
    
    # Summary
    print(f"\n✓ Cleanup complete!")
    print(f"  - Deleted: {deleted_count} files")
    print(f"  - Log saved to: outputs/cleanup_log.txt\n")

if __name__ == "__main__":
    main()
