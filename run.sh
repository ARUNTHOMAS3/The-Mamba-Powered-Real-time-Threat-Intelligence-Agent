#!/usr/bin/env bash
# Simple runner for common tasks
case "$1" in
    dashboard)
        python src/dashboard/app.py
        ;;
    test)
        echo "No tests yet"
        ;;
    *)
        echo "Usage: ./run.sh {dashboard|test}"
        ;;
esac
