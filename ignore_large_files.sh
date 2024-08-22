#!/bin/bash

# Define the size threshold in bytes (100 MB = 104857600 bytes)
THRESHOLD=104857600

# Find all files larger than the threshold
find . -type f -size +${THRESHOLD}c | while read -r FILE; do
    # Check if the file is already in .gitignore
    if ! grep -q "^$FILE$" .gitignore; then
        # Add the large file to .gitignore
        echo "Ignoring large file: $FILE"
        echo "$FILE" >> .gitignore
    fi
done

# Stage the .gitignore file for commit
