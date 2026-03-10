#!/bin/bash

# Define the subject directory
SUBJECTS_DIR="/Volumes/Workspace/FreeSurfer/subjects"
export FS_ALLOW_DEEP=1  # Enable ML routines for recon-all
export OMP_NUM_THREADS=8

# Loop over each subject directory
for SUBJECT_DIR in "$SUBJECTS_DIR"/*; do
    # Check if the item is a directory
    if [ -d "$SUBJECT_DIR" ]; then
        SUBJECT_NAME=$(basename "$SUBJECT_DIR")

        # Define the path to the FreeSurfer MRI file
        FS_MRI="$SUBJECT_DIR/mri/T1.mgz"
        
        # Check if the FreeSurfer MRI file exists
        if [ -f "$FS_MRI" ]; then
            # Loop through all .dfs files in the brainsuite directory
            for DFS_FILE in "$SUBJECT_DIR/brainsuite"/*.dfs; do
                # Check if the .dfs file exists
                if [ -f "$DFS_FILE" ]; then
                    # Generate the output .surf file name based on the .dfs file name
                    OUTPUT_SURF="${DFS_FILE%.dfs}.surf"

                    # Run the mne_convert_surface command
                    echo "Converting $DFS_FILE to surface for subject: $SUBJECT_NAME"
                    # recon-all -s "$SUBJECT_NAME" -autorecon1 -parallel
                    mne_convert_surface --dfs "$DFS_FILE" --mghmri "$FS_MRI" --surfout "$OUTPUT_SURF"
                    
                    # Check if the conversion was successful
                    if [ $? -ne 0 ]; then
                        echo "Error converting $DFS_FILE to surface for $SUBJECT_NAME. Skipping."
                        continue
                    fi

                    echo "Successfully converted $DFS_FILE to surface for subject: $SUBJECT_NAME"
                else
                    echo "No .dfs files found for subject: $SUBJECT_NAME in $SUBJECT_DIR/brainsuite. Skipping."
                fi
            done
        else
            echo "MRI file (T1.mgz) not found for subject: $SUBJECT_NAME. Skipping."
        fi
    fi
done

echo "Surface conversion completed for all subjects."
