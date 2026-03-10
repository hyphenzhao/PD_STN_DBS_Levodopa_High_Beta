#!/bin/bash

# Define the root directory containing the subject directories
ROOT_DIR="/Volumes/Workspace/NextCloud/Ruijin/NeuroImaging/subjects"

# Set the FreeSurfer subjects directory
export SUBJECTS_DIR="/Volumes/Workspace/FreeSurfer/subjects"
export FS_ALLOW_DEEP=1  # Enable ML routines for recon-all
export OMP_NUM_THREADS=8

# Function to process each subject in parallel
process_subject() {
    local SUBJECT_DIR="$1"
    local SUBJECT=$(basename "$SUBJECT_DIR")
    local DICOM_DIR="$SUBJECT_DIR/MR"
    
    # Check if the DICOM directory exists
    if [ -d "$DICOM_DIR" ]; then
        echo "Processing subject: $SUBJECT"

        # Step 1: Find a single DICOM file to use with mri_convert
        DICOM_FILE=$(find "$DICOM_DIR" -type f -name "*.dcm" | head -n 1)

        # Check if a DICOM file was found
        if [ -f "$DICOM_FILE" ]; then
            # Define the output path for the MGZ file
            MGZ_OUTPUT_PATH="$SUBJECTS_DIR/$SUBJECT/mri/orig/001.mgz"
            NII_OUTPUT_PATH="$SUBJECTS_DIR/$SUBJECT/mri/T1.nii"
            mkdir -p "$(dirname "$MGZ_OUTPUT_PATH")"

            echo "Running mri_convert to convert DICOM to MGZ for $SUBJECT..."
            mri_convert "$DICOM_FILE" "$MGZ_OUTPUT_PATH"
            if [ $? -ne 0 ]; then
                echo "Error in mri_convert for $SUBJECT. Skipping."
                return
            fi
        else
            echo "No DICOM files found in $DICOM_DIR for $SUBJECT. Skipping."
            return
        fi

        RECON_DONE_FILE="$SUBJECTS_DIR/$SUBJECT/scripts/recon-all.done"
        if [ -f "$RECON_DONE_FILE" ]; then
            echo "Skipping $SUBJECT: recon-all already completed $RECON_DONE_FILE."
            return
        fi

        # Step 2: Run FreeSurfer's recon-all command using the MGZ file as input
        echo "Running recon-all for $SUBJECT using MGZ file..."
        recon-all -s "$SUBJECT" -all -parallel 
        mri_convert "$SUBJECTS_DIR/$SUBJECT/mri/T1.mgz" "$NII_OUTPUT_PATH" 

        if [ $? -ne 0 ]; then
            echo "Error in recon-all for $SUBJECT. Skipping."
            return
        fi

        echo "Completed processing for $SUBJECT"
    else
        echo "Skipping $SUBJECT: DICOM directory not found in $DICOM_DIR"
    fi
}

# Loop over each subject directory in the root directory
for SUBJECT_DIR in "$ROOT_DIR"/*; do
    # Check if the item is a directory
    if [ -d "$SUBJECT_DIR" ]; then
        process_subject "$SUBJECT_DIR" &
    fi
done

# Wait for all background processes to finish
wait
echo "All subjects processed."
