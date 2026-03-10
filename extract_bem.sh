#!/bin/bash

SUBJECTS_DIR="/Volumes/Workspace/FreeSurfer/subjects"
OUTPUT_DIR="/Volumes/Workspace/FreeSurfer/subjects"  # The output directory is the same as SUBJECTS_DIR

# Function to process each subject
process_subject() {
    SUBJECT_DIR="$1"
    SUBJECT_NAME=$(basename "$SUBJECT_DIR")
    MRI_FILE="$SUBJECT_DIR/mri/T1.nii"
    
    # Check if the T1.nii file exists
    if [ -f "$MRI_FILE" ]; then
        # Step 1: Create a new directory called 'brainsuite' inside the subject directory
        BRAINSUITE_DIR="$SUBJECT_DIR/brainsuite"
        if [ -d "$BRAINSUITE_DIR" ]; then
            echo "BRAINSUITE_DIR already exists for subject: $SUBJECT_NAME. Skipping."
            return
        fi
        
        mkdir -p "$BRAINSUITE_DIR"

        # Step 2: Copy T1.nii into the 'brainsuite' directory
        cp "$MRI_FILE" "$BRAINSUITE_DIR/T1.nii"

        # Step 3: Create a mask (this is assumed to be some preprocessing step you have or it should be done manually)
        MASK_PATH="$BRAINSUITE_DIR/mask"  # Assuming you have a mask file available

        # Step 4: Run BrainSuite's BSE (Brain Segmentation Engine)
        echo "Running BrainSuite BSE for subject: $SUBJECT_NAME"
        BSE_OUTPUT="$BRAINSUITE_DIR/bse"

        # Run the BSE command with the mask
        bse -i "$BRAINSUITE_DIR/T1.nii" -o "$BSE_OUTPUT" --mask "$MASK_PATH"  # Ensure bse is in your PATH

        # Check if BSE ran successfully
        if [ $? -ne 0 ]; then
            echo "Error running BSE for $SUBJECT_NAME. Skipping."
            return
        fi

        # Step 5: Run BrainSuite's skullfinder (skull stripping)
        echo "Running BrainSuite skullfinder for subject: $SUBJECT_NAME"
        SKULLSTRIP_OUTPUT="$BRAINSUITE_DIR/skull_sculp"

        # Run skullfinder command with mask and other arguments
        skullfinder -i "$BRAINSUITE_DIR/T1.nii" -o "$SKULLSTRIP_OUTPUT" -m "$BRAINSUITE_DIR/mask.nii.gz" -s "$BRAINSUITE_DIR/T1"  # Ensure skullfinder is in your PATH

        # Check if skullfinder ran successfully
        if [ $? -ne 0 ]; then
            echo "Error running skullfinder for $SUBJECT_NAME. Skipping."
            return
        fi

        echo "Completed processing for subject: $SUBJECT_NAME"
    else
        echo "T1.nii not found for subject: $SUBJECT_NAME. Skipping."
    fi
}

# Loop over each subject directory in the root directory
for SUBJECT_DIR in "$SUBJECTS_DIR"/*; do
    # Check if the item is a directory
    if [ -d "$SUBJECT_DIR" ]; then
        # Call the process_subject function in a new subprocess (background job)
        process_subject "$SUBJECT_DIR" &
    fi
done

# Wait for all background processes to complete
wait

echo "Processing completed for all subjects."
