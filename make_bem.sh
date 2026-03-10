#!/bin/bash
export FS_ALLOW_DEEP=1
# Set the FreeSurfer subjects directory
SUBJECTS_DIR="/Volumes/Workspace/FreeSurfer/subjects"

# Path to FreeSurfer GCA atlas
GCA_ATLAS="$FREESURFER_HOME/average/RB_all_withskull_2020_01_02.gca"

# Loop through each subdirectory in SUBJECTS_DIR
for subject_dir in "$SUBJECTS_DIR"/*; do
    # Check if the path is a directory
    if [ -d "$subject_dir" ]; then
        # Get the subject ID (the name of the subdirectory)
        subject_id=$(basename "$subject_dir")
        
        # Define paths for input MRI and output BEM prefix
        input_mri="${subject_dir}/mri/T1.mgz"
        input_ct="${subject_dir}/ct/001.mgz"
        corrected_mri="${subject_dir}/mri/T1_corrected.mgz"  # Path for bias-corrected MRI
        bem_dir="${subject_dir}/bem"
        inner_dir="${bem_dir}/inner"
        outer_dir="${bem_dir}/outer"
        ws_dir="${bem_dir}/watershed"
        
        if [ -d "$inner_dir" ]; then
            echo "BEM has been created for: $subject_id. Skipping."
            continue
        fi
        # Check if the T1.mgz file exists
        if [ -f "$input_mri" ]; then
            echo "Processing subject: $subject_id"

            # Apply bias correction if corrected MRI doesn't exist
            if [ ! -f "$corrected_mri" ]; then
                echo "Applying bias correction to $input_mri"
                mri_nu_correct.mni --i "$input_mri" --o "$corrected_mri"
                if [ $? -ne 0 ]; then
                    echo "Error in bias correction for $subject_id, skipping..."
                    continue
                fi
            fi

            echo "Generating BEM surfaces for subject: $subject_id"
            
            # Create bem, inner, outer, and watershed directories if they don't exist
            mkdir -p "$bem_dir"
            mkdir -p "$inner_dir"
            mkdir -p "$outer_dir"
            mkdir -p "$ws_dir"
            
            # Run mri_watershed for inner surfaces with specified parameters
            inner_output_bem="${inner_dir}/${subject_id}"
            echo "Generating inner BEM surfaces with -h 10 -less for subject: $subject_id"
            mri_watershed -useSRAS -atlas -h 15 -less -surf "$inner_output_bem" "$corrected_mri" "${ws_dir}/inner_ws.mgz" &
            
            if [ $? -eq 0 ]; then
                echo "Inner BEM surfaces generated successfully for $subject_id."

                # Copy the inner BEM surfaces to the bem folder
                for surface in brain inner_skull; do
                    surf_file="${inner_output_bem}_${surface}_surface"
                    copy_path="${bem_dir}/${surface}.surf"
                    if [ -f "$surf_file" ]; then
                        cp "$surf_file" "$copy_path"
                        echo "Copied $surf_file to $copy_path"
                    else
                        echo "Inner surface file $surf_file not found, skipping copy."
                    fi
                done
            else
                echo "Error generating inner BEM surfaces for $subject_id."
            fi

            # Run mri_watershed for outer surfaces with specified parameters
            outer_output_bem="${outer_dir}/${subject_id}"
            echo "Generating outer BEM surfaces with -h 50 -more for subject: $subject_id"
            mri_watershed -useSRAS -atlas -h 80 -more -surf "$outer_output_bem" "$corrected_mri" "${ws_dir}/outer_ws.mgz" &
            
            if [ $? -eq 0 ]; then
                echo "Outer BEM surfaces generated successfully for $subject_id."

                # Copy the outer BEM surfaces to the bem folder
                for surface in outer_skull outer_skin; do
                    surf_file="${outer_output_bem}_${surface}_surface"
                    copy_path="${bem_dir}/${surface}.surf"
                    if [ -f "$surf_file" ]; then
                        cp "$surf_file" "$copy_path"
                        echo "Copied $surf_file to $copy_path"
                    else
                        echo "Outer surface file $surf_file not found, skipping copy."
                    fi
                done
            else
                echo "Error generating outer BEM surfaces for $subject_id."
            fi
        else
            echo "T1.mgz not found for $subject_id, skipping..."
        fi
        # break # Uncomment for testing a single subject
    fi

done

echo "BEM generation process completed for all subjects."
