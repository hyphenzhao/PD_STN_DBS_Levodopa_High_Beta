addpath(genpath('/Volumes/Workspace/Matlab Workspace/mne-matlab-master'))
% Define the directory containing the .surf files
inputDir = '/Volumes/Workspace/FreeSurfer/subjects';  % Adjust the path to your specific folder

% Get a list of all subject directories
subjectFolders = dir(inputDir);
subjectFolders = subjectFolders([subjectFolders.isdir] & ~startsWith({subjectFolders.name}, '.'));

% Loop over each subject folder
for i = 1:length(subjectFolders)
    % Get the subject folder path
    subjectPath = fullfile(inputDir, subjectFolders(i).name);
    brainsuiteDir = fullfile(subjectPath, 'brainsuite');
    bemDir = fullfile(subjectPath, 'bem');
    

    % Check if the brainsuite directory exists
    if strcmp(subjectFolders(i).name, '121_ShengJinlong')
       disp(subjectFolders(i).name);
       continue;
    end

    if ~isfolder(brainsuiteDir)
        fprintf('Brainsuite folder not found for subject: %s. Skipping.\n', subjectFolders(i).name);
        continue;
    end

    % Create the bem directory if it does not exist
    if ~isfolder(bemDir)
        mkdir(bemDir);
    end

    % Get all .surf files in the brainsuite directory
    surfFiles = dir(fullfile(brainsuiteDir, '*.surf'));

    % Loop over each .surf file
    for j = 1:length(surfFiles)
        % Input file path
        inputSurfPath = fullfile(brainsuiteDir, surfFiles(j).name);
        
        % Remove the "T1." prefix from the filename
        outputFileName = regexprep(surfFiles(j).name, '^T1\.', '');
        outputFileName = strrep(outputFileName, 'scalp', 'outer_skin');
        
        % Output file path
        outputSurfPath = fullfile(bemDir, outputFileName);
        
        % Display progress
        fprintf('Reducing surface: %s -> %s\n', inputSurfPath, outputSurfPath);
        
        % Call the mne_reduce_surface function
        try
            [verts, faces] = mne_reduce_surface(inputSurfPath, 20000, outputSurfPath);
            fprintf('Successfully reduced surface for %s\n', surfFiles(j).name);
        catch ME
            % Handle errors and continue
            fprintf('Error processing %s: %s\n', surfFiles(j).name, ME.message);
        end
    end
end

disp('Surface reduction completed for all subjects.');