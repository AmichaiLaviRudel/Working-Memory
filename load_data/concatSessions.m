function concatSessions()
    %% GNG Session Concatenation Tool
    % This function concatenates GNG sessions from the same animal on the same day
    % Sessions are concatenated in chronological order (by creation time)
    
    %% Step 1: Let user select the animal folder
    fprintf('Select the animal folder containing GNG sessions...\n');
    animal_folder = uigetdir('Z:\Shared\Amichai\Behavior\data', 'Select Animal Folder');
    
    if animal_folder == 0
        fprintf('No folder selected. Exiting...\n');
        return;
    end
    
    %% Step 2: Find all .mat files in the GNG Session Data subfolder
    gng_session_path = fullfile(animal_folder, 'GNG', 'Session Data');
    
    if ~exist(gng_session_path, 'dir')
        fprintf('Error: GNG/Session Data folder not found in %s\n', animal_folder);
        return;
    end
    
    % Get all .mat files in the Session Data folder
    mat_files = dir(fullfile(gng_session_path, '*.mat'));
    
    if isempty(mat_files)
        fprintf('No .mat files found in %s\n', gng_session_path);
        return;
    end
    
    % Create full file paths
    file_paths = fullfile({mat_files.folder}, {mat_files.name});
    file_paths = file_paths(:); % Convert to column vector
    
    fprintf('Found %d .mat files\n', length(file_paths));
    
    %% Step 3: Group files by date
    file_dates = cell(length(file_paths), 1);
    file_times = zeros(length(file_paths), 1);
    
    for i = 1:length(file_paths)
        file_info = dir(file_paths{i});
        file_dates{i} = datestr(file_info.datenum, 'yyyymmdd');
        file_times(i) = file_info.datenum;
    end
    
    % Find unique dates
    unique_dates = unique(file_dates);
    fprintf('Found sessions from %d different dates:\n', length(unique_dates));

    
    %% Step 4: Process each date separately
    for date_idx = 1:length(unique_dates)
        current_date = unique_dates{date_idx};
        
        % Find files from this date
        date_mask = strcmp(file_dates, current_date);
        date_files = file_paths(date_mask);
        date_times = file_times(date_mask);
        
        fprintf('\nProcessing date: %s (%d sessions)\n', current_date, length(date_files));
        
        if length(date_files) == 1
            fprintf('Only one session found for this date. Skipping concatenation.\n');
            continue;
        end
        
        % Sort files by creation time
        [~, sort_idx] = sort(date_times);
        sorted_files = date_files(sort_idx);
        
        % Display files in chronological order
        fprintf('Sessions in chronological order:\n');
        for i = 1:length(sorted_files)
            [~, filename, ~] = fileparts(sorted_files{i});
            fprintf('  %d. %s\n', i, filename);
        end
        
        %% Step 5: Concatenate sessions for this date
        try
            fprintf('Concatenating sessions...\n');
            
                         % Load first file
             first_struct = load(sorted_files{1});
             fprintf('Loaded: %s\n', sorted_files{1});
             
             % Concatenate with remaining files
             merged_struct = first_struct;
             for i = 2:length(sorted_files)
                 current_struct = load(sorted_files{i});
                 merged_struct.SessionData = concatStructs(merged_struct.SessionData, current_struct.SessionData);
                 fprintf('Concatenated: %s\n', sorted_files{i});
             end
             
             SessionData = merged_struct.SessionData;
            
            %% Step 6: Save merged file in original location
            % Use the first file's location and name as the merged file
            [filepath, basename, ext] = fileparts(sorted_files{1});
            merged_filename = sprintf('%s_merged%s', basename, ext);
            merged_path = fullfile(filepath, merged_filename);
            
            % Save concatenated data
            save(merged_path, 'SessionData', '-mat');
            fprintf('Saved concatenated data to: %s\n', merged_path);
            
            %% Step 7: Move original files to backup folder
            % Create backup folder in the same directory
            backup_folder = fullfile(filepath, 'Original_Files');
            if ~exist(backup_folder, 'dir')
                mkdir(backup_folder);
                fprintf('Created backup folder: %s\n', backup_folder);
            end
            
            % Move all original files to backup folder
            for i = 1:length(sorted_files)
                [~, filename, ext] = fileparts(sorted_files{i});
                backup_filename = sprintf('%s_original%s', filename, ext);
                backup_path = fullfile(backup_folder, backup_filename);
                movefile(sorted_files{i}, backup_path);
                fprintf('Moved: %s -> %s\n', filename, backup_filename);
            end
            
        catch ME
            fprintf('Error concatenating sessions for date %s: %s\n', current_date, ME.message);
            continue;
        end
    end
    
    fprintf('\nConcatenation complete!\n');
end



