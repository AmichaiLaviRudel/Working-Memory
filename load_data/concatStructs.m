
%% --- Helper Function ---
function result = concatStructs(s1, s2)
    % Obtain union of field names
    fields = union(fieldnames(s1), fieldnames(s2));
    result = struct();
    
    for i = 1:length(fields)
        f = fields{i};
        
        % For specific fields, keep the value from s1
        if (strcmp(f, 'SessionDate') && isfield(s1, 'SessionDate')) || ...
           (strcmp(f, 'SessionStartTime_UTC') && isfield(s1, 'SessionStartTime_UTC'))
            result.(f) = s1.(f);
            continue;
        end
        
        if isfield(s1, f) && isfield(s2, f)
            val1 = s1.(f);
            val2 = s2.(f);
            if isstruct(val1) && isstruct(val2)
                if numel(val1)==1 && numel(val2)==1
                    result.(f) = concatStructs(val1, val2);
                else
                    result.(f) = [val1, val2];
                    expected = numel(val1) + numel(val2);
                    actual = numel(result.(f));
                    if actual ~= expected
                        error('Validation failed for field %s: expected %d elements, got %d', f, expected, actual);
                    end
                end
            else
                result.(f) = [val1, val2];
                expected = numel(val1) + numel(val2);
                actual = numel(result.(f));
                if actual ~= expected
                    error('Validation failed for field %s: expected %d elements, got %d', f, expected, actual);
                end
            end
        elseif isfield(s1, f)
            result.(f) = s1.(f);
        else
            result.(f) = s2.(f);
        end
    end

end
