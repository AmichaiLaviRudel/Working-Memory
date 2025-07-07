function data = load_gng_data(file_path)
% file_path = "G:\My Drive\Study\Lab\Code_temp\tone_ass.mat";
addpath ("G:\My Drive\Study\Lab\Code_temp\GNG")
load(file_path);

p = struct(Data.p);
data = struct(Data);

data = rmfield(data,"p");
p = rmfield(p, "Stims");


data = horzcat(struct2table(data, AsArray=true), struct2table(p, AsArray=true));
data = table2struct(data);


[filepath, name] = fileparts(file_path);
save(sprintf("%s_formmated.mat",fullfile(filepath,name)),"-struct","data")

end

