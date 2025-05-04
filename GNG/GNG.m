% ========================== Opening / closing functions =================

function varargout = GNG(varargin)

    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
                       'gui_Singleton',  gui_Singleton, ...
                       'gui_OpeningFcn', @GNG_OpeningFcn, ...
                       'gui_OutputFcn',  @GNG_OutputFcn, ...
                       'gui_LayoutFcn',  [] , ...
                       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end

    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
end

function GNG_OpeningFcn(hObject, eventdata, handles, varargin)

    handles.output = hObject;
    handles.Parameters = GNG_parameters() ;
    guidata(hObject, handles) 
    
    hold(handles.anlss_ax , 'on')
    initialize_raw_ax(hObject) ;
    % update_anlss_ax(0,0,hObject,0)
    handles = guidata(hObject) ;

    handles.data_dir_button.UserData = cd ;
    handles.load_data_button.UserData = cd ;

    guidata(hObject, handles);
    
end

function main_fig_CloseRequestFcn(hObject, eventdata, handles)

    % in case the simulation is running
    if handles.start_button.Value
        for i = 1:numel(handles.Listeners)
            delete(handles.Listeners{i})
        end
        pause(1)
        stop_task(hObject)
    end
    
    try
        handles.InputSession.delete()
        handles.OutputSession.delete()
    catch
    end

    delete(hObject);
    
end

function varargout = GNG_OutputFcn(hObject, eventdata, handles) 

    varargout{1} = handles.output;
    
end

% ========================== Main callbacks ==============================

function start_button_Callback(hObject, eventdata, handles)

    if ~handles.start_button.UserData
       
        p = handles.Parameters ;
                    
        if ~handles.save_data_check.Value

            answer = questdlg('Not saving, continue?','Not saving, continue?','Yes','No','No') ;
            if ~strcmp(answer,'Yes')
                return
            end

        end

        p.add_user_parameters(handles) ;
        handles.Data = GNG_data(p) ;
        handles.Stimulus_generator = GNG_stimulus_generator(p) ;
        handles.Reinforcement_generator = GNG_reinforcement_generator(p) ;
        guidata(hObject,handles)

        create_daq_sessions(hObject)
        create_outputs(hObject)
        create_listeners(hObject)
        initialize_raw_ax(hObject)
        update_anlss_types(hObject)
        update_anlss_ax(0,0,hObject)
        
        cla(handles.stim_ax1)
        cla(handles.stim_ax2)
        handles = guidata(hObject) ;
        
        
        % changing start/stop task button
        handles.start_button.UserData = 1 ;
        handles.start_button.String = 'Stop' ;      
        handles.start_button.BackgroundColor = [0.8 0 0] ; 
        
        guidata(hObject, handles) ;
        
        % disabling all texts, buttons and slider
        object_enable(hObject,'off')
    
        % starting all simulation elements
        
        handles.Stimulus_generator.start_run()
        
        handles.InputSession.startBackground
        
    else                                        % In case the simulation is running 
        
        stop_task(hObject)

        guidata(hObject, handles) ;
        
    end
    
end

function new_button_Callback(hObject, eventdata, handles)

    Answer = questdlg('Are you sure you want to start a new experiment?' , ...
        'New experiment' , 'Yes' , 'No' , 'No') ;
    
    if ~strcmp(Answer , 'Yes')
        return
    end

    % enabling all texts, buttons and slider
    object_enable(hObject,'on')
    
    % stopping and deleting all simulation elements and listeners
    try
        handles.Stimulus_generator.delete()
    catch
    end
    
    try
        delete(handles.InputSession)
        delete(handles.OutputSession)
    catch
    end
    
    try
        for i = 1:numel(handles.Listeners)
        delete(handles.Listeners{i})
        end
    catch
    end
    
    handles.Data.delete()
    handles.data_slider.Value = 1 ; % reseting slider
    handles.data_slider.Enable = 'off' ;
    handles.new_button.Enable = 'off' ;
    handles.start_button.Enable = 'on' ;
    handles.reward_button.Enable = 'off' ;
    handles.anlss_type_menu.Value = 1 ;
    handles.anlss_type_menu.Enable = 'off' ;
    handles.anlss_ax.UserData = '' ;
    initialize_raw_ax(hObject) % clearing all axes
    handles = guidata(hObject) ;

    handles.Parameters.zero_trial_count ; % zeroing trial count for online anlss
    
end

function load_data_button_Callback(hObject, eventdata, handles)

    [file_name , file_path] = uigetfile('*.mat' , 'Choose data file',...
        handles.load_data_button.UserData) ;

    if file_name == 0
        return
    end

    handles.load_data_button.UserData = [file_path file_name] ;
    load([file_path file_name])

    handles.data_dir_button.UserData = file_path ;

    handles.Parameters = Data.p ;
    
    handles.Data = Data ;

    guidata(hObject,handles)

    match_parameters(hObject,[file_path file_name])
    initialize_raw_ax(hObject)
    full_axes(hObject)
    update_anlss_types(hObject)
    object_enable(hObject,'off')
    handles.Go_stims_menu.Enable = 'On' ;
    handles.NoGo_stims_menu.Enable = 'On' ;
    handles.catch_stims_menu.Enable = 'On' ;
    handles.load_data_button.Enable = 'on' ;
    handles.new_button.Enable = 'on' ;
    handles.start_button.Enable = 'off' ;
    handles.reward_button.Enable = 'off' ;
    handles.data_slider.Enable = 'on' ;
    handles.data_slider.Value = 0 ;
        
end

function load_env_button_Callback(hObject, eventdata, handles)

    [file_name , file_path] = uigetfile('*.mat' , 'Choose data file to copy environment') ;
    
    if file_name == 0
        return
    end
    
    match_parameters(hObject,[file_path file_name])
    
end

function altered_slider_Callback(~ , eventdata)
    
    % Original callback was changed so that the slider will change the
    % display during motion. original callback was removed
    
    handles = guidata(eventdata.AffectedObject);
    hObject = eventdata.AffectedObject ;
    p = handles.Parameters ;
    disp_time = handles.Data.time(end)*handles.data_slider.Value ;
    time_span = p.time_span ;

    % moving raw data axes
    
    y_lim = [0 length(handles.Ticks)+1] ;
    axis(handles.raw_ax , [[-time_span 0] + disp_time y_lim])
    
    % updating anlss axes if needed
    trial_count = handles.Data.get_n_trials(disp_time);
    trial_count = trial_count - mod(trial_count , p.trials_per_anlss) ;
    update_anlss_ax(0,0,hObject,trial_count)

end

% ========================== Other Callbacks =============================

function anlss_type_menu_Callback(hObject, eventdata, handles)

    p = handles.Parameters ;
        
    disp_time = handles.Data.time(end)*handles.data_slider.Value ;
    trial_count = handles.Data.get_n_trials(disp_time);
    trial_count = trial_count - mod(trial_count , p.trials_per_anlss) ;
    
    if p.get_trial_count() > p.trials_per_anlss
        update_anlss_ax(0,0,hObject,trial_count)
    end
    
end

function auto_reward_check_Callback(hObject, eventdata, handles)

    if hObject.Value
 
        fix_ITI(hObject)
        
    end
    
end

function catch_stims_menu_Callback(hObject, eventdata, handles)

    stim_id = hObject.UserData(hObject.Value) ;
    present_stimulus_details(hObject,stim_id)
    
end

function data_dir_button_Callback(hObject, eventdata, handles)

    handles.data_dir_button.UserData =...
        uigetdir(handles.data_dir_button.UserData , 'Choose folder for saving data' ) ;
    
    if handles.data_dir_button.UserData == 0
        
        handles.data_dir_button.UserData = cd ;
        
        warndlg('No folder selected, using current directory' , 'No folder selected')
        
    end   
    
end

function file_name_Callback(hObject, eventdata, handles)

    check_file_name(hObject) ;

end

function Go_stims_menu_Callback(hObject, eventdata, handles)

    stim_id = hObject.UserData(hObject.Value) ;
    present_stimulus_details(hObject,stim_id)

end

function ITI_Callback(hObject, eventdata, handles)

    check_positive(hObject)
    fix_ITI(hObject)
    
end

function ITI_range_Callback(hObject, eventdata, handles)

    check_positive(hObject)
    fix_ITI(hObject)  
    
end

function lick_threshold_Callback(hObject, eventdata, handles)

    input = str2double(get(hObject,'String'));
    
    if isnan(input) || (input <= 0)

        errordlg('You must enter a positive integer','Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))

    else
        
        hObject.String = num2str(ceil(input)) ;
        hObject.UserData = num2str(ceil(input)) ;

    end
    
end

function load_stims_button_Callback(hObject, eventdata, handles)

    [file_name , file_path] = uigetfile('*.xlsx' , 'Choose stimuli file',...
        handles.load_data_button.UserData) ;

    if file_name == 0
        return
    end
    
    handles.load_stims_button.UserData = [file_path file_name] ;
    
    valid = handles.Parameters.add_stimuli(readtable([file_path file_name])) ;
    
    if ~valid
        return
    end
    
    fix_ITI(hObject) ;
    
    update_stimuli_menus(hObject)
    
    present_stimulus_details(hObject,1)
    handles.start_button.Enable = 'On' ;
    
end

function mouse_Callback(hObject, eventdata, handles)

    check_file_name(hObject) ;

end

function no_lick_wait_time_Callback(hObject, eventdata, handles)

    input = str2double(hObject.String) ;
    
    if isnan(input) || input < 0

        errordlg('Must enter a positive number' ,'Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))

    else

        set(hObject , 'UserData' , get(hObject,'String'))

    end
    
end

function NoGo_stims_menu_Callback(hObject, eventdata, handles)

    stim_id = hObject.UserData(hObject.Value) ;
    present_stimulus_details(hObject,stim_id)
    
end

function noise_punish_dur_Callback(hObject, eventdata, handles)

    input = str2double(hObject.String) ;
    window = str2double(handles.response_window.String) ;

    if isnan(input) || (input <= 0) || (input > window)

        errordlg('You must enter a numeric between 0 and response window length','Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))        
        return

    end

    set(hObject , 'UserData' , get(hObject,'String'))
    
end

function noise_punish_check_Callback(hObject, eventdata, handles)

    if hObject.Value
        handles.noise_punish_dur.Enable = 'On' ;
    else
        handles.noise_punish_dur.Enable = 'Off' ;
    end
    
end

function repeat_incorrect_check_Callback(hObject, eventdata, handles)
end

function response_window_Callback(hObject, eventdata, handles)

    check_positive(hObject)
    fix_ITI(hObject)
    
    if str2double(hObject.String) < str2double(handles.reinforcement_delay.String)
       handles.reinforcement_delay.String = hObject.String ;
    end
    
end

function reward_button_Callback(hObject, eventdata, handles)

    handles = guidata(hObject) ;

    if ~handles.OutputSession.IsRunning
        release(handles.OutputSession)
        queueOutputData(handles.OutputSession,handles.reward_output)
        handles.OutputSession.startBackground
        handles.Data.add_reward(handles.Data.time(end))
    end
    
    guidata(hObject,handles)
    if handles.start_button.UserData
        update_reinforcement_scat(hObject,'true')
    end
    
end

function reinforcement_delay_Callback(hObject, eventdata, handles)

    input = str2double(hObject.String) ;
    window = str2double(handles.response_window.String) ;
    
    if isnan(input) || (input < 0) ||(input > window)

        errordlg('Allowed durations are between 0 and window length' ,'Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))

    else

        set(hObject , 'UserData' , get(hObject,'String'))

    end
    
end

function reward_dur_Callback(hObject, eventdata, handles)

    % making sure reward_dur is within the allowed duration range
    
    input = str2double(hObject.String) ;
    p = handles.Parameters ;
    
    if isnan(input) || (input < p.min_reward_dur) ||(input > p.max_reward_dur)

        errordlg(['Allowed durations are between ' num2str(p.min_reward_dur) ...
            ' and ' num2str(p.max_reward_dur) ' ms'] ,'Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))

    else

        set(hObject , 'UserData' , get(hObject,'String'))
        fix_ITI(hObject)

    end 
    
end

function save_data_check_Callback(hObject, eventdata, handles)
end

function task_dur_Callback(hObject, eventdata, handles)

    input = str2double(get(hObject,'String'));
    
    if isnan(input) || (input <= 0)

        errordlg('You must enter a positive integer','Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))

    else
        
        hObject.String = num2str(ceil(input)) ;
        hObject.UserData = num2str(ceil(input)) ;

    end
    
end

function timeout_punish_check_Callback(hObject, eventdata, handles)

    if hObject.Value
        handles.timeout_punish_dur.Enable = 'On' ;
    else
        handles.timeout_punish_dur.Enable = 'Off' ;
    end
    
end

function timeout_punish_dur_Callback(hObject, eventdata, handles)

    check_positive(hObject)

end

function trial_cue_check_Callback(hObject, eventdata, handles)

    if hObject.Value
        fix_ITI(hObject)
        handles.trial_cue_onset.Enable = 'On' ;
        handles.trial_cue_dur.Enable = 'On' ;
    else
        handles.trial_cue_onset.Enable = 'Off' ;
        handles.trial_cue_dur.Enable = 'Off' ;
    end
    
end

function trial_cue_dur_Callback(hObject, eventdata, handles)

    input = str2double(get(hObject,'String'));
    trial_cue_onset = str2double(handles.trial_cue_onset.String) ;

%     if isnan(input) || (input <= 0) || (input > trial_cue_onset )
% 
%         errordlg('You must enter a positive numeric smaller than the trial cue onset',...
%             'Invalid Input','modal')
%         set(hObject , 'String' , get(hObject,'UserData'))
% 
%     else
% 
%         set(hObject , 'UserData' , get(hObject,'String'))
% 
%     end 
    
end

function trial_cue_onset_Callback(hObject, eventdata, handles)

    input = str2double(get(hObject,'String'));
    trial_cue_dur = str2double(handles.trial_cue_dur.String) ;

%     if isnan(input) || (input < trial_cue_dur )
% 
%         errordlg('You must enter a numeric bigger than the trial cue duration',...
%             'Invalid Input','modal')
%         set(hObject , 'String' , get(hObject,'UserData'))
% 
%     else
% 
%         set(hObject , 'UserData' , get(hObject,'String'))
% 
%     end 
    
    fix_ITI(hObject)   
    
end

function wait_for_no_lick_check_Callback(hObject, eventdata, handles)

    if hObject.Value
        handles.no_lick_wait_time.Enable = 'On' ;
    else
        handles.no_lick_wait_time.Enable = 'Off' ;
    end
    
end

% ========================== Analysis functions ==========================

function update_anlss_ax(~,~,hObject,trial_count)
    % Function update_anlss_ax plots an online anlss of the all
    % trials up to trial_count. called by listener
    %
    % Input:
    % 2 dummie variables (due to MATLAB's listener callback)
    % hObject - a GUI object
    % trial_count - integer , number of trials to analyze
    %
    
    handles = guidata(hObject) ;
    p = handles.Parameters;
 
    if nargin < 4
        trial_count = p.get_trial_count() -  mod(p.get_trial_count() , p.trials_per_anlss) ;
    end
    
    switch handles.anlss_type_menu.String{handles.anlss_type_menu.Value}
        case 'Lick rates'
            plot_lick_rates(hObject,trial_count)
        case 'Trial counts'           
            plot_trial_counts(hObject,trial_count)
        case "d'"           
            plot_d_prime(hObject,trial_count)  
        case 'Psychometric'
            plot_psychometric(hObject,trial_count)
        case 'Response Latencies'            
            plot_response_latencies(hObject,trial_count)         
    end
        
end

function update_anlss_types(hObject)
    % function update_anlss_types adds options to the online anlss in
    % specific cases
    
    handles = guidata(hObject) ;
    
    handles.anlss_type_menu.String = {'Trial counts','Lick rates',"d'",...
        'Psychometric','Response Latencies'} ;
           
end

function plot_d_prime(hObject,trial_count)

    handles = guidata(hObject) ;
    p = handles.Parameters;
    
    if ~strcmp(handles.anlss_ax.UserData , handles.anlss_type_menu.String{handles.anlss_type_menu.Value})
        
        cla(handles.anlss_ax)
        
        handles.h_dprime_plot1 = plot(handles.anlss_ax , 1 , 1 , 'k') ;
        set(handles.h_dprime_plot1,'XData',[],'YData',[])

        handles.h_dprime_plot2 = plot(handles.anlss_ax , 1 , 1 , '*k') ;
        set(handles.h_dprime_plot2,'XData',[],'YData',[])

        ylabel(handles.anlss_ax , "d'" , 'fontsize' , 12)
        xlabel(handles.anlss_ax , ['Trials/' num2str(handles.Parameters.trials_per_anlss)]...
            , 'fontsize' , 12)
        handles.anlss_ax.XTickMode = 'auto' ;
        handles.anlss_ax.XTickLabelMode = 'auto' ;
        axis(handles.anlss_ax , [0 10 -1 5])
        
        legend(handles.anlss_ax,'Off')
        
        handles.anlss_ax.UserData = handles.anlss_type_menu.String{handles.anlss_type_menu.Value} ;

    end
    
    trial_types = get_trial_types(hObject,trial_count) ;
    
    XData = 0 ;
    hit_YData = 0 ;
    CR_YData = 0 ;
    miss_YData = 0 ;
    FA_YData = 0 ;

    dprime_YData = [] ;

    for trials = p.trials_per_anlss:p.trials_per_anlss:trial_count

        hit_YData(end+1) = sum(trial_types(1:trials) == 1) ; 
        CR_YData(end+1) = sum(trial_types(1:trials) == 2) ;       
        miss_YData(end+1) = sum(trial_types(1:trials) == 3) ; 
        FA_YData(end+1) = sum(trial_types(1:trials) == 4) ; 
        XData(end+1) = trials/p.trials_per_anlss ;

        if length(hit_YData) <=10
            Hit = hit_YData(end) ;
            Miss = miss_YData(end) ;
            CR = CR_YData(end) ;
            FA = FA_YData(end) ;
            Go_licks_frac = 100*Hit/(Hit + Miss) ;
            NoGo_licks_frac = 100*FA/(FA + CR) ;
        else
            Hit100 = hit_YData(end)-hit_YData(end-10) ;
            Miss100 = miss_YData(end)-miss_YData(end-10) ;
            CR100 = CR_YData(end)-CR_YData(end-10) ;
            FA100 = FA_YData(end)-FA_YData(end-10) ;
            Go_licks_frac = 100*Hit100/(Hit100 + Miss100) ;
            NoGo_licks_frac = 100*FA100/(FA100 + CR100) ;
        end

        if isnan(Go_licks_frac) || isinf(Go_licks_frac)
            Go_licks_frac = 0 ;
        end


        if isnan(NoGo_licks_frac) || isinf(NoGo_licks_frac)
            NoGo_licks_frac = 0 ;
        end

        dprime_YData(end+1) = max(min(norminv(Go_licks_frac/100),2.3),-2.3) -...
            max(min(norminv(NoGo_licks_frac/100),2.3),-2.3) ;

        if isnan(dprime_YData(end)) || isinf(dprime_YData(end))
            dprime_YData(end) = 0 ;
        end

    end
    
    set(handles.h_dprime_plot1 , 'XData' , XData(2:end) , 'YData' , dprime_YData)
    set(handles.h_dprime_plot2 , 'XData' , XData(2:end) , 'YData' , dprime_YData)

    axis(handles.anlss_ax , [0 max([(trial_count/p.trials_per_anlss)+1 , 10]) -1 5])
    
    guidata(hObject, handles) ;
        
end

function plot_lick_rates(hObject,trial_count)
    
    handles = guidata(hObject) ;
    p = handles.Parameters;
    
    if ~strcmp(handles.anlss_ax.UserData , handles.anlss_type_menu.String{handles.anlss_type_menu.Value})
    
        cla(handles.anlss_ax)
        
        handles.h_Go_licks_frac_plot1 = plot(handles.anlss_ax , 1 , 1 , 'b') ;
        set(handles.h_Go_licks_frac_plot1,'XData',[],'YData',[])

        handles.h_Go_licks_frac_plot2 = plot(handles.anlss_ax , 1 , 1 , '*b') ;
        set(handles.h_Go_licks_frac_plot2,'XData',[],'YData',[])

        handles.h_NoGo_licks_frac_plot1 = plot(handles.anlss_ax , 1 , 1 , 'r') ;
        set(handles.h_NoGo_licks_frac_plot1,'XData',[],'YData',[])

        handles.h_NoGo_licks_frac_plot2 = plot(handles.anlss_ax , 1 , 1 , '*r') ;
        set(handles.h_NoGo_licks_frac_plot2,'XData',[],'YData',[])

        ylabel(handles.anlss_ax , '%' , 'fontsize' , 12)
        xlabel(handles.anlss_ax , ['Trials/' num2str(handles.Parameters.trials_per_anlss)]...
            , 'fontsize' , 12)
        handles.anlss_ax.XTickMode = 'auto' ;
        handles.anlss_ax.XTickLabelMode = 'auto' ;
        axis(handles.anlss_ax , [0 10 -5 110])
        
        legend(handles.anlss_ax,[handles.h_Go_licks_frac_plot2 handles.h_NoGo_licks_frac_plot2],...
            {'Hit rate','FA rate'},'Location','northwest')
        
        handles.anlss_ax.UserData = handles.anlss_type_menu.String{handles.anlss_type_menu.Value} ;
        
    end
    
    trial_types = get_trial_types(hObject,trial_count) ;
    
    XData = [] ;
    hit_YData = 0 ;
    CR_YData = 0 ;
    miss_YData = 0 ;
    FA_YData = 0 ;
        
    Go_YData = [] ;
    NoGo_YData = [] ;

    for trials = p.trials_per_anlss:p.trials_per_anlss:trial_count

        hit_YData(end+1) = sum(trial_types(1:trials) == 1) ; 
        CR_YData(end+1) = sum(trial_types(1:trials) == 2) ;       
        miss_YData(end+1) = sum(trial_types(1:trials) == 3) ; 
        FA_YData(end+1) = sum(trial_types(1:trials) == 4) ; 
        XData(end+1) = trials/p.trials_per_anlss ;

        Hit = hit_YData(end) - hit_YData(end-1) ;
        CR = CR_YData(end) - CR_YData(end-1) ;
        Miss = miss_YData(end) - miss_YData(end-1) ;
        FA = FA_YData(end) - FA_YData(end-1) ;

        Go_licks_frac = 100*Hit/(Hit + Miss) ;
        if isnan(Go_licks_frac) || isinf(Go_licks_frac)
            Go_licks_frac = 0 ;
        end

        NoGo_licks_frac = 100*FA/(FA + CR) ;
        if isnan(NoGo_licks_frac) || isinf(NoGo_licks_frac)
            NoGo_licks_frac = 0 ;
        end

        Go_YData(end+1) = Go_licks_frac ; 
        NoGo_YData(end+1) = NoGo_licks_frac ;

    end
    
    set(handles.h_Go_licks_frac_plot1 , 'XData' , XData , 'YData' , Go_YData)
    set(handles.h_Go_licks_frac_plot2 , 'XData' , XData , 'YData' , Go_YData)
    set(handles.h_NoGo_licks_frac_plot1 , 'XData' , XData , 'YData' , NoGo_YData)
    set(handles.h_NoGo_licks_frac_plot2 , 'XData' , XData , 'YData' , NoGo_YData )

    axis(handles.anlss_ax , [0 max([(trial_count/p.trials_per_anlss)+1 , 10]) -10 110])
    
    guidata(hObject, handles) ;
    
end

function plot_psychometric(hObject,trial_count)

    handles = guidata(hObject) ;
    p = handles.Parameters;
    
    if ~strcmp(handles.anlss_ax.UserData , handles.anlss_type_menu.String{handles.anlss_type_menu.Value})
        
        cla(handles.anlss_ax)
        
        handles.h_psych_plot1 = plot(handles.anlss_ax , 1 , 1 , 'k') ;
        set(handles.h_psych_plot1,'XData',[],'YData',[])

        handles.h_psych_plot2 = plot(handles.anlss_ax , 1 , 1 , '*k') ;
        set(handles.h_psych_plot2,'XData',[],'YData',[])

        ylabel(handles.anlss_ax , "Lick chance (%)" , 'fontsize' , 12)

        Ticks = p.Stims.names ;

        set(handles.anlss_ax , 'xtick' , 1:length(Ticks) , 'xticklabel' , Ticks)
        axis(handles.anlss_ax , [0 length(Ticks)+1 -5 105])
        
        legend(handles.anlss_ax,'Off')
        
        handles.anlss_ax.UserData = handles.anlss_type_menu.String{handles.anlss_type_menu.Value} ;
        
    end
    
    [~,stim_ids,~,trial_responses,~] = handles.Data.get_data(trial_count) ;
    
    YData = zeros(1,size(p.Stims,1)) ;
    for i = 1 : size(p.Stims,1)
        YData(i) = 100*sum(stim_ids == i & trial_responses == 1)/sum(stim_ids == i) ;
        if isinf(YData(i)) || isnan(YData(i))
            YData(i) = 0 ;
        end
    end

    
    set(handles.h_psych_plot1 , 'XData' , 1 : length(YData) , 'YData' , YData)
    set(handles.h_psych_plot2 , 'XData' , 1 : length(YData) , 'YData' , YData)
    
    guidata(hObject, handles) ;
            
end

function plot_response_latencies(hObject,trial_count)

    handles = guidata(hObject) ;
    p = handles.Parameters;
    
    if ~strcmp(handles.anlss_ax.UserData , handles.anlss_type_menu.String{handles.anlss_type_menu.Value})
        
        cla(handles.anlss_ax)
        
        handles.h_latency_plot1 = plot(handles.anlss_ax , 1 , 1 , 'k') ;
        set(handles.h_latency_plot1,'XData',[],'YData',[])

        handles.h_latency_plot2 = plot(handles.anlss_ax , 1 , 1 , '*k') ;
        set(handles.h_latency_plot2,'XData',[],'YData',[])

        ylabel(handles.anlss_ax , "Response latency (sec)" , 'fontsize' , 12)

        Ticks = p.Stims.names ;

        set(handles.anlss_ax , 'xtick' , 1:length(Ticks) , 'xticklabel' , Ticks)
        axis(handles.anlss_ax , [0 length(Ticks)+1 -0.2 p.response_window])
        
        legend(handles.anlss_ax,'Off')
        
        handles.anlss_ax.UserData = handles.anlss_type_menu.String{handles.anlss_type_menu.Value} ;
        
    end
    
    [~,stim_ids,~,trial_responses,response_latencies] = handles.Data.get_data(trial_count) ;
    
    YData = zeros(1,size(p.Stims,1)) ;
    for i = 1 : size(p.Stims,1)
        YData(i) = mean(response_latencies(stim_ids == i & trial_responses == 1)) ;
        if isinf(YData(i)) || isnan(YData(i))
            YData(i) = 0 ;
        end
    end

    
    set(handles.h_latency_plot1 , 'XData' , 1 : length(YData) , 'YData' , YData)
    set(handles.h_latency_plot2 , 'XData' , 1 : length(YData) , 'YData' , YData)
    
    guidata(hObject, handles) ;
    
end

function plot_trial_counts(hObject,trial_count)

    handles = guidata(hObject) ;
    p = handles.Parameters;
    
    if ~strcmp(handles.anlss_ax.UserData , handles.anlss_type_menu.String{handles.anlss_type_menu.Value})
        
        cla(handles.anlss_ax)
        
        handles.h_hit_plot1 = plot(handles.anlss_ax , 1 , 1 , 'b') ;
        set(handles.h_hit_plot1,'XData',[],'YData',[])

        handles.h_hit_plot2 = plot(handles.anlss_ax , 1 , 1 , '*b') ;
        set(handles.h_hit_plot2,'XData',[],'YData',[])

        handles.h_CR_plot1 = plot(handles.anlss_ax , 1 , 1 , 'Color' , [.3 .75 .93]) ;
        set(handles.h_CR_plot1,'XData',[],'YData',[])

        handles.h_CR_plot2 = plot(handles.anlss_ax , 1 , 1 , '*' , 'Color' , [.3 .75 .93]) ;
        set(handles.h_CR_plot2,'XData',[],'YData',[])

        handles.h_miss_plot1 = plot(handles.anlss_ax , 1 , 1 , 'r') ;
        set(handles.h_miss_plot1,'XData',[],'YData',[])

        handles.h_miss_plot2 = plot(handles.anlss_ax , 1 , 1 , '*r') ;
        set(handles.h_miss_plot2,'XData',[],'YData',[])

        handles.h_FA_plot1 = plot(handles.anlss_ax , 1 , 1 , 'Color' , [.85 .33 .1]) ;
        set(handles.h_FA_plot1,'XData',[],'YData',[])

        handles.h_FA_plot2 = plot(handles.anlss_ax , 1 , 1 , '*' , 'Color' , [.85 .33 .1]) ;
        set(handles.h_FA_plot2,'XData',[],'YData',[])

        ylabel(handles.anlss_ax , '#' , 'fontsize' , 12)
        xlabel(handles.anlss_ax , ['Trials/' num2str(handles.Parameters.trials_per_anlss)]...
            , 'fontsize' , 12)
        handles.anlss_ax.XTickMode = 'auto' ;
        handles.anlss_ax.XTickLabelMode = 'auto' ;
        axis(handles.anlss_ax , [0 10 -1 5])
        
        legend(handles.anlss_ax,[handles.h_hit_plot2 handles.h_miss_plot2 handles.h_CR_plot2 handles.h_FA_plot2],...
            {'Hit','Miss','CR','FA'},'Location','northwest')
        
        handles.anlss_ax.UserData = handles.anlss_type_menu.String{handles.anlss_type_menu.Value} ;
    end
    
    trial_types = get_trial_types(hObject,trial_count) ;
    
    XData = 0 ;
    hit_YData = 0 ;
    CR_YData = 0 ;
    miss_YData = 0 ;
    FA_YData = 0 ;
        
    for trials = p.trials_per_anlss:p.trials_per_anlss:trial_count

        hit_YData(end+1) = sum(trial_types(1:trials) == 1) ; 
        CR_YData(end+1) = sum(trial_types(1:trials) == 2) ;       
        miss_YData(end+1) = sum(trial_types(1:trials) == 3) ; 
        FA_YData(end+1) = sum(trial_types(1:trials) == 4) ; 
        XData(end+1) = trials/p.trials_per_anlss ;

    end

    set(handles.h_hit_plot1 , 'XData' , XData , 'YData' , hit_YData)
    set(handles.h_hit_plot2 , 'XData' , XData , 'YData' , hit_YData)
    set(handles.h_CR_plot1 , 'XData' , XData , 'YData' , CR_YData)
    set(handles.h_CR_plot2 , 'XData' , XData , 'YData' , CR_YData )
    set(handles.h_miss_plot1 , 'XData' , XData , 'YData' , miss_YData)
    set(handles.h_miss_plot2 , 'XData' , XData , 'YData' , miss_YData)
    set(handles.h_FA_plot1 , 'XData' , XData , 'YData' , FA_YData)
    set(handles.h_FA_plot2 , 'XData' , XData , 'YData' , FA_YData )

    axis(handles.anlss_ax , [0 max([(trial_count/p.trials_per_anlss)+1 , 10])...
        -1 max( [hit_YData(end) CR_YData(end) miss_YData(end) FA_YData(end)] ) + 5 ] )
    
    guidata(hObject, handles) ;
    
end

% ========================== Other functions =============================

function apply_lick(~,~,hObject)
    % Function apply_lick updates all relevant objects about licks and
    % updates lick axis. called by a listener
    %
    % Input:
    % 2 dummie variables (due to MATLAB's timer callback)
    % hObject - a GUI object
    % 

    handles = guidata(hObject) ;
    lick_time = toc(handles.Stimulus_generator.task_time) ;
    handles.Data.add_lick(lick_time)
    handles.Reinforcement_generator.add_lick(lick_time)
    if handles.Parameters.is_wait_for_no_lick
        handles.Stimulus_generator.delay_stim(handles.Parameters.no_lick_wait_time,0)
    end
    guidata(hObject,handles)
    update_lick_scat(hObject,true)
    
end

function apply_punishment(~,~,hObject)
    % Function apply_punishment updates all relevant objects about punishments,
    % updates reinforcement axis, and applies the relevant punishment.
    % called by a listener
    %
    % Input:
    % 2 dummie variables (due to MATLAB's listener callback)
    % hObject - a GUI object
    %

    handles = guidata(hObject) ;
    p = handles.Parameters;
    
    if ~handles.OutputSession.IsRunning
        
        time = toc(handles.Stimulus_generator.task_time) ;
        
        if p.is_timeout_punish
            handles.Stimulus_generator.delay_stim(p.timeout_punish_dur,1)
        end
        
        if p.is_repeat_incorrect            
            handles.Stimulus_generator.repeat_incorrect(handles.Data.stim_ids(end))
        end
        
        if p.is_noise_punish
            release(handles.OutputSession)
            queueOutputData(handles.OutputSession,handles.punishment_output)
            handles.OutputSession.startBackground
        end
        
        handles.Data.add_punishment(time)
        handles.Reinforcement_generator.update_punishment()
    end

    guidata(hObject,handles)
    update_reinforcement_scat(hObject,true)
    
end

function apply_reward(~,~,hObject)
    % Function apply_reward updates all relevant objects about rewards,
    % updates reinforcement axis and delivers the reward. called by listener
    %
    % Input:
    % 2 dummie variables (due to MATLAB's listener callback)
    % hObject - a GUI object
    %

    handles = guidata(hObject) ;

    if ~handles.OutputSession.IsRunning
        time = toc(handles.Stimulus_generator.task_time) ;
        release(handles.OutputSession)
        queueOutputData(handles.OutputSession,handles.reward_output)
        handles.OutputSession.startBackground
        handles.Stimulus_generator.repeat_incorrect(0)
        handles.Reinforcement_generator.update_reward()
        handles.Data.add_reward(time)
    end
    guidata(hObject,handles)
    update_reinforcement_scat(hObject,true)
    
end

function check_file_name(hObject)
    % Function check_file_name check if a file name is legal in WINDOWS 
    %
    % Input:
    % hObject - a GUI edit-text object
    %

    S = hObject.String ;
    Msg = '';

    BadChar = '<>:"/\|?*';
    BadName = {'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$', ...
             'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', ...
             'COM7', 'COM8', 'COM9', ...
             'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', ...
             'LPT7', 'LPT8', 'LPT9'};
    bad = ismember(BadChar, S);
    if any(bad)
       Msg = ['Name contains bad characters: ', BadChar(bad)];
    elseif any(S < 32)
       Msg = ['Name contains non printable characters, ASCII:', sprintf(' %d', S(S < 32))];
    elseif ~isempty(S) && (S(end) == ' ' || S(end) == '.')
       Msg = 'A trailing space or dot is forbidden';
    elseif isempty(S)
        Msg = 'A file name must be entered';
    else
       % "AUX.txt" fails also, so extract the file name only:
       [~, name] = fileparts(S);
       if any(strcmpi(name, BadName))
          Msg = ['Name not allowed: ', name];
       end
    end
    
    if ~isempty(Msg)
        errordlg(Msg , 'Invalid Input' , 'modal')
        set(hObject , 'String' , get(hObject,'UserData'))
    end
    
end

function check_positive(hObject)
    % Function check_positive checks if a number is positive 
    %
    % Input:

    input = str2double(get(hObject,'String'));

    if isnan(input) || input<=0

        errordlg('You must enter a numeric, positive value','Invalid Input','modal')
        set(hObject , 'String' , get(hObject,'UserData'))

    else

        set(hObject , 'UserData' , get(hObject,'String'))

    end 

end

function create_daq_sessions(hObject)
    % function create_daq_sessions creates input and output daq sessions,
    % adds channels and sets daq properties

    handles = guidata(hObject) ;
    p = handles.Parameters ;

    handles.InputSession = daq.createSession('ni') ;
    handles.OutputSession = daq.createSession('ni') ;
    addAnalogOutputChannel(handles.OutputSession , p.daq_dev, p.stim_ch , 'Voltage') ;
    addAnalogInputChannel(handles.InputSession ,  p.daq_dev , p.dummy_input_ch , 'Voltage') ;
    addDigitalChannel(handles.OutputSession ,  p.daq_dev , p.stimOn_ch , 'OutputOnly') ;
    addDigitalChannel(handles.InputSession ,  p.daq_dev , p.lick_ch , 'InputOnly') ;
    addDigitalChannel(handles.OutputSession ,  p.daq_dev , p.reward_ch , 'OutputOnly') ;
    addDigitalChannel(handles.OutputSession ,  p.daq_dev , p.trial_cue_ch , 'OutputOnly') ;
    handles.InputSession.Rate = p.acq_sample_rate * 1e3 ;
    handles.InputSession.IsContinuous = true ;
    handles.OutputSession.Rate = p.out_sample_rate * 1e3 ;
    handles.OutputSession.NotifyWhenScansQueuedBelow = handles.OutputSession.Rate/20 ;
    handles.InputSession.NotifyWhenDataAvailableExceeds = handles.InputSession.Rate / p.update_rate ;
    guidata(hObject,handles)
    
    % creating listener for data acquisition session
    handles.DataAvailable_listener = addlistener(handles.InputSession ,...
        'DataAvailable' , @(src,evnt)time_point(src,evnt,hObject)) ;
        
    guidata(hObject,handles)
    
end

function create_listeners(hObject)
    % function create_listeners creates all gui listeners

    handles = guidata(hObject) ;

    handles.Listeners{1} = addlistener(handles.Stimulus_generator , 'play_stim' ,...
        @(src,evnt)play_stim(src, evnt, hObject) ) ;

    handles.Listeners{2} = addlistener(handles.Reinforcement_generator , 'reward' ,...
        @(src,evnt)apply_reward(src, evnt, hObject) ) ;

    handles.Listeners{3} = addlistener(handles.Reinforcement_generator , 'punishment' ,...
        @(src,evnt)apply_punishment(src, evnt, hObject) ) ;

    handles.Listeners{4} = addlistener(handles.Parameters , 'progress_anlss' ,...
        @(src,evnt)update_anlss_ax(src, evnt, hObject) ) ;
    
    handles.Listeners{5} = addlistener(handles.Stimulus_generator , 'end_of_task' ,...
        @(src,evnt)end_task(src, evnt, hObject) ) ;

    
    guidata(hObject,handles)
            
end

function create_outputs(hObject)
% function create_outputs creates all ouputs for NI

handles = guidata(hObject) ;
p = handles.Parameters ;
Fs = p.out_sample_rate ;

shapes = p.Stims.shapes ;
handles.stims_output = {} ;

for i = 1 : size(p.Stims,1)
    stim = shapes{i} ;
    
    handles.stims_output{i} = [[stim ; 0] , [ones(size(stim)) ;0] ,...
        [zeros(size(stim)) ;0] , [zeros(size(stim)) ;0]] ;
    
    
    %if p.is_trial_cue && (p.Stims.types(i)== 1 | p.Stims.types(i)== 0 | p.Stims.types(i)== -1)
    
%             handles.stims_output{i} = [[stim ; 0] , [ones(size(stim)) ;0] ,...
%                 [zeros(size(stim)) ;0] , [ones(size(stim)) ;0]] ;

%       handles.stim_output{i} = [[stim ; 0] , [ones(size(stim)) ;0] ,...
%                 [zeros(size(stim)) ;0] , [ones(size(stim)) ;0]] ;
% 
% %for constant opto. 
% 
%             if i > 8 %i>2 for easy task | i>4 for hard task| i>10 for catch trial session
%             handles.stims_output{i} = [[zeros(Fs*p.trial_cue_onset,1),...
%                 zeros(Fs*p.trial_cue_onset,1),zeros(Fs*p.trial_cue_onset,1) ,...
%                 [ones(Fs*p.trial_cue_onset,1)]];handles.stim_output{i}; [zeros(Fs*p.trial_cue_onset,1),...
%                 zeros(Fs*p.trial_cue_onset,1),zeros(Fs*p.trial_cue_onset,1) ,...
%                 [ones(Fs*p.trial_cue_onset,1)]] ; zeros(1,4)]   ;
%             end 

% % % % for pulses of opto. 
   pulse_length = 25000 ; % 20 Hz
   pulse_on = ones(pulse_length./5,1)' ; %generate a template of pulse, 20 Hz- 10 ms pulse 
   pulse_off = zeros (pulse_length*(4/5),1)' ; %40 ms no pulse 
   pulse_vec = [pulse_on pulse_off] ; %complete template for one pulse 
   n_rep_opto = ((Fs*p.trial_cue_onset)/pulse_length) ; %devide duration of opto. (sample freaq.*duration of opto. cue) stim to calculate # of pulses needed 
   pulse_vec_opto = repmat(pulse_vec,1, n_rep_opto)' ; % repeat pulse template according to the # above. 
   n_rep_stim = ((size(stim,1))/pulse_length) ; %devide duration of stimulus (100ms) to calculate # of pulses needed 
   pulse_vec_stim = repmat(pulse_vec,1, n_rep_stim)' ; %generate seperate pulse vector for opto during stimulus
   
   if i > 7 %i>2 for easy task | i>4 for hard task| i>10 for catch trial session
       
    handles.stim_output{i} = [[stim]  , [ones(size(stim))] ,...
                [zeros(size(stim))] , pulse_vec_stim] ;


       handles.stims_output{i} = [[zeros(Fs*p.trial_cue_onset,1),...
           zeros(Fs*p.trial_cue_onset,1),zeros(Fs*p.trial_cue_onset,1) ,...
           pulse_vec_opto ];handles.stim_output{i}; [zeros(Fs*p.trial_cue_onset,1),...
           zeros(Fs*p.trial_cue_onset,1),zeros(Fs*p.trial_cue_onset,1) ,...
           pulse_vec_opto] ; zeros(1,4)]   ;
   end
   

%     pulse_length = 6250; % 20 Hz
%    pulse_on = ones(pulse_length./5,1)' ; %generate a template of pulse, 20 Hz- 10 ms pulse 
%    pulse_off = zeros (pulse_length*(4/5),1)' ; %40 ms no pulse 
%    pulse_vec = [pulse_on pulse_off] ; %complete template for one pulse 
%    n_rep_opto = ((Fs*p.trial_cue_dur)/pulse_length) ; %devide duration of opto. (sample freaq.*duration of opto. cue) stim to calculate # of pulses needed 
%    pulse_vec_opto = repmat(pulse_vec,1, n_rep_opto)' ; % repeat pulse template according to the # above. 
%    n_rep_stim = ((size(stim,1))/pulse_length) ; %devide duration of stimulus (100ms) to calculate # of pulses needed 
%    pulse_vec_stim = repmat(pulse_vec,1, n_rep_stim)' ; %generate seperate pulse vector for opto during stimulus
%    
%        
%     handles.stim_output{i} = [[stim]  , [ones(size(stim))] ,...
%                 [zeros(size(stim))] , pulse_vec_stim] ;
% 
% 
%     handles.stim_output{i} = [[zeros(Fs*p.trial_cue_dur,1),...
%            zeros(Fs*p.trial_cue_dur,1),zeros(Fs*p.trial_cue_dur,1) ,...
%            pulse_vec_opto ];handles.stim_output{i};] ;
%    

   
   %             handles.stims_output{i} =[[zeros(Fs*p.trial_cue_onset,1);0],...
%                 [zeros(Fs*p.trial_cue_onset,1); 0],...
%                 [zeros(Fs*p.trial_cue_onset,1);0] ,...
%                 [ones(Fs*p.trial_cue_onset,1); 0],...
%                 ;handles.stim_output{i};]% ...
%                       [zeros(Fs*p.trial_cue_dur,1);0],...
%                 [zeros(Fs*p.trial_cue_dur,1); 0],...
%                 [zeros(Fs*p.trial_cue_dur,1);0] ,...
%                 [ones(Fs*p.trial_cue_dur,1);0]];
%               
             

            
    %end
           
           if p.is_auto_reward && (p.Stims.types(i)==1)
           
           %if p.is_auto_reward && (p.Stims.types(i)==1)| p.is_auto_reward && (p.Stims.types(i)==2)
            handles.stims_output{i} = [handles.stims_output{i} ;...
                [zeros(Fs*p.reward_dur,1) ,zeros(Fs*p.reward_dur,1),...
                ones(Fs*p.reward_dur,1),zeros(Fs*p.reward_dur,1)] ; zeros(1,4)] ;
            
    end
    
    
    % create reward output
    handles.reward_output = [[zeros(Fs*p.reward_dur,1) ;0],...
        [zeros(Fs*p.reward_dur,1) ;0],[ones(Fs*p.reward_dur,1) ;0] ,...
        [zeros(Fs*p.reward_dur,1) ;0]] ;
    
    % create white noise output as punishment
    Noise = p.max_voltage * (-1 + 2*rand(Fs*p.noise_punish_dur*1e3,1)) ;
    handles.punishment_output = [[Noise ;0] , [zeros(size(Noise)) ;0] ,...
        [zeros(size(Noise)) ;0] , [zeros(size(Noise)) ;0]];
    
    
    guidata(hObject,handles) ;
end
end

function end_task(~,~,hObject)
    % function end_task stops the task as in stop_task and disables start
    % button. called by listener
    %
    % Input:
    % 2 dummie variables (due to MATLAB's listener callback)
    % hObject - a GUI object
    %
    
    handles = guidata(hObject) ;
    stop_task(hObject)
    handles.Start_button.Enable = 'off' ;

end

function fix_ITI(hObject)
    % function fix_ITI changes the ITI if necessary (it is too short for
    % all the in-between)
    
    handles = guidata(hObject) ;
    p = handles.Parameters ;
       
    ITI = str2double(handles.ITI.String) ; 
    ITI_range = str2double(handles.ITI_range.String) ;
    window = str2double(handles.response_window.String) ;
    stim_dur = p.longest_dur ;
    trial_cue_dur = str2double(handles.trial_cue_onset.String) * handles.trial_cue_check.Value * 1e-3 ;
    reward_dur = str2double(handles.reward_dur.String) * handles.auto_reward_check.Value * 1e-3 ;
    
    min_ITI = ITI_range + window + stim_dur + trial_cue_dur + reward_dur + 0.5 ;
    
    if ITI < min_ITI
        warndlg('ITI is too short, changing it to the minimum' , 'Changing ITI' , 'modal')
        handles.ITI.String = num2str(min_ITI) ;
        handles.ITI.UserData = num2str(min_ITI) ;
    end
    
    guidata(hObject,handles)
    
end

function full_axes(hObject)
    % function full_axes loads all data in to all axes for slider
    % navigation once the task is stopped

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    update_stim_rect(hObject,false)
    update_window_rect(hObject,false)
    if p.is_trial_cue
        update_trial_cue_rect(hObject,false)
    end
    update_lick_scat(hObject,false)
    update_reinforcement_scat(hObject,false)
    
    guidata(hObject, handles) ;
end

function trial_types = get_trial_types(hObject,trial_count)

    handles = guidata(hObject) ;

    [~,~,stim_types,trial_responses,~] = handles.Data.get_data(trial_count) ;
    trial_types = zeros(size(stim_types)) ;
    trial_types(stim_types == 1 & trial_responses == 1) = 1 ;
    trial_types(stim_types == -1 & trial_responses == 0) = 2 ;
    trial_types(stim_types == 1 & trial_responses == 0) = 3 ;
    trial_types(stim_types == -1 & trial_responses == 1) = 4 ;
    
end

function initialize_raw_ax(hObject)
    % Function initialize_raw_ax brings raw_ax to initial codition
    % where appearance is adjusted but x and y data are empty
    %
    % Input:
    % hObject - a GUI object
    %

    handles = guidata(hObject) ;
    p = handles.Parameters ;

    handles.Ticks = {'Punishment' , 'Reward' , 'Licks' , 'Catch' , 'NoGo' , 'Go'} ;
    grid(handles.raw_ax , 'on')
    hold(handles.raw_ax , 'on')
    set(handles.raw_ax , 'ytick' , 1:length(handles.Ticks) , 'yticklabel' , handles.Ticks)
    xlabel(handles.raw_ax , 'Time (seconds)' , 'fontsize' , 12)

    cla(handles.raw_ax)
    cla(handles.anlss_ax)

    y_lim = [0 length(handles.Ticks)+1] ;
    axis(handles.raw_ax , [-p.time_span 0 y_lim])

    handles.h_trial_cue_rect = patch(handles.raw_ax) ;
    set(handles.h_trial_cue_rect , 'XData' , [] , 'YData' , [] ,'FaceColor', [0.8 0.8 0.8])
    
    handles.h_window_rect = patch(handles.raw_ax) ;
    set(handles.h_window_rect , 'XData' , [] , 'YData' , [] ,'FaceColor', [1 1 1])

    handles.h_lick_scatter = scatter(handles.raw_ax , [] , [] ,...
        p.scatter_size , [0 0.5 0] , 'd' , 'filled' , 'MarkerEdgeColor' , 'k') ;

    handles.h_reward_scatter = scatter(handles.raw_ax , [] , [] ,...
        p.scatter_size , 'c' , 'd' , 'filled' ,...
        'MarkerEdgeColor' , 'k') ;

    handles.h_punishment_scatter = scatter(handles.raw_ax , [] , [] ,...
        p.scatter_size , 'r' , 'd' , 'filled' ,...
        'MarkerEdgeColor' , 'k') ;
    
    for i = 1 : size(p.Stims)

        handles.h_stim_rect{i} = patch(handles.raw_ax) ;
        set(handles.h_stim_rect{i} , 'XData' , [] , 'YData' , [] ,'FaceColor', [p.Stims.R(i) p.Stims.G(i) p.Stims.B(i)])
        
    end

    guidata(hObject,handles)

end

function match_parameters(hObject,File)

    handles = guidata(hObject) ;
    
    load(File)
    
    Parameters = Data.p ;
    
    handles.data_dir_button.UserData = Parameters.data_dir ;
    handles.mouse.String = Parameters.mouse ;
    
    handles.ITI.String = num2str(Parameters.ITI) ;
    handles.ITI_range.String = num2str(Parameters.ITI_range) ;
    handles.response_window.String = num2str(Parameters.response_window) ;
    handles.reinforcement_delay.String = num2str(Parameters.reinforcement_delay) ;
    handles.task_dur.String = num2str(Parameters.task_dur) ;
    handles.lick_threshold.String = num2str(Parameters.lick_threshold) ;

    handles.auto_reward_check.Value = Parameters.is_auto_reward ;
    handles.wait_for_no_lick.Value = Parameters.is_wait_for_no_lick ;
    handles.no_lick_wait_time.String = num2str(Parameters.no_lick_wait_time) ;
    handles.noise_punish_check.Value = Parameters.is_noise_punish ;
    handles.noise_punish_dur.String = num2str(Parameters.noise_punish_dur) ;
    handles.timeout_punish_check.Value = Parameters.is_timeout_punish ;
    handles.timeout_punish_dur.String = num2str(Parameters.timeout_punish_dur) ;
    handles.repeat_incorrect_check.Value = Parameters.is_repeat_incorrect ;

    handles.trial_cue_check.Value = Parameters.is_trial_cue ;
    handles.trial_cue_dur.String = num2str(Parameters.trial_cue_dur) ;
    handles.trial_cue_onset.String = num2str(Parameters.trial_cue_onset) ;

    handles.reward_dur.String = num2str(Parameters.reward_dur) ;
    
    enable_str = {'Off','On'} ;
    
    handles.no_lick_wait_time.Enable = enable_str{handles.wait_for_no_lick_check.Value+1} ;
    handles.noise_punish_dur.Enable = enable_str{handles.noise_punish_check.Value+1} ;
    handles.timeout_punish_dur.Enable = enable_str{handles.timeout_punish_check.Value+1} ;
    handles.trial_cue_dur.Enable = enable_str{handles.trial_cue_check.Value+1} ;
    handles.trial_cue_onset.Enable = enable_str{handles.trial_cue_check.Value+1} ;
    
    handles.Parameters.Stims = Parameters.Stims ;
    
    update_stimuli_menus(hObject)
    handles.Go_stims_menu.Enable = 'On' ;
    handles.NoGo_stims_menu.Enable = 'On' ;
    handles.catch_stims_menu.Enable = 'On' ;
    present_stimulus_details(hObject,1)    
    handles.start_button.Enable = 'On' ;
    
end

function object_enable(hObject,str)
    % Function object_enable , enables or disables relevant GUI objects 
    %
    % Input:
    % handles - GUI handles
    % str - string , either 'on' or 'off'
    %

    handles = guidata(hObject) ;

    set( findall(handles.main_fig, '-property', 'Enable'), 'Enable', str)


    handles.start_button.Enable = 'on' ;        
    handles.debug_button.Enable = 'on' ;
    handles.anlss_type_menu.Enable = 'on' ;        
    handles.reward_button.Enable = 'on' ;        
    handles.Data_cursor.Enable = 'on' ;


    handles.stim_name_txt.Enable = 'On' ;
    handles.stim_type_txt.Enable = 'On' ;
    handles.stim_prob_txt.Enable = 'On' ;
    handles.stim_dur_txt.Enable = 'On' ;
    handles.stim_color_txt.Enable = 'On' ;

    if ~handles.wait_for_no_lick_check.Value
        handles.no_lick_wait_time.Enable = 'Off' ;
    end
    if ~handles.noise_punish_check.Value
        handles.noise_punish_dur.Enable = 'Off' ;
    end
    if ~handles.timeout_punish_check.Value
        handles.timeout_punish_dur.Enable = 'Off' ;
    end
    if ~handles.trial_cue_check.Value
        handles.trial_cue_dur.Enable = 'Off' ;
        handles.trial_cue_onset.Enable = 'Off' ;
    end
        
    if isempty(handles.Go_stims_menu.UserData)
        handles.Go_stims_menu.Enable = 'Off' ;
    end
    
    if isempty(handles.NoGo_stims_menu.UserData)    
        handles.NoGo_stims_menu.Enable = 'Off' ;
    end
    
    if isempty(handles.catch_stims_menu.UserData)
        handles.catch_stims_menu.Enable = 'Off' ;
    end

end

function play_stim(~,evnt_data,hObject)
    % Function play_stim updates all relevant objects about the stimulus.
    % called by a listener
    %
    % Input:
    % dummie variable (due to MATLAB's listener callback)
    % evnt_data - an event data object that specifies whether to present
    % odor before tone
    % hObject - a GUI object
    %

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    if ~handles.OutputSession.IsRunning

        release(handles.OutputSession)
        time = toc(handles.Stimulus_generator.task_time) + p.trial_cue_onset * p.is_trial_cue*1e-3 ;

        stim_dur = p.Stims.durations(evnt_data.stim_id) ;

        queueOutputData(handles.OutputSession,handles.stims_output{evnt_data.stim_id})
        handles.OutputSession.startBackground

        handles.Data.add_stim(time , evnt_data.stim_id)
        handles.Reinforcement_generator.update_last_stim(time , evnt_data.stim_id)
        handles.Parameters.increase_trial_count()

        if p.is_auto_reward && p.Stims.types(evnt_data.stim_id)== 1 
        %if p.is_auto_reward && p.Stims.types(evnt_data.stim_id)== 1 | p.is_auto_reward && p.Stims.types(evnt_data.stim_id)== 2
            handles.Data.add_reward(time + stim_dur*1e-3)
            update_reinforcement_scat(hObject,true)
        end

        update_stim_rect(hObject,true)
        update_window_rect(hObject,true)
        if p.is_trial_cue
            update_trial_cue_rect(hObject,true)
        end

        present_stimulus_details(hObject,evnt_data.stim_id)

        guidata(hObject,handles)
        
    end
    
end

function plot_stim(hObject,stim)

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    Fs = p.out_sample_rate ; % kHz
    dur = length(stim)/Fs ; % ms
    
    handles.stim_ax1 = subplot(2 , 1 , 1 , 'Parent' , handles.stim_shape_panel) ;
    plot(gca,(1:length(stim))/Fs,stim)
    ylabel(gca,'Volt. (V)','fontsize',10)
    set(gca,'XTick',[])
    axis(gca,[0 dur -10 10])
    handles.stim_ax2 = subplot(2 , 1 , 2 , 'Parent' , handles.stim_shape_panel) ;
    spectrogram(stim,floor(length(stim)/9),floor(length(stim)/10),2048,Fs*1e3,'yaxis')
    set(gca,'YScale','log','YTick',2.^(2:6))
    ylim(gca,[4 100])
    ylabel(gca,'Freq. (kHz)','fontsize',10)
    colorbar(gca,'off')
    
    guidata(hObject,handles)
    
end

function present_stimulus_details(hObject,stim_id)
    
    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    %types = {'NoGo','Catch','Go'} ;
    
    types = {'NoGo','Catch','Go'} ;
    
    names = p.Stims.names ;
    stim_shapes = p.Stims.shapes ;
    handles.stim_name_txt.String = names{stim_id} ;
    handles.stim_type_txt.String = types{p.Stims.types(stim_id) + 2} ;
    handles.stim_prob_txt.String = num2str(p.Stims.probabilities(stim_id)) ;
    handles.stim_dur_txt.String = [sprintf('%0.0f',p.Stims.durations(stim_id)) ' ms'] ;
    handles.stim_color_txt.ForegroundColor = [p.Stims.R(stim_id) p.Stims.G(stim_id) p.Stims.B(stim_id)] ;
    
    if ~handles.start_button.UserData
        plot_stim(hObject,stim_shapes{stim_id})
    end
    
end

function save_data(handles)
    % Function save_data saves all important data 
    %
    % Input:
    % handles - a GUI handles object
    %	

    p = handles.Parameters ;
    
    % making sure file does not exist and if it does add a number
    
    date = char(datetime('now','Format','yyyy-MM-dd')) ;
    mouse_dir = [p.data_dir '\' p.mouse] ;
    date_dir = [mouse_dir '\' [date(3:4) date(6:7) date(9:10)]] ;
    if ~exist(date_dir,'dir')
        mkdir(date_dir)
    end
    file_name = fullfile(date_dir , p.data_file) ;
    if exist([file_name '.mat'] , 'file') == 2
        orig_file_name = file_name ;
        file_name = [file_name '_1'] ;
    end
    n = 1 ;
    while exist([file_name '.mat'] , 'file') == 2
        n = n + 1 ;
        file_name = [orig_file_name '_' num2str(n)] ;
    end
       
    handles.Data.save_data([file_name '.mat'])
    
end

function stop_task(hObject)
    % function stop_task stops all gui objects , and saves data if
    % necessary

    handles = guidata(hObject) ;
    p = handles.Parameters ;

    % changing start/stop task button
    handles.start_button.UserData = 0 ;
    handles.start_button.String = 'Start' ;                             
    handles.start_button.BackgroundColor = [.47 .67 .19] ; % button text is now green

    handles.Stimulus_generator.stop_run()

    handles.InputSession.stop()

    if p.is_save_data
        save_data(handles) % saving data
    end

    full_axes(hObject)

    handles.new_button.Enable = 'on' ;
    handles.start_button.Enable = 'off' ;
    handles.reward_button.Enable = 'on' ;
    handles.data_slider.Enable = 'on' ;

    guidata(hObject, handles) ;
        
end

function time_point(~,event,hObject)
    % Function time_point updates all relevant objects about a new
    % time point, updates axes, and checks for licks since the last time point 
    %
    % Input:
    % dummie variable (due to MATLAB's timer callback)
    % an event in case of a data acquisition callback (with information
    % about inputs, and specifically licks)
    % hObject - a GUI object
    %
    % Usage example:
    % time_point(0,0,hObject)
    %


    handles = guidata(hObject);
    
    new_point = toc(handles.Stimulus_generator.task_time) ;
    
    handles.Data.propogate_time(new_point)

    % checking the buffered data for licks
    if sum(diff(event.Data(:,2)) == 1) >= handles.Parameters.n_pulses
        apply_lick(0,0,hObject)
    end
    
    % moving axes
    xlim(handles.raw_ax , [-handles.Parameters.time_span 0] + new_point)    
    
end

function update_lick_scat(hObject,is_running)
    % Function update_lick_scat updates the x and y data of the lick scatter 
    %
    % Input:
    % hObject - a GUI object
    % is_running - a logical, whether the task is running
    %
	

    handles = guidata(hObject) ;
    
    if is_running
        x_lims = handles.raw_ax.XLim ;
        bottom_lim = x_lims(1) - 5 ;
        top_lim = x_lims(2) + 5 ;
    else
        bottom_lim = 0 ;
        top_lim = inf ;
    end
    
    x_data = handles.Data.lick_times(handles.Data.lick_times > bottom_lim &...
        handles.Data.lick_times < top_lim) ;
    tick_loc = find(strcmp('Licks',handles.Ticks)) ;
    set(handles.h_lick_scatter , 'XData' , x_data , 'YData' , tick_loc * ones(1 , length( x_data )) ) ;
       
    guidata(hObject, handles) ;

end 

function update_trial_cue_rect(hObject,is_running)
    % Function update_trial_cue_rect updates the odor rectangles before tones 
    %
    % Input:
    % hObject - a GUI object
    % is_running - a logical, whether the task is running
    %

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    if is_running
        x_lims = handles.raw_ax.XLim ;
        bottom_lim = x_lims(1) - 5 ;
        top_lim = x_lims(2) + 5 ;
    else
        bottom_lim = 0 ;
        top_lim = inf ;
    end
    
    tick_locs = [find(strcmp('NoGo',handles.Ticks)),...
                 find(strcmp('Catch',handles.Ticks)),...
                 find(strcmp('Go',handles.Ticks))] ;
             
    stim_times = handles.Data.stim_times(handles.Data.stim_times > bottom_lim &...
        handles.Data.stim_times < top_lim) ;
    stim_ids = handles.Data.stim_ids(handles.Data.stim_times > bottom_lim &...
        handles.Data.stim_times < top_lim) ;
    trial_cue_times = stim_times - p.trial_cue_onset * 1e-3 ;
    stim_types = p.Stims.types(stim_ids) ;
    stim_ticks = tick_locs(stim_types + 2) ;

    x_data = [trial_cue_times ; trial_cue_times ;...
        trial_cue_times + p.trial_cue_dur*1e-3 ; trial_cue_times + p.trial_cue_dur*1e-3] ;

    y_data = [(stim_ticks-0.2) ; (stim_ticks+0.2) ; (stim_ticks+0.2) ; (stim_ticks-0.2)] ;
    
    set(handles.h_trial_cue_rect , 'XData' , x_data , 'YData' , y_data ) ;
       
    guidata(hObject, handles) ;

end

function update_reinforcement_scat(hObject,is_running)
    % Function update_reinforcement_scat updates the x and y data of the
    % reinforcement scatter 
    %
    % Input:
    % hObject - a GUI object
    % is_running - a logical, whether the task is running
    %	

    handles = guidata(hObject) ;
    
    if is_running
        x_lims = handles.raw_ax.XLim ;
        bottom_lim = x_lims(1) - 5 ;
        top_lim = x_lims(2) + 5 ;
    else
        bottom_lim = 0 ;
        top_lim = inf ;
    end
    
    x_data = handles.Data.reward_times(handles.Data.reward_times > bottom_lim &...
        handles.Data.reward_times < top_lim ) ;
    tick_loc = find(strcmp('Reward',handles.Ticks)) ;
    set(handles.h_reward_scatter , 'XData' , x_data , 'YData' , tick_loc*ones(1 , length( x_data )) ) ;
    
    x_data = handles.Data.punishment_times(handles.Data.punishment_times > bottom_lim &...
        handles.Data.punishment_times < top_lim ) ;
    tick_loc = find(strcmp('Punishment',handles.Ticks)) ;
    set(handles.h_punishment_scatter , 'XData' , x_data , 'YData' , tick_loc*ones(1 , length( x_data )) ) ;
        
    guidata(hObject, handles) ;

end

function update_stim_rect(hObject,is_running)
    % Function update_stim_rect updates the x and y data of the stim rectangles 
    %
    % Input:
    % hObject - a GUI object
    % is_running - a logical, whether the task is running
    %	

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    if is_running
        x_lims = handles.raw_ax.XLim ;
        bottom_lim = x_lims(1) - 5 ;
        top_lim = x_lims(2) + 5 ;
    else
        bottom_lim = 0 ;
        top_lim = inf ;
    end
    
    tick_locs = [find(strcmp('NoGo',handles.Ticks)),...
                 find(strcmp('Catch',handles.Ticks)),...
                 find(strcmp('Go',handles.Ticks))] ;
             
     for i = 1 : size(p.Stims,1)
    
        stim_times = handles.Data.stim_times(handles.Data.stim_times > bottom_lim &...
            handles.Data.stim_times < top_lim) ;
        stim_ids = handles.Data.stim_ids(handles.Data.stim_times > bottom_lim &...
            handles.Data.stim_times < top_lim) ;
        stim_times = stim_times(stim_ids==i) ;
        stim_durations = p.Stims.durations(i) * ones(size(stim_times)) ;
        stim_type = p.Stims.types(i) ;
        stim_ticks = tick_locs(stim_type + 2) * ones(size(stim_times)) ;
    
        x_data = [stim_times ; stim_times ; stim_times + stim_durations*1e-3 ; stim_times + stim_durations*1e-3] ;
        y_data = [(stim_ticks-0.2) ; (stim_ticks+0.2) ; (stim_ticks+0.2) ; (stim_ticks-0.2)] ;

        set(handles.h_stim_rect{i} , 'XData' , x_data , 'YData' , y_data ) ;
        
     end
        
    guidata(hObject, handles) ;

end

function update_stimuli_menus(hObject)

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    names = p.Stims.names ;
    
    handles.Go_stims_menu.UserData = find(p.Stims.types == 1) ;
    %handles.Go_stims_menu.UserData = find(p.Stims.types == 1 | p.Stims.types == 2) ;
    if ~isempty(handles.Go_stims_menu.UserData)
        handles.Go_stims_menu.String = names(handles.Go_stims_menu.UserData) ;
        handles.Go_stims_menu.Enable = 'On' ;
    else
        handles.Go_stims_menu.String = 'Go stimuli...' ;
        handles.Go_stims_menu.Enable = 'Off' ;
    end
    
    handles.NoGo_stims_menu.UserData = find(p.Stims.types == -1) ;
        %handles.NoGo_stims_menu.UserData = find(p.Stims.types == -1 | p.Stims.types == -2) ;

    if ~isempty(handles.NoGo_stims_menu.UserData)    
        handles.NoGo_stims_menu.String = names(handles.NoGo_stims_menu.UserData) ;
        handles.NoGo_stims_menu.Enable = 'On' ;
    else
        handles.NoGo_stims_menu.String = 'NoGo stimuli...' ;
        handles.NoGo_stims_menu.Enable = 'Off' ;
    end
    
    handles.catch_stims_menu.UserData = find(p.Stims.types == 0) ;
    if ~isempty(handles.catch_stims_menu.UserData)
        handles.catch_stims_menu.String = names(handles.catch_stims_menu.UserData) ;
        handles.catch_stims_menu.Enable = 'On' ;
    else
        handles.catch_stims_menu.String = 'Catch stimuli...' ;
        handles.catch_stims_menu.Enable = 'Off' ;
    end
    
end

function update_window_rect(hObject,is_running)
    % Function update_window_rect updates the response_window rectangles after tones 
    %
    % Input:
    % hObject - a GUI object
    % is_running - a logical, whether the task is running
    %

    handles = guidata(hObject) ;
    p = handles.Parameters ;
    
    if is_running
        x_lims = handles.raw_ax.XLim ;
        bottom_lim = x_lims(1) - 5 ;
        top_lim = x_lims(2) + 5 ;
    else
        bottom_lim = 0 ;
        top_lim = inf ;
    end
    
    tick_locs = [find(strcmp('NoGo',handles.Ticks)),...
             find(strcmp('Catch',handles.Ticks)),...
             find(strcmp('Go',handles.Ticks))] ;
             
    stim_times = handles.Data.stim_times(handles.Data.stim_times > bottom_lim &...
        handles.Data.stim_times < top_lim) ;
    stim_ids = handles.Data.stim_ids(handles.Data.stim_times > bottom_lim &...
        handles.Data.stim_times < top_lim) ;
    stim_durations = reshape(p.Stims.durations(stim_ids),1,[]) ;
    window_times = stim_times + stim_durations*1e-3 ;
    stim_types = p.Stims.types(stim_ids) ;
    stim_ticks = tick_locs(stim_types + 2) ;

    x_data = [window_times ; window_times ; window_times + p.response_window ; window_times + p.response_window] ;

    y_data = [(stim_ticks-0.2) ; (stim_ticks+0.2) ; (stim_ticks+0.2) ; (stim_ticks-0.2)] ;
    
    set(handles.h_window_rect , 'XData' , x_data , 'YData' , y_data ) ;
       
    guidata(hObject, handles) ;

end

% ========================== Creation functions ==========================

function anlss_type_menu_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
end

function catch_stims_menu_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

end

function data_slider_CreateFcn(hObject, eventdata, handles)

    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    % Creating slider listener
    handles.slider_listener = addlistener(hObject, 'Value' , 'PostSet' ,...
        @altered_slider_Callback);
    
    guidata(hObject,handles)
    
end

function file_name_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;

    
end

function Go_stims_menu_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
end

function ITI_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function ITI_range_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function lick_threshold_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function mouse_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function no_lick_wait_time_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
end

function NoGo_stims_menu_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
end

function noise_punish_dur_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function response_window_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function reinforcement_delay_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
end

function reward_dur_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function task_dur_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function timeout_punish_dur_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function trial_cue_dur_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

function trial_cue_onset_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    hObject.UserData = hObject.String ;
    
end

% ========================== Debug function ==============================

function debug_button_Callback(hObject, eventdata, handles)

    handles ;
    
end
