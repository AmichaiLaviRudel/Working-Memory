
clear all
close all

addpath ('Z:\Shared\GNG')
load("C:\Users\Owner\Desktop\Rec_OFC_2.mat")
%% Set parameters
time = Data.time; 
stim_times = Data.stim_times;
stim_ids = Data.stim_ids;
stim_types = Data.stim_types;
lick_times = Data.lick_times;
trial_responses = Data.trial_responses;
reward_times = Data.reward_times;
punishment_times = Data.punishment_times;
response_latencies = Data.response_latencies;
p = Data.p;
Stims = p.Stims;
Stims_name = Stims.names;
%% Psychometric Data
    YData = zeros(1,size(p.Stims,1)) ;
    for i = 1 : size(p.Stims,1)
        YData(i) = 100*sum(stim_ids == i & trial_responses == 1)/sum(stim_ids == i) ;
        if isinf(YData(i)) || isnan(YData(i))
            YData(i) = 0 ;
        end
    end
    
XData = [8.49 11.89]

    figure
    axis tight
    
set(gca,'XScale','log'); 
set(gca, 'XDir','reverse');
    plot ( XData, YData,'--o','Color','k')
    set(gca, xlim = ([7 14])) 
    set(gca, ylim =([0 100]))
    ylabel('lickrate')
    xlabel('freq')
    title ('psychometric curve')

%% Response Latencies 
    YData = zeros(1,size(p.Stims,1)) ;
    for i = 1 : size(p.Stims,1)
        YData(i) = mean(response_latencies(stim_ids == i & trial_responses == 1)) ;
        if isinf(YData(i)) || isnan(YData(i))
            YData(i) = 0 ;
        end
    end
    
    figure 
    plot (YData(1:2),'--o','Color','g')
    hold on
    plot (YData(3:4),'--o','Color','b')

   
        ylim = ([0 1])
        ylabel('time(ms)')
        xlabel ('8.409','9.1708','10.503','11.89')
        legend ('0.25 Oct','0.5 Oct')
        title ('response latencies')

    
%% Trial Responses

    XData = 0 ;
    hit_YData = 0 ;
    CR_YData = 0 ;
    miss_YData = 0 ;
    FA_YData = 0 ;       
    

    trial_types = zeros(size(stim_types)) ;
        trial_types(stim_types == 1 & trial_responses == 1) = 1 ;
   for trials = 1:length(trial_responses)
        hit_YData(end+1) = sum(trial_types(1:trials) == 1) ; 
   end
    
    trial_types(stim_types == -1 & trial_responses == 0) = 2 ;
       for trials = 1:length(trial_responses)
        CR_YData(end+1) = sum(trial_types(1:trials) == 2) ;   
    end 
  
    trial_types(stim_types == 1 & trial_responses == 0) = 3 ;
        for trials = 1:length(trial_responses)
        miss_YData(end+1) = sum(trial_types(1:trials) == 3) ; 
        end
    
      trial_types(stim_types == -1 & trial_responses == 1) = 4 ;
            for trials = 1:length(trial_responses)
        FA_YData(end+1) = sum(trial_types(1:trials) == 4) ; 
            end
    
        
    for trials = 1:length(trial_responses)
        XData(end+1) = trials;
    end 
    
    
    figure 
    plot (XData,hit_YData,'g')
    hold on
    plot (XData,CR_YData,'b')
    hold on 
    plot (XData,miss_YData,'r')
    hold on
    plot (XData,FA_YData,'y')
    legend ('hit','CR','miss','FA')
    xlabel ('n trials total')
    ylabel ('n trials per trial type')
    
    
 

    
%% Lick Rates
  XData = [] ;
    hit_YData = 0 ;
    CR_YData = 0 ;
    miss_YData = 0 ;
    FA_YData = 0 ;
        
    Go_YData = [] ;
    NoGo_YData = [] ;
     
    trial_types = zeros(size(stim_types)) ;
    trial_types(stim_types == 1 & trial_responses == 1) = 1 ;
    trial_types(stim_types == -1 & trial_responses == 0) = 2 ;
    trial_types(stim_types == 1 & trial_responses == 0) = 3 ;
    trial_types(stim_types == -1 & trial_responses == 1) = 4 ;

    for trials = 1:50:length(trial_responses)
        trial_types(stim_types == 1 & trial_responses == 1) = 1 ;
        hit_YData(end+1) = sum(trial_types(1:trials) == 1) ; 
        trial_types(stim_types == -1 & trial_responses == 0) = 2 ;
        CR_YData(end+1) = sum(trial_types(1:trials) == 2) ;
        trial_types(stim_types == 1 & trial_responses == 0) = 3 ;
        miss_YData(end+1) = sum(trial_types(1:trials) == 3) ;
        trial_types(stim_types == -1 & trial_responses == 1) = 4 ;
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
    
    figure
    plot (XData,Go_YData,'b','Marker','o')
    hold on
    plot (XData,NoGo_YData,'r','Marker','o')
    ylim([0 110])
    yticks([ 20 40 60 80 100])
    ylabel('% of trials')
    xlabel('trialbins')

   
%% d'prime 
    XData = 0 ;
    hit_YData = 0 ;
    CR_YData = 0 ;
    miss_YData = 0 ;
    FA_YData = 0 ;

    dprime_YData = [] ;
    
    trial_types = zeros(size(stim_types)) ;
    trial_types(stim_types == 1 & trial_responses == 1) = 1 ;
    trial_types(stim_types == -1 & trial_responses == 0) = 2 ;
    trial_types(stim_types == 1 & trial_responses == 0) = 3 ;
    trial_types(stim_types == -1 & trial_responses == 1) = 4 ;


    for trials = 1:10:length(trial_responses)
        
        trial_types(stim_types == 1 & trial_responses == 1) = 1 ;
        hit_YData(end+1) = sum(trial_types(1:trials) == 1) ;
        
        trial_types(stim_types == -1 & trial_responses == 0) = 2 ;
        CR_YData(end+1) = sum(trial_types(1:trials) == 2) ;  
        
        trial_types(stim_types == 1 & trial_responses == 0) = 3 ;
        miss_YData(end+1) = sum(trial_types(1:trials) == 3) ; 
        
        trial_types(stim_types == -1 & trial_responses == 1) = 4 ;
        FA_YData(end+1) = sum(trial_types(1:trials) == 4) ; 
        
       % XData(end+1) = trials/p.trials_per_anlss ;

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
    
    for trials = 10:10:length(trial_responses)
            XData(end+1) = trials/p.trials_per_anlss ;
    end 

     XData = 1:length (dprime_YData)
 
    figure 
    plot (XData, dprime_YData,'Color','k','LineStyle','-','Marker','o')
    xlabel('trial bins')
    ylabel('dprime')
    xlim([10 length(XData)+10])
    ylim([ -1 4])
    yticks([ 0 1 2 3 4])
    
%%
