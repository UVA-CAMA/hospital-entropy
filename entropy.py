import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from plotnine.data import *
import seaborn as sns

def entropy_by_lab(df, prior = "none", lab_name = "All Labs", dept_names = ["All Departments"], window_size = 28, step_size = 7, include_3am = True, smooth = False):
    """
    Calculates the shannon entropy for a lab over time
    df: a dataframe containing four components:
    dept_name: list for departments lab is ordered in
    lab_name = column that identifies the name of lab 
    order_date = date time column in format %Y%m%d
    hour_ordered = variable defining one hour block lab is ordered in. Can be engineered from date/time ordered
    window_size: int, size of window for analysis in days
    step_size: int, size of step of the moving window
    df_prior: str, default is all, using all data for prior distribution for KL divergence. Otherwise input a date in the format "YYYY-MM-dd"
    returns a dataframe containing the date and entropy for that day
    """
    if prior == "none":
        df_prior = df
    else:
        df_prior = df[df["order_date"] < prior]
        df = df[df["order_date"] > prior]
    # set accumulating df list
    dfs = []
    kl_dfs = []
    ks_dfs = []
    totals_ordered_df = []
    for dept_name in dept_names:
        if dept_name == "All Departments":
            df_filtered = df
            df_filtered_prior = df_prior
        else:
            df_filtered = df[df["dept"] == dept_name]
            df_filtered_prior = df_prior[df_prior["dept"] == dept_name]
        # filter for specific lab you are analyzing
        if lab_name == "All Labs":
            pass
        else:
            df_filtered = df_filtered[df_filtered["lab_name"]==lab_name]
            df_filtered_prior = df_filtered_prior[df_filtered_prior["lab_name"]==lab_name]
        # get groups for each individual day
        days = df_filtered.groupby("order_date")
        # list to store value for each day
        shannon_entropy_by_week = []
        # list to store value for each day
        kl_divergence_over_time = []
        ks_entropy_over_time = []
        # list to store number of labs taken each day 
        labs_by_week = []
        # calculate probability distribution for all time
        probs_all_time = []
        for hour in range(24):
            probs_all_time.append(np.sum(df_filtered["hour_ordered"]==hour)/(len(df_filtered))+ 0.0001) 
        weeks = []
        # loop through each day
        accumulator = 0
        week = []
        for i in range(len(days)):
            accumulator += 1
            # grab day we are analyzing
            day = days.get_group((list(days.groups)[i]))
            # get counts for the hours the labs were ordered
            values_day = []
            for hour in range(24):
                values_day.append(np.sum(day["hour_ordered"]==hour))
            week.append(values_day)
            if accumulator == window_size:
                week_total = [sum(k) for k in zip(*week)]
                # add one to each hour to avoid log of 0 
                week_total = [x+0.0001 for x in week_total]
                weeks.append(week_total)
                week = week[step_size:]
                accumulator = accumulator-step_size
        for i in range(len(weeks)):
            # grab day we are analyzing
            week = weeks[i]
            # get counts for the hours the labs were ordered
            if i != 0:
                prev_week = weeks[i-1]
            else:
                prev_week = week
            # set up baseline accumulator for later loop
            kl_divergence = 0
            ks_entropy = 0
            # another accumulator for shannon smoothed
            shannon_entropy = 0
            total_labs_x_weeks = sum(week)
            labs_by_week.append(total_labs_x_weeks)
            total_labs_previous_bin = sum(prev_week)
            #print(len(labs_by_hour_9_day))
            for l in range(len(week)):
                # get probabilities for each hour bins
                probability_previous_window = prev_week[l]/total_labs_previous_bin
                probability_x_weeks = week[l]/total_labs_x_weeks
                # calculate KS entropy of different distributions with -plog(p/q)
                entropy = np.log(probability_previous_window/probability_x_weeks) * probability_previous_window
                shannon_entropy_week = -np.log(probability_x_weeks) *probability_x_weeks
                kl_divergence_week = np.log(probs_all_time[l]/probability_x_weeks)*probs_all_time[l]
                # add to total for the day 
                ks_entropy += entropy
                kl_divergence += kl_divergence_week
                shannon_entropy += shannon_entropy_week
            # add the totaled KS entropy to the one over time
            kl_divergence_over_time.append(kl_divergence)
            ks_entropy_over_time.append(ks_entropy)
            # add total to smooth shannon 
            shannon_entropy_by_week.append(shannon_entropy)
            # take out the first observation in this list of 10 days
        # get unique list for days    
        days = np.unique(df_filtered.order_date)
        days = days[window_size:]
        days = days[::step_size]
        time_frame_kl_divergence = (len(kl_divergence_over_time))
        kl_days = days
        #kl_days = days[-time_frame_kl_divergence:]
        #print(len(days))
        # create a result df. This will contain two columns. Date and shannon entropy for that day
        # make column for department
        dept = [dept_name] * len(days)
        kl_dept = [dept_name] * len(kl_days)
        ordered_dept = [dept_name] * len(labs_by_week)
        # create results df and append it to list of dfs 
        results_df = pd.DataFrame(list(zip(days, shannon_entropy_by_week, dept)),
                columns =['Date', 'Entropy', 'Dept'])
        dfs.append(results_df)
        #results_df_smooth = pd.DataFrame(list(zip(kl_days, shannon_entropy_by_day_smoothed, kl_dept)),
        #        columns =['Date', 'Entropy', 'Dept'])
        #dfs_smoothed.append(results_df_smooth)
        results_df_kl = pd.DataFrame(list(zip(kl_days, kl_divergence_over_time, kl_dept)),
                columns =['Date', 'KL_Divergence', 'Dept'])
        kl_dfs.append(results_df_kl)
        results_df_ks = pd.DataFrame(list(zip(kl_days, ks_entropy_over_time, kl_dept)),
            columns =['Date', 'KS_Entropy', 'Dept'])
        ks_dfs.append(results_df_ks)
        total_labs_df = pd.DataFrame(list(zip(days, labs_by_week, ordered_dept)),
                columns =['Date', 'Count', 'Dept'])
        totals_ordered_df.append(total_labs_df)
    # combine dataframes for different departments for plotting    
    entropy_df = pd.concat(dfs)
    kl_df = pd.concat(kl_dfs)
    count_df = pd.concat(totals_ordered_df)
    #entropy_df_smoothed = pd.concat(dfs_smoothed)
    ks_df = pd.concat(ks_dfs)
    combined_df = pd.merge(entropy_df, count_df)
    
    # mesner: 2023-08-17: convert to categorical to preserve ordering in legend
    entropy_df["Dept"] = pd.Categorical(entropy_df["Dept"], categories=dept_names)
    kl_df["Dept"] = pd.Categorical(kl_df["Dept"], categories=dept_names)
    #mesneer: 2024-01-04: hack to take diff of entropies for KS entropy
    ks_df["KS_Entropy"] = entropy_df["Entropy"].rolling(8).mean()-entropy_df["Entropy"].rolling(4).mean()
    combined_df["Dept"] = pd.Categorical(combined_df["Dept"], categories=dept_names)
    
    # create plot over time
    if smooth == True:
        # p = (ggplot(entropy_df, aes(x = "Date", y = "Entropy", color = 'Dept')) +
        #     geom_smooth() +
        #     scale_y_continuous(limits = (0,3.2), breaks=(0,0.5,1,1.5,2,2.5,3,3.5)) +
        #     labs(x='Date', y = "Entropy", title = "Entropy of "+lab_name+" Over Time"))
        # print(p)
        s = (ggplot(entropy_df, aes(x = "Date", y = "Entropy", color = 'Dept')) +
            geom_smooth() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            scale_y_continuous(limits = (0,3.2), breaks=(0,0.5,1,1.5,2,2.5,3,3.5)) +
            labs(x='Date', y = "Shannon Entropy", title = "Shannon Entropy of "+lab_name+" - UVA"))
        print(s)
        q = (ggplot(kl_df, aes(x = "Date", y = "KL_Divergence", color = 'Dept')) +
            geom_smooth() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            #scale_y_continuous(limits = (0,1), breaks=(0,0.2,0.4,0.6,0.8,1)) +
            labs(x='Date', y = "KL Divergence", title = "KL Divergence of "+lab_name+" - UVA"))
        print(q)
        r = (ggplot(count_df, aes(x = "Date", y = "Count", color = 'Dept')) +
            geom_smooth() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            #scale_y_continuous(limits = (0,0.05), breaks=(0,0.01,0.02,0.03,0.04,0.05)) +
            labs(x='Date', y = "Count", title = "Total Amount of "+lab_name+" Ordered - UVA"))
        print(r)
        t = (ggplot(combined_df, aes(x = "Count", y = "Entropy", color = 'Dept')) +
            geom_point() +
            ylab("Shannon Entropy") +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            scale_y_continuous(limits = (0,3.2), breaks=(0,0.5,1,1.5,2,2.5,3,3.5)) +
            labs(x='Count', y = "Shannon entropy", title = "Shannon Entropy of "+lab_name+" By Amount Ordered - UVA"))
        print(t)
        u = (ggplot(ks_df, aes(x = "Date", y = "KS_Entropy", color = 'Dept')) +
            geom_smooth() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            #scale_y_continuous(limits = (0,0.05), breaks=(0,0.01,0.02,0.03,0.04,0.05)) +
            labs(x='Count', y = "KS Entropy", title = "KS Entropy of "+lab_name+" - UVA"))
        print(u)
    else:
        # p = (ggplot(entropy_df, aes(x = "Date", y = "Entropy", color = 'Dept')) +
        #     geom_line() +
        #     scale_y_continuous(limits = (0,np.log(24)), breaks=(np.log(1),np.log(2),np.log(3), np.log(4), np.log(5),np.log(6), np.log(9), np.log(12), np.log(15), np.log(18), np.log(21), np.log(24))) +
        #     labs(x='Date', y = "Entropy", title = "Entropy of "+lab_name+" Over Time"))
        # print(p)
        s = (ggplot(entropy_df, aes(x = "Date", y = "Entropy", color = 'Dept')) +
            geom_line() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            scale_y_continuous(limits = (0,3.2), breaks=(0,0.5,1,1.5,2,2.5,3,3.5)) +
            labs(x='Date', y = "Shannon Entropy", title = "Shannon Entropy of "+lab_name+" - UVA"))
        s.save(filename = "out/Shannon Entropy of "+lab_name+" - UVA" +'.png', height=5, width=5, units = 'in', dpi=1000)
        print(s)
        
        q = (ggplot(kl_df, aes(x = "Date", y = "KL_Divergence", color = 'Dept')) +
            geom_line() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            #scale_y_continuous(limits = (0,1), breaks=(0,0.2,0.4,0.6,0.8,1)) +
            #scale_y_continuous(limits = (0,0.1), breaks=(0,0.02,0.04,0.06,0.08,0.1)) +
            labs(x='Date', y = "KL Divergence", title = "KL Divergence of "+lab_name+" - UVA"))
        q.save(filename = "out/KL_Divergence of "+lab_name+" - UVA" +'.png', height=5, width=5, units = 'in', dpi=1000)
        print(q)
        
        r = (ggplot(count_df, aes(x = "Date", y = "Count", color = 'Dept')) +
            geom_line() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            #scale_y_continuous(limits = (0,0.05), breaks=(0,0.01,0.02,0.03,0.04,0.05)) +
            labs(x='Date', y = "Count", title = "Total Amount of "+lab_name+" Ordered - UVA"))
        r.save(filename = "out/Count of "+lab_name+" - UVA" +'.png', height=5, width=5, units = 'in', dpi=1000)
        print(r)
        
        # color_values = dict(ED="orange red", STICU="light green", MICU="green",NICU="medium slate blue", Floor="orchid")
        # color_values_list = ["orange red", "light green", "green", "medium slate blue", "orchid"]
        color_values_list = ["#ff4500", "#90EE90", "#008000", "#7b68ee", "#DA70D6"]

        t = (ggplot(combined_df, aes(x = "Count", y = "Entropy", color = 'Dept')) +
            geom_point() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            scale_y_continuous(limits = (0,3.2), breaks=(0,0.5,1,1.5,2,2.5,3,3.5)) +
            scale_x_log10(limits=(100,13000), labels = lambda l: [f"{x:n}" for x in l]) +
            scale_color_manual(values=color_values_list) +
            labs(x='Count', y = "Shannon entropy", title = "Shannon Entropy of "+lab_name+" By Amount Ordered - UVA"))
        t.save(filename = "out/Shannon Entropy of "+lab_name+" By Amount Ordered - UVA" +'.png', height=6, width=10, units = 'in', dpi=300)
        print(t)
        
        u = (ggplot(ks_df, aes(x = "Date", y = "KS_Entropy", color = 'Dept')) +
            geom_line() +
            theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            #scale_y_continuous(limits = (0,0.05), breaks=(0,0.01,0.02,0.03,0.04,0.05)) +
            labs(x='Count', y = "KS Entropy", title = "KS Entropy of "+lab_name+" - UVA"))
        u.save(filename = "out/KS Entropy "+lab_name+" - UVA" +'.png', height=5, width=5, units = 'in', dpi=1000)
        print(u)

        ##
        ## 1-combimed plot for shannon entropy and count 
        ##
        # Generate the individual plots
        fig, ax1 = plt.subplots()  # Create the figure and the first y-axis
        plt.xticks(rotation=45)
        ax2 = ax1.twinx()  # Create the second y-axis

        # Plot for Shannon Entropy
        color_palette = plt.get_cmap("Set3")
        entropy_handles = []
        for i, dept in enumerate(dept_names):
            dept_data = entropy_df[entropy_df["Dept"] == dept]
            line, = ax1.plot(dept_data["Date"], dept_data["Entropy"], color=color_palette(i))
            entropy_handles.append(line)
        
        ax1.set_ylabel("Shannon Entropy")
        ax1.set_ylim(0, 3.2)
        
        # Plot for Count
        count_handles = []
        for i, dept in enumerate(dept_names):
            dept_data = count_df[count_df["Dept"] == dept]
            bar = ax2.bar(dept_data["Date"], dept_data["Count"], alpha=0.6, color=color_palette(i))
            count_handles.append(bar[0])
        
        ax2.set_ylabel("Count")
        # ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(), rotation=45, ha='right')
       
        # Set x-axis labels and title
        # ax1.set_xlabel("Date")
        ax1.set_title(f"Shannon Entropy vs count of {lab_name} - UVA")
        # ax2.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Create a combined legend with custom labels inside the plot on the left side
        combined_legend_labels = [plt.Line2D([], [], color=color_palette(i)) for i in range(len(dept_names))]
        combined_legend = ax1.legend(combined_legend_labels, dept_names, loc='center left', bbox_to_anchor=(0, 0.5),  borderaxespad=0.,prop={'size': 7})
        ax1.add_artist(combined_legend)
        plt.savefig("out/combined_shannon_count_"+lab_name+".png")
        
        plt.show()

    return entropy_df, count_df, combined_df, kl_df, entropy_df

def sodium_entropy_over_time(df_covid_hosp: pd.DataFrame, df_entropy: pd.DataFrame, filename: str):
    """
    Plots a timeseries of Shannon entropy over time, with a shaded region of COVID-19 hospitalizations
    df_covid_hosp: a pandas dataframe containing 'Date' and 'Weekly COVID-19 Hospital Admissions' columns
    df_entropy: a pandas dataframe containing 'Date' and 'Entropy' columns
    filename: where to save the image
    """
    df_covid_hosp_filtered = df_covid_hosp[(df_covid_hosp["Date"] >= np.min(df_entropy["Date"])) & (df_covid_hosp["Date"] <= np.max(df_entropy["Date"]))]

    s = ggplot(df_entropy, aes(x = "Date", y = "Entropy", color = 'Dept')) + \
                geom_line() + \
                theme_bw() + \
                theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1), legend_box_spacing=(0.15)) + \
                scale_y_continuous(limits = (0,3.2), breaks=(0,0.5,1,1.5,2,2.5,3,3.5)) + \
                labs(x='Date', y = "Shannon entropy", title = "Shannon Entropy of Sodium Over Time - UVA")
    fig = s.draw()
    axs = fig.get_axes()[0]
    axs_covid = axs.twinx()
    axs_covid.set_ylim(0,4500)
    axs_covid.fill_between(df_covid_hosp_filtered.Date, 0, df_covid_hosp_filtered["Weekly COVID-19 Hospital Admissions"], color="lightgrey", alpha=0.4)
    axs_covid.set_ylabel("Counts of COVID hospitalizations", labelpad=10, rotation=270)
    fig.savefig(filename)
    return


def create_surprisal_heatmaps(df, lab_list, dept_list, by_lab = True):
    """
    Plots heatmaps for surprisal values by lab or by department, saving each image separately
    df: a pandas dataframe with the following
    dept_name: list for departments lab is ordered in
    lab_name = column that identifies the name of lab 
    order_date = date time column in format %Y%m%d
    hour_ordered = variable defining one hour block lab is ordered in. Can be engineered from date/time ordered
    by_lab: a boolean indicating whether to plot surprisal by lab or by department
    """
    #sns.set(rc={'figure.figsize':(11.7,8.27)})
    #np.seterr(divide='ignore')
    hours_column_names = ["12 AM", "01 AM", "02 AM", "03 AM", "04 AM", "05 AM", "06 AM",
"07 AM", "08 AM", "09 AM", "10 AM", "11 AM", "12 PM", "01 PM", "02 PM", "03 PM", "04 PM", 
"05 PM", "06 PM", "07 PM", "08 PM", "09 PM", "10 PM", "11 PM"]
    sns.set(rc={'figure.figsize':(8.7,5.27)})
    #sns.set(rc={'axes.facecolor':'black'})
    #sns.set_facecolor("black")
    if by_lab:
        labs_probs_lists = []
        labs_surprisal_lists = []
        labs_entropy_lists = []
        for i in lab_list:
            if i != "All Labs":
                lab_df = df[df["lab_name"]==i]
            else:
                lab_df = df
            dept_prob_lists = []
            dept_surprisal_lists = []
            dept_entropy_lists = []
            for j in dept_list:
                lab_and_dept_df = lab_df[lab_df["dept"]==j]
                lab_values = lab_and_dept_df["hour_ordered"].value_counts()
                total_labs = sum(lab_values)
                list_of_probs = []
                surprisal_list = []
                shannon_entropy_by_dept = 0
                for k in range(24):
                    hour_df =  lab_and_dept_df[lab_and_dept_df["hour_ordered"]==k]
                    number_ordered_in_hour = len(hour_df)
                    probability_per_hour = number_ordered_in_hour/(total_labs or 1)
                    surprisal_per_hour = -np.log(probability_per_hour)
                    if probability_per_hour != 0:
                        shannon_entropy_per_hour = -np.log(probability_per_hour) * probability_per_hour
                    else:
                        shannon_entropy_per_hour = 0
                    shannon_entropy_by_dept += shannon_entropy_per_hour
                    list_of_probs.append(probability_per_hour)
                    surprisal_list.append(surprisal_per_hour)
                dept_prob_lists.append(list_of_probs)
                dept_surprisal_lists.append(surprisal_list)
                dept_entropy_lists.append(shannon_entropy_by_dept)
            prob_df = pd.DataFrame(dept_prob_lists)
            surprisal_df = pd.DataFrame(dept_surprisal_lists)
            prob_df.columns = hours_column_names
            surprisal_df.columns = hours_column_names
            prob_df.index = dept_list
            surprisal_df.index = dept_list
            surprisal_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            plt.figure()
            heatmap = sns.heatmap(prob_df, cmap="magma", vmin = 0, vmax = 0.3)
            heatmap.set_facecolor("black")
            heatmap.set(xlabel='Time of Day', ylabel='Department', title = "Probability Heatmap For " +i+ " Across Departments - UVA")
            plt.yticks(rotation=0)
            fig = heatmap.get_figure()
            fig.savefig(f"out/heatmap_prob_{i}.png")
            #print(heatmap)
            #heatmap.clf()
            plt.figure()
            heatmap = sns.heatmap(surprisal_df, cmap="magma", vmin =0, vmax = 6)
            heatmap.set_facecolor("black")
            heatmap.set(xlabel='Time of Day', ylabel='Department', title = "Surprisal Heatmap For " +i+ " Across Departments - UVA")
            plt.yticks(rotation=0)
            fig = heatmap.get_figure()
            fig.savefig(f"out/heatmap_surprise_{i}.png")
            #print(heatmap)
            #heatmap.clf()
            labs_probs_lists.append(dept_prob_lists)
            labs_surprisal_lists.append(dept_surprisal_lists)
            labs_entropy_lists.append(dept_entropy_lists)
        # create dataframe for shannon entropy with labels 
        lab_entropies_df = pd.DataFrame(labs_entropy_lists)
        lab_entropies_df.columns = dept_list
        lab_entropies_df.index = lab_list
        plt.figure()
        #create stacked bar chart
        #sns.set(style='white')
        ax = lab_entropies_df.plot(kind='bar', stacked=True, colormap="magma", figsize=(15,10))
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2 + bar.get_y(),
                    round(bar.get_height(), ndigits=3), ha = 'center',
                    color = 'gray', weight = 'bold', size = 10)
        lab_entropies_df.to_csv("out/shannon_all_labs_by_dept.csv",index=True)
        # lab_entropies_df.plot(kind='bar', stacked=True, color = ["#663399", "#7953a9", "#8b74bd", "#b9bfff", "#4066e0"], colormap="magma")
        #add overall title
        plt.title('Shannon Entropy all labs by dept', fontsize=16)
        #add axis titles
        plt.xlabel('Lab')
        plt.ylabel('Shannon Entropy')
        #rotate x-axis labels
        plt.xticks(rotation=90)
        plt.legend(labels = dept_list[::-1],handles = reversed(plt.legend().legendHandles), loc=1)
        plt.tight_layout()
        # plt.legend(labels = dept_list[::-1],handles = reversed(plt.legend().legendHandles), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig("out/shannon_all_labs_by_dept.png")
        #return labs_entropy_lists
        
    else:
        dept_prob_lists = []
        dept_surprisal_lists = []
        dept_entropy_lists = []
        for i in dept_list:
            dept_df = df[df["dept"]==i]
            labs_probs_lists = []
            labs_surprisal_lists = []
            labs_entropy_lists = []
            for j in lab_list:
                if j != "All Labs":
                    lab_and_dept_df = dept_df[dept_df["lab_name"]==j]
                else:
                    lab_and_dept_df = dept_df
                #lab_and_dept_df = dept_df[dept_df["lab_name"]==j]
                lab_values = lab_and_dept_df["hour_ordered"].value_counts()
                total_labs = sum(lab_values)
                list_of_probs = []
                surprisal_list = []
                shannon_entropy_by_lab = 0
                for k in range(24):
                    hour_df =  lab_and_dept_df[lab_and_dept_df["hour_ordered"]==k]
                    number_ordered_in_hour = len(hour_df)
                    probability_per_hour = number_ordered_in_hour/(total_labs or 1)
                    surprisal_per_hour = -np.log(probability_per_hour)
                    if probability_per_hour != 0:
                        shannon_entropy_per_hour = -np.log(probability_per_hour) * probability_per_hour
                    else:
                        shannon_entropy_per_hour = 0
                    shannon_entropy_by_lab += shannon_entropy_per_hour
                    list_of_probs.append(probability_per_hour)
                    surprisal_list.append(surprisal_per_hour)
                labs_probs_lists.append(list_of_probs)
                labs_surprisal_lists.append(surprisal_list)
                labs_entropy_lists.append(shannon_entropy_by_lab)
            prob_df = pd.DataFrame(labs_probs_lists)
            surprisal_df = pd.DataFrame(labs_surprisal_lists)
            prob_df.columns = hours_column_names
            surprisal_df.columns = hours_column_names
            prob_df.index = lab_list
            surprisal_df.index = lab_list
            surprisal_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            plt.figure()
            heatmap = sns.heatmap(prob_df, cmap="magma", vmin= 0, vmax = 0.3)
            heatmap.set_facecolor("black")
            heatmap.set(xlabel='Time of Day', ylabel='Department', title = "Probability Heatmap For " +i+ " Across Labs - UVA")
            plt.yticks(rotation=0)
            fig = heatmap.get_figure()
            fig.savefig(f"out/heatmap_prob_{i}.png")
            plt.figure()
            heatmap = sns.heatmap(surprisal_df, cmap="magma", vmin = 0, vmax = 6)
            heatmap.set_facecolor("black")
            heatmap.set(xlabel='Time of Day', ylabel='Department', title = "Surprisal Heatmap For " +i+ " Across Labs - UVA")
            plt.yticks(rotation=0)
            fig = heatmap.get_figure()
            fig.savefig(f"out/heatmap_surprise_{i}.png")
            #print(heatmap)
            #heatmap.clf()
            dept_prob_lists.append(labs_probs_lists)
            dept_surprisal_lists.append(labs_surprisal_lists)
            dept_entropy_lists.append(labs_entropy_lists)
        # create dataframe for shannon entropy with labels 
        dept_entropies_df = pd.DataFrame(dept_entropy_lists)
        dept_entropies_df.columns = lab_list
        dept_entropies_df.index = dept_list
        plt.figure()
        #create stacked bar chart
        #sns.set(style='white')
        # dept_entropies_df.plot(kind='bar', stacked=True, colormap="magma", figsize=(15,10))
        ax = dept_entropies_df.plot(kind='bar', stacked=True, colormap='magma', figsize=(15,10))
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2 + bar.get_y(),
                    round(bar.get_height(), ndigits=3), ha = 'center',
                    color = 'gray', weight = 'bold', size = 10)
        dept_entropies_df.to_csv("out/shannon_all_depts_by_lab.csv",index=True)
        #add overall title
        plt.title('Shannon Entropy all departments by lab', fontsize=16)
        #add axis titles
        plt.xlabel('Department')
        plt.ylabel('Shannon Entropy')
        #rotate x-axis labels
        plt.xticks(rotation=90)
        #sns.move_legend(plt, "upper left", bbox_to_anchor=(1, 1))
        #fig, ax = plt.subplots()
        #handles, labels = ax.get_legend_handles_labels()
        plt.legend(labels = lab_list[::-1],handles = reversed(plt.legend().legendHandles), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig("out/shannon_all_depts_by_lab.png",  bbox_inches="tight")
        #plt.legend(labels = lab_list[::-1],handles = lab_list[::-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

