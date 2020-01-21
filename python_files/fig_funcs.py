def dist_plot_fig()
    '''Seabor'''
    
    plt.figure(figsize=(10, 6))
    sns.distplot(auto_df['TARGET_AMT'],bins=66, hist=True, kde=False, color='#2196f3')
    plt.xlabel('Target Amount')
    plt.xlim(left=)

    plt.show()