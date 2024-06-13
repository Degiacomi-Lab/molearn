#%%
# Import libraries
import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.foldingnet import AutoEncoder
import torch
from molearn.analysis import MolearnAnalysis
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from molearn.utils import as_numpy
import pandas as pd
import seaborn as sns
import numpy as np
# %%
# Import data
if __name__ == '__main__':
    # what follows is a method to re-create the training and test set
    # by defining the manual see and loading the dataset in the same order as when
    #the neural network was trained, the same train-test split will be obtained
    data = PDBData()
    data.import_pdb(f'data{os.sep}MurD_closed_selection.pdb')
    data.import_pdb(f'data{os.sep}MurD_open_selection.pdb')
    #data.import_pdb('/home3/pghw87/trajectories/MurD/MurD_closed.pdb')
    #data.import_pdb('/home3/pghw87/trajectories/MurD/MurD_open.pdb')
    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    data.prepare_dataset()
    data_train, data_valid = data.split(manual_seed=25)
    data_test = PDBData()
    data_test.import_pdb(f'data{os.sep}MurD_closed_apo_selection.pdb')
    data_test.std = data.std
    data_test.mean = data.mean
    data_test.fix_terminal()
    data_test.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    data_test.prepare_dataset()

#%%
# Load network, test it, and get training, validation, and test errors
def load_models(architecture, learning_rates): 
    results = []
    base_dir = os.getcwd()

    for learning_rate in learning_rates:
        folder_name = f'siren_checkpoints_{architecture}_e-{learning_rate}'
        full_path = os.path.join(base_dir, folder_name)

        for fname in os.listdir(full_path):
            if fname.startswith('checkpoint') :
                checkpoint_path = os.path.join(full_path, fname)
                checkpoint = torch.load(checkpoint_path, map_location = torch.device('cuda'))

                net = AutoEncoder(**checkpoint['network_kwargs'])
                net.load_state_dict(checkpoint['model_state_dict'])

                MA = MolearnAnalysis()
                MA.set_network(net)
                MA.batch_size = 4
                MA.processes = 2
                
                MA.set_dataset("training", data_train)
                MA.set_dataset("valid", data_valid)
                MA.set_dataset('test', data_test)

                err_train = MA.get_error('training')
                err_valid = MA.get_error('valid')
                err_test = MA.get_error('test')

                results.append({
                    'architecture': architecture,
                    'learning_rate': f'e-{learning_rate}',
                    'checkpoint_name': fname,
                    'train_error': err_train,
                    'valid_error': err_valid,
                    'test_error': err_test

                })
    return results

#%%
# Define parameters
architectures = ['3x512', '4x512', '3x1024']
learning_rates = ['4', '5', '6']
all_models = []

# #%%
# Create the list of results for all architectures and lrs 
for architecture in architectures:
    all_models.append(load_models(architecture, learning_rates))

#%%
# Convert lists into dataframe 
flat_models = [item for sublist in all_models for item in sublist]
df_models = pd.DataFrame(flat_models)
# Create a new column that uniquely identifies each architecture-learning rate-checkpoint combination (model ID)
df_models['arch_lr_iter'] = df_models.apply(lambda x: f"arch_{x['architecture']}_lr{x['learning_rate']}_iter{((x.name)%5)+1}", axis=1)
#%%
df_models['mean_test_error'] = df_models['test_error'].apply(lambda x: sum(x) / len(x))
mean_arch_lr = df_models.groupby('arch_lr')['mean_test_error'].idxmin()
df_models_unique = df_models.loc[mean_arch_lr]
# %%
# Create seperate dictionaries: key being model ID and value being train, valid, and test errors 
train_error_dict = pd.Series(df_models.train_error.values,index=df_models.arch_lr_iter).to_dict()
valid_error_dict = pd.Series(df_models.valid_error.values,index=df_models.arch_lr_iter).to_dict()
test_error_dict = pd.Series(df_models.test_error.values,index=df_models.arch_lr_iter).to_dict()

# %%
# Use dictionary values and reverse the list for plotting
train_values = list(train_error_dict.values())
valid_values = list(valid_error_dict.values())
test_values = list(test_error_dict.values())

#%%
### Compressed version of violin benchmark plot
fig, ax = plt.subplots(figsize=(10, 15))
positions = range(1, len(train_error_dict) + 1)

violin_train = ax.violinplot(train_values, vert = False, showmeans = True, )
violin_valid = ax.violinplot(valid_values, vert = False, showmeans = True, )
violin_test = ax.violinplot(test_values, vert = False, showmeans = True, )

def set_axis_style(ax, labels):
    ax.get_yaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels, rotation=45)
    ax.set_ylim(0.25, len(labels) + 0.75)
    ax.set_ylabel('Trained model')

# Use dictionary keys as labels and reverse the list for matching plot order
labels = list(train_error_dict.keys())
set_axis_style(ax, labels)

ax.set_title('RMSD of training, validation and test set')

fig.gca().set_xlabel(r'RMSD ($ \AA$)')
plt.savefig('RMSD_plot.png')

plt.show()
# %%
### Extended version of violin benchmarck plot
# Get the values from the dictionaries
train_values = list(train_error_dict.values())
valid_values = list(valid_error_dict.values())
test_values = list(test_error_dict.values())

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 50))

# Define the positions for each set
num_models = len(train_error_dict)
positions = np.arange(1, num_models * 3, step=3)  # Spacing of 3 units between each model

# Adjust positions for the validation and test sets
valid_positions = [p + 1 for p in positions]
test_positions = [p + 2 for p in positions]

# Create the violin plots
violin_train = ax.violinplot(train_values, positions=positions, vert=False, widths=0.9, showmeans=True)
violin_valid = ax.violinplot(valid_values, positions=valid_positions, vert=False, widths=0.9, showmeans=True)
violin_test = ax.violinplot(test_values, positions=test_positions, vert=False, widths=0.9, showmeans=True)

# Define the axis styling function
def set_axis_style(ax, labels):
    ax.get_yaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    # We triple the number of y-ticks because we have three sets now
    ax.set_yticks(np.arange(2, num_models * 3, step=3))  # Position ticks in between the groups
    ax.set_yticklabels(labels, rotation=45)
    ax.set_ylim(0, num_models * 3+1)
    ax.set_ylabel('Trained model')

# Set the labels
labels = list(train_error_dict.keys())
set_axis_style(ax, labels)


# Set the title and labels
ax.set_title('RMSD of training, validation and test set')

fig.gca().set_xlabel(r'RMSD ($ \AA$)')

# Add legend
# Setting the colors for the violins
colors = ['blue', 'orange', 'green']
for vp, color in zip([violin_train, violin_valid, violin_test], colors):
    for partname in ('bodies', 'cbars','cmins','cmaxes','cmeans'):
        vp_part = vp[partname]
        if partname == 'bodies':  # 'bodies' is a list of patches
            for patch in vp_part:
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
        else:  # Other parts are lines
            vp_part.set_edgecolor(color)

# Create legend patches
legend_patches = [
    Patch(color=colors[0], label='Training'),
    Patch(color=colors[1], label='Validation'),
    Patch(color=colors[2], label='Test')
]

# Add the legend to the plot
ax.legend(handles=legend_patches, loc='upper right', title='Legend')
# Save and show the plot
plt.savefig('benchmark-plot.png')
plt.show()

#%%
#### Create sub-plots to compare 5 iterations
df_models['arch_lr'] = df_models.apply(lambda x: f"arch_{x['architecture']}_lr{x['learning_rate']}", axis=1)

train_error_list = []
valid_error_list = []
test_error_list = []

for arch_lr, group in df_models.groupby('arch_lr'):

    train_error_iter_dict = {}
    valid_error_iter_dict = {}
    test_error_iter_dict = {}

    for iteration in group['arch_lr_iter'].unique():
        iteration_data = group[group['arch_lr_iter'] == iteration]
        
        train_error_iter_dict[iteration] = iteration_data['train_error'].tolist()
        valid_error_iter_dict[iteration] = iteration_data['valid_error'].tolist()
        test_error_iter_dict[iteration] = iteration_data['test_error'].tolist()

    # Append the dictionaries to their respective lists
    train_error_list.append(train_error_iter_dict)
    valid_error_list.append(valid_error_iter_dict)
    test_error_list.append(test_error_iter_dict)

# %%
# Define colors for training, validation, and test
train_color = 'blue'
valid_color = 'orange'
test_color = 'green'

# Create a figure with 9 subplots arranged in 3 rows and 3 columns
fig, axs = plt.subplots(3, 3, figsize=(15, 20))  # Adjust the size as needed

# Flatten the array of axes for easy iteration
axs = axs.flatten()

arch_lr_info = [
    'arch_foldingnet_lre-4',
    'arch_3x1024_lre-4', 'arch_3x1024_lre-5', 'arch_3x1024_lre-6',  
    'arch_3x512_lre-4', 'arch_3x512_lre-5', 'arch_3x512_lre-6',
    'arch_4x512_lre-4', 'arch_4x512_lre-5', 'arch_4x512_lre-6'
]

# Iterate over each model
for i, arch_lr in enumerate(arch_lr_info):
    # Get the dictionaries for the current model
    train_dict = train_error_list[i]
    valid_dict = valid_error_list[i]
    test_dict = test_error_list[i]

    # Prepare the data for the violin plots
    data_to_plot = []
    positions = []
    
    for j in range(1, 6):  # Assuming there are 5 iterations per dictionary
        key = f'{arch_lr}_iter{j}'
        data_to_plot.extend([(train_dict[key], train_color), 
                            (valid_dict[key], valid_color), 
                            (test_dict[key], test_color)])
        # Define positions for each group of violins
        base_pos = j * 3  # Change this factor as needed for spacing
        positions.extend([base_pos - 0.9, base_pos, base_pos + 0.9])
        
      # Create violin plots for this model
    for data, color, pos in zip(data_to_plot, [train_color, valid_color, test_color]*5, positions):
        vp = axs[i].violinplot(data[0], positions=[pos], vert=False, showmeans=True)
        # Set the color for each part of the violin
        for partname in ('bodies', 'cmins', 'cmaxes', 'cbars', 'cmeans'):
            vp_part = vp[partname]
            if partname == 'bodies':
                for patch in vp_part:
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
            else:
                vp_part.set_edgecolor(color)

    # Formatting the subplot
    axs[i].set_title(arch_lr)
    axs[i].set_xlim(0, 10)  # Set appropriate limits
    axs[i].set_yticks(range(3, 18, 3))
    axs[i].set_yticklabels([f'Iteration {j}' for j in range(1, 6)])

# Create legend patches
legend_patches = [
    Patch(color=train_color, label='Training'),
    Patch(color=valid_color, label='Validation'),
    Patch(color=test_color, label='Test')
]

# Add the legend to the plot
fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3)

# Add an overall title and labels for the axes
fig.suptitle('Violin Plots for Train, Validation, and Test by Model')
plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95])  
fig.text(0.5, 0.04, r'RMSD ($ \AA$)', va='center')
plt.show()

# %%
valid_mean = []
for i in valid_error_list:
    error_means = [np.mean(np.array(arr)) for arr in i.values()]
    valid_errro_mean = np.mean(error_means)
    valid_mean.append(valid_errro_mean)
# %%
sums_and_counts = {key: {'sum': 0, 'count': 0} for key in valid_error_list[0]}

# Sum the lists and count them for each key across all dictionaries
for d in valid_error_list:
    for key, value_list in d.items():
        sums_and_counts[key]['sum'] += np.sum(value_list)
        sums_and_counts[key]['count'] += len(value_list)

# Calculate the mean for each key and create a new list of dictionaries as specified
result_list = [{key: sums_and_counts[key]['sum'] / sums_and_counts[key]['count']} for key in sums_and_counts]

result_list
result_list = [{key: np.mean(values) for key, values in d.items()} for d in valid_error_list]
new_list = []

for d in result_list:
    # Find the key with the lowest value
    lowest_key = min(d, key=d.get)
    # Create a new dictionary with just this key-value pair
    new_dict = {lowest_key: d[lowest_key]}
    # Append the new dictionary to the new list
    new_list.append(new_dict)

# new_list now contains 9 dictionaries, each with one key-value pair (the one with the lowest value)
print(new_list)
# %%
master_dict = {}

for index, row in df_models_unique.iterrows():
    arch_lr = row['arch_lr']
    master_dict[arch_lr] = {
        'train_error': row['train_error'],
        'valid_error': row['valid_error'],
        'test_error': row['test_error']
    }

# %%
# Determine the layout of the subplots
num_rows = 3  # Adjust based on the number of elements in master_dict
num_columns = 3  # Adjust based on the number of elements in master_dict
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))  # Adjust figsize as needed
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Colors for each violin plot
colors = ['blue', 'orange', 'green']

# Loop through the master_dict and create a subplot for each key
for idx, (arch_lr, error_dict) in enumerate(master_dict.items()):
    ax = axes[idx]
    # Create the violin plot for each error type and color them
    vp = ax.violinplot(error_dict['train_error'], positions=[1], showmeans=False, showmedians=True, widths=0.6)
    for pc in vp['bodies']:
        pc.set_facecolor(colors[0])
        pc.set_edgecolor(colors[0])
    
    vp = ax.violinplot(error_dict['valid_error'], positions=[2], showmeans=False, showmedians=True, widths=0.6)
    for pc in vp['bodies']:
        pc.set_facecolor(colors[1])
        pc.set_edgecolor(colors[1])
    
    vp = ax.violinplot(error_dict['test_error'], positions=[3], showmeans=False, showmedians=True, widths=0.6)
    for pc in vp['bodies']:
        pc.set_facecolor(colors[2])
        pc.set_edgecolor(colors[2])

    # Add custom legend
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Train', 'Validation', 'Test'])
    ax.set_title(arch_lr)
    ax.set_ylim(0, 8)

# Create legend patches
legend_patches = [
    Patch(color=train_color, label='Training'),
    Patch(color=valid_color, label='Validation'),
    Patch(color=test_color, label='Test')
]

# Add the legend to the plot
fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3)
fig.suptitle('Violin Plots for Train, Validation, and Test by model and learning rate')
plt.tight_layout(rect=[0.05, 0.05, 0.97, 0.95])  
fig.text(0.04, 0.5, r'RMSD ($ \AA$)', ha='center', va='center', rotation = 'vertical')

plt.show()
# %%

learning_rate_colors = {
    'e-4': 'blue',
    'e-5': 'orange',
    'e-6': 'green'
}
     
# Plotting
for lr, color in learning_rate_colors.items():
    # Filter the DataFrame for each learning rate
    df_filtered = df_models_unique[df_models_unique['learning_rate'] == lr]
    # Plot the filtered DataFrame
    plt.scatter(df_filtered['architecture'], df_filtered['mean_test_error'], color=color, label=f'LR={lr}')
    #plt.plot(df_filtered['architecture'], df_filtered['mean_test_error'], color=color, linestyle='-', marker='o')

plt.title('Average of the mean Test error over five iterations')
plt.xlabel('Architecture')
plt.ylabel('Mean Test Error' r'RMSD ($ \AA$)')
plt.legend()  

plt.show()
# %%
# Load network, test it, and get training, validation, and test errors
def load_foldingnet(): 
    results = []

    for fname in os.listdir('xbb_foldingnet_checkpoints'):

        if fname.startswith('checkpoint') :
            checkpoint_path = os.path.join('xbb_foldingnet_checkpoints', fname)
            checkpoint = torch.load(checkpoint_path, map_location = torch.device('cuda'))

            net = AutoEncoder(**checkpoint['network_kwargs'])
            net.load_state_dict(checkpoint['model_state_dict'])

            MA = MolearnAnalysis()
            MA.set_network(net)
            MA.batch_size = 4
            MA.processes = 2
            
            MA.set_dataset("training", data_train)
            MA.set_dataset("valid", data_valid)
            MA.set_dataset('test', data_test)

            err_train = MA.get_error('training')
            err_valid = MA.get_error('valid')
            err_test = MA.get_error('test')

            results.append({
                'architecture': 'foldingnet',
                'learning_rate': 'e-4',
                'checkpoint_name': fname,
                'train_error': err_train,
                'valid_error': err_valid,
                'test_error': err_test

                })
    return results
# %%
df_models = pd.read_csv('models-benchmark.csv')
#%%
folding_models = []
folding_models.append(load_foldingnet())
# %%
flat_folding = [item for sublist in folding_models for item in sublist]
df_folding = pd.DataFrame(flat_folding)
# %%
df_folding['arch_lr_iter'] = df_folding.apply(lambda x: f"arch_{x['architecture']}_lr{x['learning_rate']}_iter{((x.name)%5)+1}", axis=1)
df_folding['arch_lr'] = df_folding.apply(lambda x: f"arch_{x['architecture']}_lr{x['learning_rate']}", axis=1)

# %%
df_models = pd.concat([df_folding, df_models], ignore_index=True)

# %%
# Define colors for training, validation, and test
train_color = 'blue'
valid_color = 'orange'
test_color = 'green'

# Create a figure with 9 subplots arranged in 3 rows and 3 columns
fig, axs = plt.subplots(3, 3, figsize=(15, 20))  # Adjust the size as needed

# Flatten the array of axes for easy iteration
axs = axs.flatten()

arch_lr_info = [
    'arch_foldingnet_lre-4'
]

# Iterate over each model
for i, arch_lr in enumerate(arch_lr_info):
    # Get the dictionaries for the current model
    train_dict = train_error_list[i]
    valid_dict = valid_error_list[i]
    test_dict = test_error_list[i]

    # Prepare the data for the violin plots
    data_to_plot = []
    positions = []
    
    for j in range(1, 6):  # Assuming there are 5 iterations per dictionary
        key = f'{arch_lr}_iter{j}'
        data_to_plot.extend([(train_dict[key], train_color), 
                            (valid_dict[key], valid_color), 
                            (test_dict[key], test_color)])
        # Define positions for each group of violins
        base_pos = j * 3  # Change this factor as needed for spacing
        positions.extend([base_pos - 0.9, base_pos, base_pos + 0.9])
        
      # Create violin plots for this model
    for data, color, pos in zip(data_to_plot, [train_color, valid_color, test_color]*5, positions):
        vp = axs[i].violinplot(data[0], positions=[pos], vert=False, showmeans=True)
        # Set the color for each part of the violin
        for partname in ('bodies', 'cmins', 'cmaxes', 'cbars', 'cmeans'):
            vp_part = vp[partname]
            if partname == 'bodies':
                for patch in vp_part:
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
            else:
                vp_part.set_edgecolor(color)

    # Formatting the subplot
    axs[i].set_title(arch_lr)
    axs[i].set_xlim(0, 10)  # Set appropriate limits
    axs[i].set_yticks(range(3, 18, 3))
    axs[i].set_yticklabels([f'Iteration {j}' for j in range(1, 6)])

# Create legend patches
legend_patches = [
    Patch(color=train_color, label='Training'),
    Patch(color=valid_color, label='Validation'),
    Patch(color=test_color, label='Test')
]

# Add the legend to the plot
fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=3)

# Add an overall title and labels for the axes
fig.suptitle('Violin Plots for Train, Validation, and Test by Model')
plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95])  
fig.text(0.5, 0.04, r'RMSD ($ \AA$)', va='center')
plt.show()
