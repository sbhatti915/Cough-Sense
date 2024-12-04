# Creates datasets using TB and Covid data to make dataset of viral, bacterial, and neither
import pandas as pd
import os
import shutil

cough_score_threshold = 0.8

covid_data_folder_path = '/home/sameer/Cough-Sense/coughvid_data'
forced_cough_folder_path = '/media/sameer/Sameer Mem/TBscreen_Dataset/Forced_coughs/Audio_files'
passive_cough_folder_path = '/media/sameer/Sameer Mem/TBscreen_Dataset/Passive_coughs/Audio_files'

coughvid_metadata_path = '/home/sameer/Cough-Sense/coughvid_data/metadata_compiled.csv'
tb_screen_forced_path = '/media/sameer/Sameer Mem/TBscreen_Dataset/Forced_coughs/Forced_coughs.csv'
tb_screen_passive_path = '/media/sameer/Sameer Mem/TBscreen_Dataset/Passive_coughs/Passive_coughs.csv'

# Define the folder where all the data should go
destination_folder = '/home/sameer/Cough-Sense/data'

df_coughvid = pd.read_csv(coughvid_metadata_path)
df_forced = pd.read_csv(tb_screen_forced_path)
df_passive = pd.read_csv(tb_screen_passive_path)

# Get covid uuids into a dataframe
filtered_coughvid_df = df_coughvid.loc[df_coughvid['cough_detected'] > cough_score_threshold]
covid_uuids_df = filtered_coughvid_df.loc[filtered_coughvid_df['status'] == 'COVID-19']['uuid']
healthy_uuids_df = filtered_coughvid_df.loc[filtered_coughvid_df['status'] == 'healthy'].sample(n=700, random_state=42)['uuid']

# Check if files in tb dataframes exist in folder because for some reason they don't. Create a filtered dataframe with only files we have.
forced_file_list = os.listdir(forced_cough_folder_path)
passive_file_list = os.listdir(passive_cough_folder_path)

forced_file_list_no_extension = [os.path.splitext(file)[0] for file in forced_file_list]
passive_file_list_no_extension = [os.path.splitext(file)[0] for file in passive_file_list]

df_forced = df_forced[df_forced['path'].isin(forced_file_list_no_extension)]
df_passive = df_passive[df_passive['path'].isin(passive_file_list_no_extension)]

# Get TB paths into a dataframe
tb_forced_pixel_df = df_forced[(df_forced['Label'] == 'TB') & (df_forced['device'] == 'pixel')].sample(n=117, random_state=42)['path']
tb_forced_codec_df = df_forced[(df_forced['Label'] == 'TB') & (df_forced['device'] == 'codec')].sample(n=117, random_state=42)['path']
tb_forced_yeti_df = df_forced[(df_forced['Label'] == 'TB') & (df_forced['device'] == 'yeti')].sample(n=117, random_state=42)['path']

tb_passive_pixel_df = df_passive[(df_passive['Label'] == 'TB') & (df_passive['device'] == 'pixel')].sample(n=117, random_state=42)['path']
tb_passive_codec_df = df_passive[(df_passive['Label'] == 'TB') & (df_passive['device'] == 'codec')].sample(n=117, random_state=42)['path']
tb_passive_yeti_df = df_passive[(df_passive['Label'] == 'TB') & (df_passive['device'] == 'yeti')].sample(n=117, random_state=42)['path']

# Create viral dataframe
labels = ['viral'] * len(covid_uuids_df)
labels = pd.DataFrame(labels)
covid_uuids_df.reset_index(drop=True, inplace=True)

viral_df = pd.concat([covid_uuids_df, labels], axis=1, ignore_index=False)
viral_df.columns = ['file_name', 'Label']

# Create bacterial dataframe
labels = ['bacterial'] * (len(tb_forced_pixel_df) + len(tb_forced_codec_df) + len(tb_forced_yeti_df))
labels = pd.DataFrame(labels)

tb_forced_pixel_df.reset_index(drop=True, inplace=True)
tb_forced_codec_df.reset_index(drop=True, inplace=True)
tb_forced_yeti_df.reset_index(drop=True, inplace=True)
tb_passive_pixel_df.reset_index(drop=True, inplace=True)
tb_passive_codec_df.reset_index(drop=True, inplace=True)
tb_passive_yeti_df.reset_index(drop=True, inplace=True)

bacterial_total_forced_df = pd.concat([tb_forced_pixel_df, tb_forced_codec_df, tb_forced_yeti_df], ignore_index=True)
bacterial_total_passive_df = pd.concat([tb_passive_pixel_df, tb_passive_codec_df, tb_passive_yeti_df], ignore_index=True)

bacterial_forced_df = pd.concat([bacterial_total_forced_df, labels], axis=1, ignore_index=False)
bacterial_forced_df.columns = ['file_name', 'Label']

bacterial_passive_df = pd.concat([bacterial_total_passive_df, labels], axis=1, ignore_index=False)
bacterial_passive_df.columns = ['file_name', 'Label']

# Create neither dataframe
labels = ['neither'] * len(healthy_uuids_df)
labels = pd.DataFrame(labels)

healthy_uuids_df.reset_index(drop=True, inplace=True)

neither_df = pd.concat([healthy_uuids_df, labels], axis=1, ignore_index=False)

neither_df.columns = ['file_name', 'Label']

# Move files to data folder
def copy_files(src_folder, dest_folder, df):
    file_list = df['file_name'].tolist()
    for file in file_list:
        dest_file_path = f"{os.path.join(dest_folder, file)}.wav"
        if os.path.exists(dest_file_path):
            continue
        else:
            src_file_path = f"{os.path.join(src_folder, file)}.wav"
            if os.path.exists(src_file_path):
                shutil.copy(src_file_path, dest_file_path)
            else:
                print('Source file ' + src_file_path + " not found")
                continue
    print("Copied Files")


## Move all files to data folder
# move_files(covid_data_folder_path, destination_folder, viral_df)
# move_files(forced_cough_folder_path, destination_folder, bacterial_forced_df)
# move_files(passive_cough_folder_path, destination_folder, bacterial_passive_df)
# move_files(covid_data_folder_path, destination_folder, neither_df)

# Move all files back just in case I messed up
# move_files(destination_folder, covid_data_folder_path, viral_df)
# move_files(destination_folder, forced_cough_folder_path, bacterial_forced_df)
# move_files(destination_folder, passive_cough_folder_path, bacterial_passive_df)
# move_files(destination_folder, covid_data_folder_path, neither_df)

# Copy all files to data folder
copy_files(covid_data_folder_path, destination_folder, viral_df)
copy_files(forced_cough_folder_path, destination_folder, bacterial_forced_df)
copy_files(passive_cough_folder_path, destination_folder, bacterial_passive_df)
copy_files(covid_data_folder_path, destination_folder, neither_df)

# Concatenate dataframes to make one labels csv file
dataset_labels = pd.concat([viral_df, bacterial_forced_df, bacterial_passive_df, neither_df], ignore_index=True)


dataset_labels.to_csv(f'{destination_folder}/dataset_labels.csv', index=False)

print(f'Saved labels to {destination_folder}/dataset_labels.csv')