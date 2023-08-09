import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import numpy as np


def generateRelativeFeatures(df):
    df['rel_vContextSwitches'] = df['vContextSwitches']/df['duration']
    df['rel_userdiff'] = df.apply(lambda row: row['userDiff'] / 1000 / row['duration'], axis=1)
    df['rel_sysdiff'] = df.apply(lambda row: row['sysDiff'] / 1000 / row['duration'], axis=1)
    df['rel_rss'] = df['rss']/292552704
    df['rel_heapUsed'] = df['heapUsed']/292552704
    df['rel_fsRead'] = df['fsRead']/df['duration']
    df['rel_fsWrite'] = df['fsWrite']/df['duration']
    df['rel_mallocMem'] = df['mallocMem']/292552704
    df['rel_netByRx'] = df['netByRx']/df['duration']
    df['rel_netPkgRx'] = df['netPkgRx']/df['duration']
    df['rel_netByTx'] = df['netByTx']/df['duration']
    df['rel_netPkgTx'] = df['netPkgTx']/df['duration']
    return df


def loadValidationData(data):
    valid_columns = ['id', 'duration', 'maxRss', 'fsRead', 'fsWrite', 'vContextSwitches', 'ivContextSwitches', 
                     'userDiff', 'sysDiff', 'rss', 'heapTotal', 'heapUsed', 'mallocMem', 'netByRx', 'netPkgRx', 'netByTx', 'netPkgTx']
    single_rep = []
    for _, folder in enumerate(os.listdir(f"../data/validation-data-{data}")):
        for _, filename in enumerate(filter(lambda filename: filename.endswith(".csv"), os.listdir(f"../data/validation-data-{data}/{folder}/Repetition_2/"))):
            df = pd.read_csv(f"../data/validation-data-{data}/{folder}/Repetition_2/{filename}")
            df = df.drop(df.columns.difference(valid_columns), 1)
            df = generateRelativeFeatures(df)
            full = df.mean().to_frame().transpose().add_suffix("_mean")
            full = full.merge((df.apply(np.std).to_frame().transpose()/df.mean().to_frame().transpose()*100).add_suffix("_cov"), left_index=True, right_index=True)
            if len(filename.split(".")) > 2:
                full["function"] = filename.split(".")[2]
            else:
                full["function"] = filename.split(".")[0]
            full["f_size"] = int(folder.replace('MB', ''))
            single_rep.append(full)
    single_rep = pd.concat(single_rep)
    return single_rep


def loadValidationDataAllReps(data):
    all_reps = []
    for _, folder in enumerate(os.listdir(f"../data/validation-data-{data}")):
        for _, filename in enumerate(filter(lambda filename: filename.endswith(".csv"), os.listdir(f"../data/validation-data-{data}/{folder}/Repetition_2/"))):
            durations = []
            for i in range(0, 3):
                df = pd.read_csv(f"../data/validation-data-{data}/{folder}/Repetition_" + str(i) + f"/{filename}")
                duration = df['duration'].mean()
                durations.append(duration)
            full = pd.DataFrame(columns=['duration', 'function', 'f_size'])
            if len(filename.split(".")) > 2:
                nam = filename.split(".")[2]
            else:
                nam = filename.split(".")[0]
            dicc = {'duration': durations, 'function': nam, 'f_size': int(folder.replace('MB', ''))}
            full = full.append(dicc, ignore_index=True)
            all_reps.append(full)
    all_reps = pd.concat(all_reps)
    all_reps = all_reps.sort_values(by=['f_size', 'function'])
    all_reps = all_reps.reset_index(drop=True)
    return all_reps


def loadTrainingData():
    if os.path.isfile('../results/processed_training_data_custom.csv'):
        df =  pd.read_csv(f"../results/processed_training_data_custom.csv")
    else:
        df = pd.read_csv(f"../data/processed_training_data.csv")
    # Drop malfunctioned measurements
    excludes = ['generated-560-white-frog', 'generated-302-cool-feather', 'json2yaml-decompress-readfile', 'generated-4-weathered-wind', 'decompress-json2yaml-json2yaml',
                'generated-234-old-sun', 'generated-476-young-silence', 'generated-389-dry-bush', 'generated-441-shy-wave', 'generated-712-white-glade',                
                 'readfile-json2yaml-json2yaml', 'generated-235-silent-moon', 'decompress-json2yaml-readfile', 'generated-82-weathered-tree', 'generated-795-red-snow', 
                 'generated-59-blue-moon', 'generated-875-polished-shadow', 'json2yaml-readfile-decompress', 'generated-781-lively-wind', 
                 'json2yaml-floatoperations-floatoperations', 'generated-820-hidden-water', 'floatoperations-readfile-floatoperations', 'generated-527-restless-frost',
                 'generated-696-frosty-pine', 'generated-699-bitter-cherry', 'generated-174-polished-snowflake', 'generated-786-misty-cloud', 'generated-709-empty-paper']
    df = df[~df.f_name.isin(excludes)]
    return df

        
if __name__ == '__main__':
    from glob import glob
    single_rep = []
    i = 0
    file_list = filter(lambda filename: filename.endswith(".csv"), os.listdir(f"../data/training-data/"))
    # file_list = glob("../data/validation-data/**/*.csv", recursive=True)
    for _, filename in enumerate(file_list):
        i = i + 1
        print(i, "/", 240)
        df = pd.read_csv(f"../data/training-data/{filename}")
        # df = pd.read_csv(filename)[:80]
        df = generateRelativeFeatures(df)
        full = df.mean().to_frame().transpose().add_suffix("_mean")
        full = full.merge((df.apply(np.std).to_frame().transpose()/df.mean().to_frame().transpose()*100).add_suffix("_cov"), left_index=True, right_index=True)
        splits = filename.rsplit('-', 1)
        full["f_size"] = int(splits[1].split(".")[0])
        full['f_name'] = splits[0]
        # splits = filename.split('/')
        # full["f_size"] = int(splits[3][:-2])
        # full['f_name'] = splits[5][:-4]
        # print('>>>', full["f_name"], full["f_size"])
        single_rep.append(full)
    single_rep = pd.concat(single_rep)

    single_rep.to_csv("../data/processed_training_data.csv")
    single_rep = pd.read_csv("../data/processed_training_data.csv")
    for index, row in single_rep.iterrows():
        # print(index)
        # print(single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 128)]['duration_mean'].iloc[0])
        single_rep.at[index, 'y_128'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 128)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_256'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 256)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_512'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 512)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_1024'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 1024)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_2048'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 2048)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_3008'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 3008)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_4096'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 4096)]['duration_mean'].iloc[0]
        single_rep.at[index, 'y_16384'] = single_rep[(single_rep['f_name'] == row['f_name']) & (single_rep['f_size'] == 16384)]['duration_mean'].iloc[0]

    single_rep.to_csv("../data/processed_training_data.csv")
