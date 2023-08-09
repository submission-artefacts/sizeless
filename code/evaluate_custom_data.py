import math
import pickle
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
import numpy as np
import Helper
import custom_data_loader as dl
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import sys
import warnings

warnings.filterwarnings("ignore")

data = sys.argv[1]

# duration is in  millisecs
# userDiff and sysDiff are in microsecon
stringMemory = ["128", "256", "512", "1024", "2048", "3008", "4096", "16384"]
memory_sizes = [128, 256, 512, 1024, 2048, 3008, 4096, 16384]

used_features = [
    'userDiff_mean', 'sysDiff_mean', 'rel_vContextSwitches_mean', 'rel_userdiff_cov',
    'rel_userdiff_mean', 'rel_sysdiff_mean', 'rel_fsWrite_mean', 'rel_netByRx_mean',
    'heapUsed_mean', 'heapUsed_cov', 'mallocMem_cov']

# used_features = [
#     'userDiff_mean', 'sysDiff_mean', 'rel_ivContextSwitches_mean', 'rel_vContextSwitches_mean',
#     'rel_userdiff_cov', 'rel_sysdiff_cov', 'rel_userdiff_mean', 'rel_sysdiff_mean',
#     'rel_fsWrite_mean', 'rel_fsRead_mean', 'rel_netByRx_mean', 'rel_netByTx_mean',
#     # 'rss_mean', 'rss_cov', 'mallocMem_cov'
# ]

# Load data for single repetition
single_rep = dl.loadValidationData(data)
print(single_rep.shape)
single_rep = single_rep.sort_values(by=['f_size', 'function'])
single_rep = single_rep.reset_index(drop=True)

with open('../models/models.pickle', 'rb') as handle:
    models = pickle.load(handle)

with open('../models/scs.pickle', 'rb') as handle:
    scs = pickle.load(handle)

with open('../models/scs2.pickle', 'rb') as handle:
    scs2 = pickle.load(handle)

# Generate predictions
prediction = pd.DataFrame(columns=stringMemory)
for memory_size in memory_sizes:
    X = single_rep.copy()
    X = X[X['f_size'] == memory_size]
    
    X = X.drop(X.columns.difference(used_features), 1)

    # Minmax scaler
    sc2 = scs2[memory_size]
    X = sc2.transform(X)
    sc = scs[memory_size]
    X = sc.transform(X)

    # Generate predictions
    model = models[memory_size]
    pred = model.predict(X)
    cols = stringMemory.copy()
    cols.remove(str(memory_size))
    pred = pd.DataFrame(data=pred, columns=cols)
    prediction = prediction.append(pred)
prediction = prediction.reset_index(drop=True)

# Load data for all repetitions
all_reps = dl.loadValidationDataAllReps(data)

# Unratio results
all_reps['128'] = prediction['128'] * single_rep['duration_mean']
all_reps['256'] = prediction['256'] * single_rep['duration_mean']
all_reps['512'] = prediction['512'] * single_rep['duration_mean']
all_reps['1024'] = prediction['1024'] * single_rep['duration_mean']
all_reps['2048'] = prediction['2048'] * single_rep['duration_mean']
all_reps['3008'] = prediction['3008'] * single_rep['duration_mean']
all_reps['4096'] = prediction['4096'] * single_rep['duration_mean']
all_reps['16384'] = prediction['16384'] * single_rep['duration_mean']
all_reps['duration_mean_single'] = single_rep['duration_mean']
all_reps['duration_mean'] = all_reps['duration'].apply(np.mean)

# Aggregate and plot results
if 'fb' in data:
    function_names = ["chameleon", "cnn_image_classification", "driver", "feature_extractor", "feature_reducer", "float_operation", 
                      "image_processing", "linpack", "mapper", "matmul", "ml_lr_prediction", "ml_video_face_detection", 
                      "model_training", "orchestrator", "pyaes", "reducer", "rnn_generate_character_level", "video_processing"]

    function_names_real = ["chameleon", "cnn_image_classification", "driver", "feature_extractor", "feature_reducer", "float_operation", 
                            "image_processing", "linpack", "mapper", "matmul", "ml_lr_prediction", "ml_video_face_detection", 
                            "model_training", "orchestrator", "pyaes", "reducer", "rnn_generate_character_level", "video_processing"]
elif 'sb' in data:
    function_names = ["alloc_res", "wage-db-writer", "wage-format", "wage-insert"]

    function_names_real = ["alloc_res", "wage-db-writer", "wage-format", "wage-insert"]

else:
    function_names = ["chameleon", "cnn_image_classification", "driver", "feature_extractor", "feature_reducer", "float_operation", 
                        "image_processing", "linpack", "mapper", "matmul", "ml_lr_prediction", "ml_video_face_detection", 
                        "model_training", "orchestrator", "pyaes", "reducer", "rnn_generate_character_level", "video_processing",
                        "alloc_res", "wage-db-writer", "wage-format", "wage-insert"]

    function_names_real = ["chameleon", "cnn_image_classification", "driver", "feature_extractor", "feature_reducer", "float_operation", 
                            "image_processing", "linpack", "mapper", "matmul", "ml_lr_prediction", "ml_video_face_detection", 
                            "model_training", "orchestrator", "pyaes", "reducer", "rnn_generate_character_level", "video_processing",
                            "alloc_res", "wage-db-writer", "wage-format", "wage-insert"]

cs = stringMemory.copy()
cs = cs.remove("256")
acc_table = pd.DataFrame(index=function_names_real, columns=cs)
all_accs = []
all_abs_accs = []
all_sq_errs = []
# Iterate over all functions
for idx, function_name in enumerate(function_names):
    all_reps_single_fun = all_reps[all_reps['function'] == function_name]
    all_reps_single_fun = all_reps_single_fun.sort_values('f_size')

    # Plot ground truth
    fig, ax1 = plt.subplots(figsize=[12, 8])
    ax1.errorbar(all_reps_single_fun['f_size'].apply(str), all_reps_single_fun['duration'].apply(np.mean), yerr=all_reps_single_fun['duration'].apply(np.std), capsize=4, label="Measured")
    
    # Iterate over base size
    for idx2, base in enumerate([128, 256, 512, 1024, 2048, 3008, 4096, 16384]):
        # Plot all predictions for base size
        all_reps_single_fun_single_base = all_reps_single_fun[all_reps_single_fun['f_size'] == base]
        x3 = all_reps_single_fun['f_size'].apply(str).tolist()
        y3 = [all_reps_single_fun_single_base['128'], all_reps_single_fun_single_base['256'], all_reps_single_fun_single_base['512'], 
                all_reps_single_fun_single_base['1024'], all_reps_single_fun_single_base['2048'], all_reps_single_fun_single_base['3008'],
              all_reps_single_fun_single_base['4096'], all_reps_single_fun_single_base['16384']]
        try:
            del y3[idx2]
            del x3[idx2]
        except Exception as ex:
            print(ex)
            print(idx2)
        ax1.scatter(x3, y3, c="C" + str(idx2), s=100, linewidth=1, label=str(base) + "MB", marker="x")
        
        # Generate overview
        if base == 256:
            accs = []
            for target in [128, 512, 1024, 2048, 3008, 4096, 16384]:
                pred = all_reps_single_fun_single_base[str(target)].reset_index(drop=True)[0]
                real = np.mean(all_reps_single_fun[all_reps_single_fun['f_size'] == target]['duration'].reset_index(drop=True)[0])
                print(function_names_real[idx], f"{target}MB", "real:", real, "pred:", pred)
                all_accs.append(mean_absolute_percentage_error([real], [pred])*100)
                all_abs_accs.append(mean_absolute_error([real], [pred]))
                acc_table.at[function_names_real[idx], str(target)] = mean_absolute_percentage_error([real], [pred])*100
                all_sq_errs.append((real - pred)**2)
    a = ['capture-stripe-metrics', 'charge-stripe-metrics', 'collect-payment-metrics']
    if function_name in a:
        plt.legend(loc='lower right', ncol=2)
    else:
        plt.legend(loc='upper right', ncol=2)
    ax1.set_ylabel("Execution time [ms]")
    ax1.set_xticks(stringMemory)
    ax1.set_xlabel("Memory size [MB]", labelpad=12)
    ax1.set_ylim(0, ax1.get_ylim()[1])
    plt.title(function_names_real[idx], fontweight="bold")
    plt.tight_layout()
    plt.savefig("../results/" + function_name + ".pdf")
    plt.close()

airline_acc_table = acc_table[0:8]
airline_acc_table = airline_acc_table.copy()
airline_acc_table.at['All functions', "128"] = airline_acc_table['128'].mean()
airline_acc_table.at['All functions', "512"] = airline_acc_table['512'].mean()
airline_acc_table.at['All functions', "1024"] = airline_acc_table['1024'].mean()
airline_acc_table.at['All functions', "2048"] = airline_acc_table['2048'].mean()
airline_acc_table.at['All functions', "3008"] = airline_acc_table['3008'].mean()
airline_acc_table.at['All functions', "4096"] = airline_acc_table['4096'].mean()
airline_acc_table.at['All functions', "16384"] = airline_acc_table['16384'].mean()
airline_acc_table.to_csv("../results/validation_table_airline.csv")

face_acc_table = acc_table[8:13]
face_acc_table = face_acc_table.copy()
face_acc_table.at['All functions', "128"] = face_acc_table['128'].mean()
face_acc_table.at['All functions', "512"] = face_acc_table['512'].mean()
face_acc_table.at['All functions', "1024"] = face_acc_table['1024'].mean()
face_acc_table.at['All functions', "2048"] = face_acc_table['2048'].mean()
face_acc_table.at['All functions', "3008"] = face_acc_table['3008'].mean()
face_acc_table.at['All functions', "4096"] = face_acc_table['4096'].mean()
face_acc_table.at['All functions', "16384"] = face_acc_table['16384'].mean()
face_acc_table.to_csv("../results/validation_table_face.csv")

event_acc_table = acc_table[13:20]
event_acc_table = event_acc_table.copy()
event_acc_table.at['All functions', "128"] = event_acc_table['128'].mean()
event_acc_table.at['All functions', "512"] = event_acc_table['512'].mean()
event_acc_table.at['All functions', "1024"] = event_acc_table['1024'].mean()
event_acc_table.at['All functions', "2048"] = event_acc_table['2048'].mean()
event_acc_table.at['All functions', "3008"] = event_acc_table['3008'].mean()
event_acc_table.at['All functions', "4096"] = event_acc_table['4096'].mean()
event_acc_table.at['All functions', "16384"] = event_acc_table['16384'].mean()
event_acc_table.to_csv("../results/validation_table_event.csv")

retail_acc_table = acc_table[20:]
retail_acc_table = retail_acc_table.copy()
retail_acc_table.at['All functions', "128"] = retail_acc_table['128'].mean()
retail_acc_table.at['All functions', "512"] = retail_acc_table['512'].mean()
retail_acc_table.at['All functions', "1024"] = retail_acc_table['1024'].mean()
retail_acc_table.at['All functions', "2048"] = retail_acc_table['2048'].mean()
retail_acc_table.at['All functions', "3008"] = retail_acc_table['3008'].mean()
retail_acc_table.at['All functions', "4096"] = retail_acc_table['4096'].mean()
retail_acc_table.at['All functions', "16384"] = retail_acc_table['16384'].mean()
retail_acc_table.to_csv("../results/validation_table_retail.csv")

# Preprocess real execution time
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
dataset = all_reps
dataset['real_128'] = np.nan
dataset['real_256'] = np.nan
dataset['real_512'] = np.nan
dataset['real_1024'] = np.nan
dataset['real_2048'] = np.nan
dataset['real_3008'] = np.nan
dataset['real_4096'] = np.nan
dataset['real_16384'] = np.nan


for index, row in dataset.iterrows():
    for other in [128, 256, 512, 1024, 2048, 3008, 4096, 16384]:
        sub = dataset[dataset['f_size'] == other]
        sub = sub[sub['function'] == row['function']]
        sub = sub.reset_index(drop=True)
        dataset.at[index, 'real_' + str(other)] = sub['duration_mean'][0]
            

# Calculate predicted costsavings and speedups
for tradeoff in [0.25, 0.5, 0.75]:
# for tradeoff in [0, 0.5, 1]:
    size = 256
    subsetbase = dataset[dataset['f_size'] == size]
    subsetbase = subsetbase.copy()
    subsetbase['cost_128'] = (subsetbase['128']*128/1024*0.00001667 + 0.0000002)
    subsetbase['cost_256'] = (subsetbase['256']*256/1024*0.00001667 + 0.0000002)
    subsetbase['cost_512'] = (subsetbase['512']*512/1024*0.00001667 + 0.0000002)
    subsetbase['cost_1024'] = (subsetbase['1024']*1024/1024*0.00001667 + 0.0000002)
    subsetbase['cost_2048'] = (subsetbase['2048']*2048/1024*0.00001667 + 0.0000002)
    subsetbase['cost_3008'] = (subsetbase['3008']*3008/1024*0.00001667 + 0.0000002)
    subsetbase['cost_4096'] = (subsetbase['4096'] * 4096 / 1024 * 0.00001667 + 0.0000002)
    subsetbase['cost_16384'] = (subsetbase['16384'] * 16384 / 1024 * 0.00001667 + 0.0000002)

    subsetbase['real_cost_128'] = (subsetbase['real_128']*128/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_256'] = (subsetbase['real_256']*256/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_512'] = (subsetbase['real_512']*512/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_1024'] = (subsetbase['real_1024']*1024/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_2048'] = (subsetbase['real_2048']*2048/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_3008'] = (subsetbase['real_3008']*3008/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_4096'] = (subsetbase['real_4096']*4096/1024*0.00001667 + 0.0000002)
    subsetbase['real_cost_16384'] = (subsetbase['real_16384']*16384/1024*0.00001667 + 0.0000002)
    # perf*cost
    subsetbase['perfcost_128'] = (subsetbase['128']**2*128/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_256'] = (subsetbase['256']**2*256/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_512'] = (subsetbase['512']**2*512/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_1024'] = (subsetbase['1024']**2*1024/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_2048'] = (subsetbase['2048']**2*2048/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_3008'] = (subsetbase['3008']**2*3008/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_4096'] = (subsetbase['4096']**2*4096/1024*0.00001667 + 0.0000002)
    subsetbase['perfcost_16384'] = (subsetbase['16384']**2*16384/1024*0.00001667 + 0.0000002)

    subsetbase['real_perfcost_128'] = (subsetbase['real_128']**2*128/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_256'] = (subsetbase['real_256']**2*256/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_512'] = (subsetbase['real_512']**2*512/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_1024'] = (subsetbase['real_1024']**2*1024/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_2048'] = (subsetbase['real_2048']**2*2048/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_3008'] = (subsetbase['real_3008']**2*3008/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_4096'] = (subsetbase['real_4096']**2*4096/1024*0.00001667 + 0.0000002)
    subsetbase['real_perfcost_16384'] = (subsetbase['real_16384']**2*16384/1024*0.00001667 + 0.0000002)
    
    subsetbase['perf_128'] = subsetbase['128']
    subsetbase['perf_256'] = subsetbase['256']
    subsetbase['perf_512'] = subsetbase['512']
    subsetbase['perf_1024'] = subsetbase['1024']
    subsetbase['perf_2048'] = subsetbase['2048']
    subsetbase['perf_3008'] = subsetbase['3008']
    subsetbase['perf_4096'] = subsetbase['4096']
    subsetbase['perf_16384'] = subsetbase['16384']

    subsetbase['real_perf_128'] = subsetbase['real_128']
    subsetbase['real_perf_256'] = subsetbase['real_256']
    subsetbase['real_perf_512'] = subsetbase['real_512']
    subsetbase['real_perf_1024'] = subsetbase['real_1024']
    subsetbase['real_perf_2048'] = subsetbase['real_2048']
    subsetbase['real_perf_3008'] = subsetbase['real_3008']
    subsetbase['real_perf_4096'] = subsetbase['real_4096']
    subsetbase['real_perf_16384'] = subsetbase['real_16384']
    # Calculate min cost/perf
    subsetbase['min_cost'] = subsetbase[['cost_128','cost_256', 'cost_512', 'cost_1024', 'cost_2048', 'cost_3008', 'cost_4096', 'cost_16384']].min(axis=1)
    subsetbase['real_min_cost'] = subsetbase[['real_cost_128','real_cost_256', 'real_cost_512', 'real_cost_1024', 'real_cost_2048', 'real_cost_3008', 'real_cost_4096', 'real_cost_16384']].min(axis=1)
    subsetbase['min_perf'] = subsetbase[['perf_128','perf_256', 'perf_512', 'perf_1024', 'perf_2048', 'perf_3008', 'perf_4096', 'perf_16384']].min(axis=1)
    subsetbase['real_min_perf'] = subsetbase[['real_perf_128','real_perf_256', 'real_perf_512', 'real_perf_1024', 'real_perf_2048', 'real_perf_3008', 'real_perf_4096', 'real_perf_16384']].min(axis=1)

    for memory_size in [128, 512, 1024, 2048, 3008, 4096, 16384]:
        subsetbase['score_' + str(memory_size)] = tradeoff * (subsetbase['cost_' + str(memory_size)] / subsetbase['min_cost']) + (1 - tradeoff) * (subsetbase['perf_' + str(memory_size)] / subsetbase['min_perf'])
        subsetbase['real_score_' + str(memory_size)] = tradeoff * (subsetbase['real_cost_' + str(memory_size)] / subsetbase['real_min_cost']) + (1 - tradeoff) * (subsetbase['real_perf_' + str(memory_size)] / subsetbase['real_min_perf'])
    subsetbase['real_score_256'] = tradeoff * (subsetbase['real_cost_256'] / subsetbase['real_min_cost']) + (1 - tradeoff) * (subsetbase['real_perf_256'] / subsetbase['real_min_perf'])
    subsetbase['score_256'] = subsetbase['real_score_256'] # We measured 256MB and do not need to predict it

    selections = []
    costdecreases = []
    speedups = []
    speedcostimpvs = []

    perf_scores = []
    cost_scores = []
    perfcost_scores = []

    for index, row in subsetbase.iterrows():
        real_scores_by_mem = [row['real_score_128'], row['real_score_256'], row['real_score_512'], row['real_score_1024'], row['real_score_2048'], row['real_score_3008'], row['real_score_4096'], row['real_score_16384']]
        real_scores_by_score = real_scores_by_mem.copy()
        real_scores_by_score.sort()
        scores_by_mem = [row['score_128'], row['score_256'], row['score_512'], row['score_1024'], row['score_2048'], row['score_3008'], row['score_4096'], row['score_16384']]
        scores_by_score = scores_by_mem.copy()
        scores_by_score.sort()
        selected_score = scores_by_score[0]
        real_selected_score = real_scores_by_mem[scores_by_mem.index(selected_score)]
        selected_n_best = real_scores_by_score.index(real_selected_score) + 1
        selections.append(selected_n_best)
        selected_memory_size = scores_by_mem.index(selected_score)
        if selected_memory_size == 0:
            selected_memory_size = 128
        if selected_memory_size == 1:
            selected_memory_size = 256
        if selected_memory_size == 2:
            selected_memory_size = 512
        if selected_memory_size == 3:
            selected_memory_size = 1024
        if selected_memory_size == 4:
            selected_memory_size = 2048
        if selected_memory_size == 5:
            selected_memory_size = 3008
        if selected_memory_size == 6:
            selected_memory_size = 4096
        if selected_memory_size == 7:
            selected_memory_size = 16384
        speedup = 1 - row['real_perf_' + str(selected_memory_size)]/row['real_perf_256']
        speedups.append(speedup)
        costdecrease = 1 - row['real_cost_' + str(selected_memory_size)]/row['real_cost_256']
        costdecreases.append(costdecrease)
        speedcostimpv = 1 - (row['real_perf_' + str(selected_memory_size)] * row['real_cost_' + str(selected_memory_size)]) / (row['real_perf_256'] * row['real_cost_256'])
        speedcostimpvs.append(speedcostimpv)

        optimum_score = real_scores_by_score[0]
        optimum_memory_size = real_scores_by_mem.index(optimum_score)
        

        if optimum_memory_size == 0:
            optimum_memory_size = 128
        if optimum_memory_size == 1:
            optimum_memory_size = 256
        if optimum_memory_size == 2:
            optimum_memory_size = 512
        if optimum_memory_size == 3:
            optimum_memory_size = 1024
        if optimum_memory_size == 4:
            optimum_memory_size = 2048
        if optimum_memory_size == 5:
            optimum_memory_size = 3008
        if optimum_memory_size == 6:
            optimum_memory_size = 4096
        if optimum_memory_size == 7:
            optimum_memory_size = 16384

        print(row['function'], "Predicted conf:", selected_memory_size, "Optimum conf:", optimum_memory_size)

        pred_conf_perf = row['real_perf_' + str(selected_memory_size)]
        optim_conf_perf = row['real_perf_' + str(optimum_memory_size)]
        # print("Perf on predicted conf:", pred_conf_perf)
        # print("Perf on optimum conf:", optim_conf_perf)

        pred_conf_cost = row['real_cost_' + str(selected_memory_size)]
        optim_conf_cost = row['real_cost_' + str(optimum_memory_size)]
        # print("Cost on predicted conf:", pred_conf_cost)
        # print("Cost on optimum conf:", optim_conf_cost)

        pred_conf_perfcost = row['real_perfcost_' + str(selected_memory_size)]
        optim_conf_perfcost = row['real_perfcost_' + str(optimum_memory_size)]
        # print("Cost on predicted conf:", pred_conf_perfcost)
        # print("Cost on optimum conf:", optim_conf_perfcost)


        perf_score = (pred_conf_perf-optim_conf_perf)/((pred_conf_perf+optim_conf_perf)/2)
        perf_scores.append(perf_score)
        cost_score = (pred_conf_cost-optim_conf_cost)/((pred_conf_cost+optim_conf_cost)/2)
        cost_scores.append(cost_score)
        perfcost_score = (pred_conf_perfcost-optim_conf_perfcost)/((pred_conf_perfcost+optim_conf_perfcost)/2)
        perfcost_scores.append(perfcost_score)
        

    fig, ax = plt.subplots(figsize=[12, 7])
    labels = ['Best', '2nd\nbest', '3rd\nbest', '4th\nbest', '5th\nbest', '6th\nbest']
    values = [selections[0:8].count(1), selections[0:8].count(2), selections[0:8].count(3), selections[0:8].count(4), selections[0:8].count(5), selections[0:8].count(6)]
    values2 = [selections[8:13].count(1), selections[8:13].count(2), selections[8:13].count(3), selections[8:13].count(4), selections[8:13].count(5), selections[8:13].count(6)]
    values3 = [selections[13:20].count(1), selections[13:20].count(2), selections[13:20].count(3), selections[13:20].count(4), selections[13:20].count(5), selections[13:20].count(6)]
    values4 = [selections[20:].count(1), selections[20:].count(2), selections[20:].count(3), selections[20:].count(4), selections[20:].count(5), selections[20:].count(6)]
    print(selections.count(1), selections.count(2), selections.count(3), selections.count(4), selections.count(5), selections.count(6))
    p1 = ax.bar(labels, values)
    p2 = ax.bar(labels, values2, bottom=values)
    p3 = ax.bar(labels, values3, bottom=np.array(values)+np.array(values2))
    p4 = ax.bar(labels, values4, bottom=np.array(values)+np.array(values2)+np.array(values3))
    ax.set_xlabel('Selected memory size')
    ax.set_ylabel('Number of functions')
    plt.ylim((0, 25))
    plt.title("t = " + str(tradeoff))
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Airline Booking', 'Facial Recognition', 'Event Processing', 'Hello Retail'))
    plt.tight_layout()
    plt.savefig("../results/selected_memory_sizes_" + str(tradeoff) + ".pdf") 

    print("t = " + str(tradeoff))
    # print("COST DECREASE - Airline", np.mean(costdecreases[0:8])*100)
    # print("SPEEDUP - Airline", np.mean(speedups[0:8])*100)
    # print("COST DECREASE - Face", np.mean(costdecreases[8:13])*100)
    # print("SPEEDUP - Face", np.mean(speedups[8:13])*100)
    # print("COST DECREASE - Event", np.mean(costdecreases[13:20])*100)
    # print("SPEEDUP - Event", np.mean(speedups[13:20])*100)
    # print("COST DECREASE - Retail", np.mean(costdecreases[20:])*100)
    # print("SPEEDUP - Retail", np.mean(speedups[20:])*100)
    print("COST DECREASE - TOTAL", np.mean(costdecreases)*100)
    print("SPEEDUP - TOTAL", np.mean(speedups)*100)
    print("COST*PERF Improvement - TOTAL", np.mean(speedcostimpvs)*100)

    # print("Probability of predicting within 15% from optimum COST - Airline", (np.array(cost_scores)[0:8]<.15).sum()/len(cost_scores[0:8])*100)
    # print("Probability of predicting within 15% from optimum PERF - Airline", (np.array(perf_scores)[0:8]<.15).sum()/len(perf_scores[0:8])*100)
    # print("Probability of predicting within 15% from optimum PERF*COST - Airline", (np.array(perfcost_scores)[0:8]<.15).sum()/len(perfcost_scores[0:8])*100)
    #
    # print("Probability of predicting within 15% from optimum COST - Face", (np.array(cost_scores)[8:13]<.15).sum()/len(cost_scores[8:13])*100)
    # print("Probability of predicting within 15% from optimum PERF - Face", (np.array(perf_scores)[8:13]<.15).sum()/len(perf_scores[8:13])*100)
    # print("Probability of predicting within 15% from optimum PERF*COST - Face", (np.array(perfcost_scores)[8:13]<.15).sum()/len(perfcost_scores[8:13])*100)
    #
    # print("Probability of predicting within 15% from optimum COST - Event", (np.array(cost_scores)[13:20]<.15).sum()/len(cost_scores[13:20])*100)
    # print("Probability of predicting within 15% from optimum PERF - Event", (np.array(perf_scores)[13:20]<.15).sum()/len(perf_scores[13:20])*100)
    # print("Probability of predicting within 15% from optimum PERF*COST - Event", (np.array(perfcost_scores)[13:20]<.15).sum()/len(perfcost_scores[13:20])*100)
    #
    # print("Probability of predicting within 15% from optimum COST - Retail", (np.array(cost_scores[20:])<.15).sum()/len(cost_scores[20:])*100)
    # print("Probability of predicting within 15% from optimum PERF - Retail", (np.array(perf_scores)[20:]<.15).sum()/len(perf_scores[20:])*100)
    # print("Probability of predicting within 15% from optimum PERF*COST - Retail", (np.array(perfcost_scores)[20:]<.15).sum()/len(perfcost_scores[20:])*100)

    print("Probability of predicting within 15% from optimum COST - TOTAL", (np.array(cost_scores)<.15).sum()/len(cost_scores)*100)
    print("Probability of predicting within 15% from optimum PERF - TOTAL", (np.array(perf_scores)<.15).sum()/len(perf_scores)*100)
    print("Probability of predicting within 15% from optimum PERF*COST - TOTAL", (np.array(perfcost_scores)<.15).sum()/len(perfcost_scores)*100)


# print(airline_acc_table.to_latex(float_format="%.1f"))
# print(face_acc_table.to_latex(float_format="%.1f"))
# print(event_acc_table.to_latex(float_format="%.1f"))
# print(retail_acc_table.to_latex(float_format="%.1f"))


print("Overall RMSE:", (sum(all_sq_errs)/len(all_sq_errs))**0.5)