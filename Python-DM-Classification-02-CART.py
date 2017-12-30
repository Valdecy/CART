############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Decision Trees - CART

# Citation: 
# PEREIRA, V. (2018). Project: CART, File: Python-DM-Classification-02-CART.py, GitHub repository: <https://github.com/Valdecy/CART>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np
from random import randint
from scipy import stats
from copy import deepcopy

# Function: Returns True, if a Column is Numeric
def is_number(string):
    for i in range(0, len(string)):
        if pd.isnull(string[i]) == False:          
            try:
                float(string[i])
                return True
            except ValueError:
                return False

# Function: Returns True, if a Value is Numeric
def is_number_value(value):
    if pd.isnull(value) == False:          
        try:
            float(value)
            return True
        except ValueError:
            return False

# Function: Performs a Chi_Squared Test or Fisher Exact Test           
def chi_squared_test(label_df, feature_df):
    label_df.reset_index(drop=True, inplace=True)
    feature_df.reset_index(drop=True, inplace=True)
    data = pd.concat([pd.DataFrame(label_df.values.reshape((label_df.shape[0], 1))), feature_df], axis = 1)
    data.columns=["label", "feature"]
    contigency_table = pd.crosstab(data.iloc[:,0], data.iloc[:,1], margins = False)
    m = contigency_table.values.sum()
    if m <= 10000 and contigency_table.shape == (2,2):
        p_value = stats.fisher_exact(contigency_table)
    else:
        p_value = stats.chi2_contingency(contigency_table, correction = False) # (No Yates' Correction)
    return p_value[1]

# Function: Prediction           
def prediction_dt_cart(model, Xdata):
    Xdata = Xdata.reset_index(drop=True)
    ydata = pd.DataFrame(index=range(0, Xdata.shape[0]), columns=["Prediction"])
    data  = pd.concat([ydata, Xdata], axis = 1)
    rule = []
    for j in range(0, data.shape[1]):
        if data.iloc[:,j].dtype == "bool":
            data.iloc[:,j] = data.iloc[:, j].astype(str)
    
    dt_model = deepcopy(model)

    count = 0
    end_count = data.shape[1]
    while (count < end_count-1):
        count = count + 1
        if is_number(data.iloc[:, 1]) == False:
            col_name = data.iloc[:, 1].name
            new_col  = data.iloc[:, 1].unique()
            for k in range(0, len(new_col)):
                one_hot_data = data.iloc[:, 1]
                one_hot_data = pd.DataFrame({str(col_name) + "[" + str(new_col[k]) + "]": data.iloc[:, 1]})
                for L in range (0, one_hot_data.shape[0]):
                    if one_hot_data.iloc[L, 0] == new_col[k]:
                        one_hot_data.iloc[L, 0] = " 1 "
                    else:
                        one_hot_data.iloc[L, 0] = " 0 "
                data = pd.concat([data, one_hot_data.astype(np.int32)], axis = 1)
            data.drop(col_name, axis = 1, inplace = True)
            end_count = data.shape[1]
        else:
            col_name = data.iloc[:, 1].name
            one_hot_data = data.iloc[:, 1]
            data.drop(col_name, axis = 1, inplace = True)
            data = pd.concat([data, one_hot_data], axis = 1)

    # Preprocessing - Binary Values
    for i in range(0, data.shape[0]):
        for j in range(1, data.shape[1]):
            if data.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
               if data.iloc[i,j] == 0:
                   data.iloc[i,j] = str(0)
               else:
                   data.iloc[i,j] = str(1)
    
    for i in range(0, len(dt_model)):
        dt_model[i] = dt_model[i].replace("{", "")
        dt_model[i] = dt_model[i].replace("}", "")
        dt_model[i] = dt_model[i].replace(".", "")
        dt_model[i] = dt_model[i].replace("IF ", "")
        dt_model[i] = dt_model[i].replace("AND", "")
        dt_model[i] = dt_model[i].replace("THEN", "")
        dt_model[i] = dt_model[i].replace("=", "")
        dt_model[i] = dt_model[i].replace("<", "<=")
        dt_model[i] = dt_model[i].replace(" 0 ", "<=0")
        dt_model[i] = dt_model[i].replace(" 1 ", ">0")
    
    for i in range(0, len(dt_model) -2): 
        splited_rule = [x for x in dt_model[i].split(" ") if x]
        rule.append(splited_rule)
   
    for i in range(0, Xdata.shape[0]): 
        for j in range(0, len(rule)):
            rule_confirmation = len(rule[j])/2 - 1
            rule_count = 0
            for k in range(0, len(rule[j]) - 2, 2):
                if (rule[j][k] in list(data.columns.values)) == False:
                    zeros = pd.DataFrame(0, index = range(0, data.shape[0]), columns = [rule[j][k]])
                    data  = pd.concat([data, zeros], axis = 1)
                if is_number_value(data[rule[j][k]][i]) == False:
                    if (data[rule[j][k]][i] in rule[j]):
                        rule_count = rule_count + 1
                        if (rule_count == rule_confirmation):
                            data.iloc[i,0] = rule[j][len(rule[j]) - 1]
                    else:
                        k = len(rule[j])
                elif is_number_value(data[rule[j][k]][i]) == True:
                     if rule[j][k+1].find("<=") == 0:
                         if float(data[rule[j][k]][i]) <= float(rule[j][k+1].replace("<=", "")): 
                             rule_count = rule_count + 1
                             if (rule_count == rule_confirmation):
                                 data.iloc[i,0] = rule[j][len(rule[j]) - 1]
                         else:
                             k = len(rule[j])
                     elif rule[j][k+1].find(">") == 0:
                         if float(data[rule[j][k]][i]) > float(rule[j][k+1].replace(">", "")): 
                             rule_count = rule_count + 1
                             if (rule_count == rule_confirmation):
                                 data.iloc[i,0] = rule[j][len(rule[j]) - 1]
                         else:
                             k = len(rule[j])
    
    for i in range(0, Xdata.shape[0]):
        if pd.isnull(data.iloc[i,0]):
            data.iloc[i,0] = dt_model[len(dt_model)-1]
    
    return data

# Function: Calculates the Gini Index  
def gini_index(target, feature = [], uniques = []):
    gini = 0
    weighted_gini = 0
    denominator_1 = feature.count()
    data = pd.concat([pd.DataFrame(target.values.reshape((target.shape[0], 1))), feature], axis = 1)
    for word in range(0, len(uniques)):
        denominator_2 = feature[(feature == uniques[word])].count() #12
        if denominator_2[0] > 0:
            for lbl in range(0, len(np.unique(target))):
                numerator_1 = data.iloc[:,0][(data.iloc[:,0] == np.unique(target)[lbl]) & (data.iloc[:,1]  == uniques[word])].count()
                if numerator_1 > 0:
                    gini = gini + (numerator_1/denominator_2)**2 
        gini = 1 - gini
        weighted_gini = weighted_gini + gini*(denominator_2/denominator_1)
        gini = 0
    return float(weighted_gini)

# Function: Binary Split on Continuous Variables 
def split_me(feature, split):
    result = pd.DataFrame(feature.values.reshape((feature.shape[0], 1)))
    for fill in range(0, len(feature)):
        result.iloc[fill,0] = feature.iloc[fill]
    lower = "<=" + str(split)
    upper = ">" + str(split)
    for convert in range(0, len(feature)):
        if float(feature.iloc[convert]) <= float(split):
            result.iloc[convert,0] = lower
        else:
            result.iloc[convert,0] = upper
    binary_split = []
    binary_split = [lower, upper]
    return result, binary_split

# Function: CART Algorithm
def dt_cart(Xdata, ydata, cat_missing = "none", num_missing = "none", pre_pruning = "none", chi_lim = 0.1, min_lim = 5):
    
    ################     Part 1 - Preprocessing    #############################
    # Preprocessing - Creating Dataframe
    name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    dataset = pd.concat([ydata, Xdata], axis = 1)
    
     # Preprocessing - Boolean Values
    for j in range(0, dataset.shape[1]):
        if dataset.iloc[:,j].dtype == "bool":
            dataset.iloc[:,j] = dataset.iloc[:, j].astype(str)

    # Preprocessing - Missing Values
    if cat_missing != "none":
        for j in range(1, dataset.shape[1]): 
            if is_number(dataset.iloc[:, j]) == False:
                for i in range(0, dataset.shape[0]):
                    if pd.isnull(dataset.iloc[i,j]) == True:
                        if cat_missing == "missing":
                            dataset.iloc[i,j] = "Unknow"
                        elif cat_missing == "most":
                            dataset.iloc[i,j] = dataset.iloc[:,j].value_counts().idxmax()
                        elif cat_missing == "remove":
                            dataset = dataset.drop(dataset.index[i], axis = 0)
                        elif cat_missing == "probability":
                            while pd.isnull(dataset.iloc[i,j]) == True:
                                dataset.iloc[i,j] = dataset.iloc[randint(0, dataset.shape[0] - 1), j]            
    elif num_missing != "none":
            if is_number(dataset.iloc[:, j]) == True:
                for i in range(0, dataset.shape[0]):
                    if pd.isnull(dataset.iloc[i,j]) == True:
                        if num_missing == "mean":
                            dataset.iloc[i,j] = dataset.iloc[:,j].mean()
                        elif num_missing == "median":
                            dataset.iloc[i,j] = dataset.iloc[:,j].median()
                        elif num_missing == "most":
                            dataset.iloc[i,j] = dataset.iloc[:,j].value_counts().idxmax()
                        elif cat_missing == "remove":
                            dataset = dataset.drop(dataset.index[i], axis = 0)
                        elif num_missing == "probability":
                            while pd.isnull(dataset.iloc[i,j]) == True:
                                dataset.iloc[i,j] = dataset.iloc[randint(0, dataset.shape[0] - 1), j]  
   
    # Preprocessing - One Hot Encode
    count = 0
    end_count = dataset.shape[1]
    while (count < end_count-1):
        count = count + 1
        if is_number(dataset.iloc[:, 1]) == False:
            col_name = dataset.iloc[:, 1].name
            new_col  = dataset.iloc[:, 1].unique()
            for k in range(0, len(new_col)):
                one_hot_data = dataset.iloc[:, 1]
                one_hot_data = pd.DataFrame({str(col_name) + "[" + str(new_col[k]) + "]": dataset.iloc[:, 1]})
                for L in range (0, one_hot_data.shape[0]):
                    if one_hot_data.iloc[L, 0] == new_col[k]:
                        one_hot_data.iloc[L, 0] = " 1 "
                    else: 
                        one_hot_data.iloc[L, 0] = " 0 "
                dataset = pd.concat([dataset, one_hot_data.astype(np.int32)], axis = 1)
            dataset.drop(col_name, axis = 1, inplace = True)
            end_count = dataset.shape[1]
        else:
            col_name = dataset.iloc[:, 1].name
            one_hot_data = dataset.iloc[:, 1]
            dataset.drop(col_name, axis = 1, inplace = True)
            dataset = pd.concat([dataset, one_hot_data], axis = 1)
    
    bin_names = list(dataset)
     
    # Preprocessing - Binary Values
    for i in range(0, dataset.shape[0]):
        for j in range(1, dataset.shape[1]):
            if dataset.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
               bin_names[j] = "binary"
               if dataset.iloc[i,j] == 0:
                   dataset.iloc[i,j] = str(0)
               else:
                   dataset.iloc[i,j] = str(1)
                
    # Preprocessing - Unique Words List
    unique = []
    uniqueWords = []
    for j in range(0, dataset.shape[1]): 
        for i in range(0, dataset.shape[0]):
            token = dataset.iloc[i, j]
            if not token in unique:
                unique.append(token)
        uniqueWords.append(unique)
        unique = []  
    
    # Preprocessing - Label Matrix
    label = np.array(uniqueWords[0])
    label = label.reshape(1, len(uniqueWords[0]))
    
    ################    Part 2 - Initialization    #############################
    # CART - Initializing Variables
    i = 0
    branch = [None]*1
    branch[0] = dataset
    gini_vector = np.empty([1, branch[i].shape[1]])
    lower = " 0 "
    root_index = 0
    rule = [None]*1
    rule[0] = "IF "
    skip_update = False
    stop = 2
    upper = " 1 "
    
    ################     Part 3 - CART Algorithm    #############################
    # CART - Algorithm
    while (i < stop):
        gini_vector.fill(1)
        for element in range(1, branch[i].shape[1]):
            if len(branch[i]) == 0:
                skip_update = True 
                break
            if len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1:
                 if "." not in rule[i]:
                     rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + "."
                     rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                     if i == 1 and (rule[i].find("{0}") != -1 or rule[i].find("{1}")!= -1):
                         rule[i] = rule[i].replace(".", "")
                 skip_update = True
                 break
            if i > 0 and is_number(dataset.iloc[:, element]) == False and pre_pruning == "chi_2" and chi_squared_test(branch[i].iloc[:, 0], branch[i].iloc[:, element]) > chi_lim:
                 if "." not in rule[i]:
                     rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x:x.value_counts().index[0])[0] + "."
                     rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                 skip_update = True
                 continue
            if is_number(dataset.iloc[:, element]) == True and bin_names[element] != "binary":
                gini_vector[0, element] = 1.0
                value = np.sort(branch[i].iloc[:, element].unique())
                skip_update = False
                for bin_split in range(0, len(value)):
                    bin_sample = split_me(feature = branch[i].iloc[:, element], split = value[bin_split])
                    if i > 0 and pre_pruning == "chi_2" and chi_squared_test(branch[i].iloc[:, 0], bin_sample[0]) > chi_lim:
                        if "." not in rule[i]:
                             rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x:x.value_counts().index[0])[0] + "."
                             rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                        skip_update = True
                        continue
                    g_index = gini_index(target = branch[i].iloc[:, 0], feature = bin_sample[0], uniques = bin_sample[1])
                    if g_index < float(gini_vector[0, element]):
                        gini_vector[0, element] = g_index
                        uniqueWords[element] = bin_sample[1]
            if (is_number(dataset.iloc[:, element]) == False or bin_names[element] == "binary"):
                gini_vector[0, element] = 1.0
                skip_update = False
                g_index = gini_index(target = branch[i].iloc[:, 0], feature =  pd.DataFrame(branch[i].iloc[:, element].values.reshape((branch[i].iloc[:, element].shape[0], 1))), uniques = uniqueWords[element])
                gini_vector[0, element] = g_index
            if i > 0 and pre_pruning == "min" and len(branch[i]) <= min_lim:
                 if "." not in rule[i]:
                     rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x:x.value_counts().index[0])[0] + "."
                     rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                 skip_update = True
                 continue
                   
        if skip_update == False:
            root_index = np.argmin(gini_vector)
            rule[i] = rule[i] + list(branch[i])[root_index]          
            for word in range(0, len(uniqueWords[root_index])):
                uw = str(uniqueWords[root_index][word]).replace("<=", "")
                uw = uw.replace(">", "")
                lower = "<=" + uw
                upper = ">" + uw
                if uniqueWords[root_index][word] == lower and bin_names[root_index] != "binary":
                    branch.append(branch[i][branch[i].iloc[:, root_index] <= float(uw)])
                elif uniqueWords[root_index][word] == upper and bin_names[root_index] != "binary":
                    branch.append(branch[i][branch[i].iloc[:, root_index]  > float(uw)])
                else:
                    branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])
                node = uniqueWords[root_index][word]
                rule.append(rule[i] + " = " + "{" +  str(node) + "}")            
            for logic_connection in range(1, len(rule)):
                if len(np.unique(branch[i][0])) != 1 and rule[logic_connection].endswith(" AND ") == False  and rule[logic_connection].endswith("}") == True:
                    rule[logic_connection] = rule[logic_connection] + " AND "
        
        skip_update = False
        i = i + 1
        print("iteration: ", i)
        stop = len(rule)
    
    for i in range(len(rule) - 1, -1, -1):
        if rule[i].endswith(".") == False:
            del rule[i]    
    
    rule.append("Total Number of Rules: " + str(len(rule)))
    rule.append(dataset.agg(lambda x:x.value_counts().index[0])[0])
    
    return rule

    ############### End of Function ##############

######################## Part 4 - Usage ####################################

df = pd.read_csv('Python-DM-Classification-02-CART.csv', sep = ';')

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

dt_model = dt_cart(X, y)

# Prediction
test =  df.iloc[0:2, 0:4]
prediction_dt_cart(dt_model, test)

########################## End of Code #####################################
