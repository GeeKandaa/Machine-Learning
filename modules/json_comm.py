import argparse
import json

def pairs(arg):
    left = arg.split(":")[0] 
    right = arg.split(':')[1]
    if right == "":
        right = "subsection_id"
    return [left, right]

def UnString_Value(entry):
    for entry_key,entry_val in entry.items():
        if type(entry_val) is dict:
            UnString_Value(entry_val)
        else:
            if isinstance(entry_val, str):
                entry_val.strip()
                if entry_val.lower() == "none":
                    entry[entry_key] = None
                    continue
                if entry_val.lower() == "true":
                    entry[entry_key] = True
                    continue
                if entry_val.lower() == "false":
                    entry[entry_key] = False
                    continue
                if entry_val[0] == "[" and entry_val[-1] == "]":
                    entry[entry_key] = entry_val.strip("[]").split(",")
                    for i in range(0,len(entry[entry_key])):
                        try:
                            test = float(entry[entry_key][i])
                            if test.is_integer():
                                entry[entry_key][i] = int(test)
                            else:
                                entry[entry_key][i] = test
                        except:
                            continue
                try:
                    test = float(entry_val)
                    if test.is_integer():
                        entry[entry_key] = int(entry_val)
                    else:
                        entry[entry_key] = test
                except:
                    continue


def get_param():
    import argparse
    parser =  argparse.ArgumentParser(description='Assign hyperparameters and run experiments.')
    subparsers = parser.add_subparsers(help="sub-command help")

    optimiser_parser = subparsers.add_parser('optimiser', help="optimiser help")
    parser.add_argument("--optimiser", type=pairs, nargs='+', help="define optimiser parameters as key-value pairs")

    callback_parser = subparsers.add_parser('callback', help="callback help")
    parser.add_argument("--callback", type=pairs, action='append', nargs='+', help="define callback parameters as key-value pairs")

    compile_parser = subparsers.add_parser('compile', help="compile help")
    parser.add_argument("--compile", type=pairs, nargs='+', help="define compile parameters as key-value pairs")

    compile_parser = subparsers.add_parser('store_multiple', help="store_multiple help")
    parser.add_argument("--store_multiple", type=str, help="store data from multiple repeated experiments")

    compile_parser = subparsers.add_parser('model', help="model help")
    parser.add_argument("--model", type=pairs, nargs='+', help="define model parameters")

    compile_parser = subparsers.add_parser('gridsearch', help="gridsearch help")
    parser.add_argument("--gridsearch", type=pairs, nargs='+', help="pass parameter to gridsearch as key:value pair, define 'step':value to alter step size.")

    args = parser.parse_args()

    param_vals = {}
    with open('Support_Files\parameters.json') as json_file:
        assigned = json.load(json_file)
        if assigned["optimiser"]["Active"]==True:
            param_vals["optimiser"] = {
                "name":assigned["optimiser"]["name"],
                "lr":assigned["optimiser"]["lr"],
                "beta_1":assigned["optimiser"]["beta_1"],
                "beta_2":assigned["optimiser"]["beta_2"],
                "epsilon":assigned["optimiser"]["epsilon"],
                "decay":assigned["optimiser"]["decay"]
            }
        if assigned["callback"]["Active"]==True:
            param_vals["callback"] = {}
            for entry_key,entry_val in assigned["callback"].items():
                if type(entry_val) is dict and entry_val["Active"]==True:
                    param_vals["callback"][entry_key] = {}
                    for param in entry_val["params"]:
                        param_vals["callback"][entry_key][param] = assigned["callback"][entry_val["name"]][param]
                elif not(type(entry_val) is dict):
                    param_vals["callback"][entry_key] = assigned["callback"][entry_key]


        param_vals["compile"] = {
            "loss":assigned["compile"]["loss"],
            "metrics":assigned["compile"]["metrics"],
            "loss_weights":assigned["compile"]["loss_weights"],
            "weighted_metrics":assigned["compile"]["weighted_metrics"],
            "run_eagerly":assigned["compile"]["run_eagerly"]
        }
        param_vals["iteration"] = {
            "#":assigned["iteration"]["#"]
        }
        param_vals["model"] = {
            "data":assigned["model"]["data"],
            "data_num":assigned["model"]["data_num"],
            "save":assigned["model"]["save"],
            "epoch":assigned["model"]["epoch"],
            "threshold":assigned["model"]["threshold"],
            "class_weights":assigned["model"]["class_weights"],
            "three_class_weights":assigned["model"]["three_class_weights"]
        }
    # debug
    # print(" _____________________________________________________\n|Non-Default Parameters:")
    # print("|\n|     optimiser:",args.optimiser)
    # print("|\n|     callback:",args.callback)
    # print("|\n|     compile:",args.compile)
    # print("|\n|     model:",args.model)
    # print("|_____________________________________________________")

    if args.optimiser != None:
        new_val = args.optimiser
        for i in range(len(new_val)):
            param_vals["optimiser"][new_val[i][0]]=new_val[i][1]

    if args.compile != None:
        new_val = args.compile
        for i in range(len(new_val)):
            param_vals["compile"][new_val[i][0]]=new_val[i][1]

    if args.callback != None:
        new_val = args.callback
        for i in range(len(new_val)):
            for j in range(len(new_val[i])-1):
                param_vals["callback"][new_val[i][0][0]][new_val[i][j+1][0]]=new_val[i][j+1][1]

    if args.store_multiple != None:
        param_vals["iteration"]["#"] = args.store_multiple[0]

    if args.model != None:
        new_val = args.model
        for i in range(len(new_val)):
            param_vals["model"][new_val[i][0]]=new_val[i][1]
    
    if args.gridsearch != None:
        from mpi4py import MPI
        comm= MPI.COMM_WORLD
        rank = comm.Get_rank()
        key_value = args.gridsearch
        stepsize = 1
        if len(args.gridsearch) == 2:
            stepsize = float(args.gridsearch[1][1])
        if rank == 1:
            param_vals[key_value[0][0]][key_value[0][1]]=param_vals[key_value[0][0]][key_value[0][1]]
        else:
            param_vals[key_value[0][0]][key_value[0][1]]=param_vals[key_value[0][0]][key_value[0][1]]+stepsize*(rank-1)

    for entry_key,entry_val in param_vals.items():
        UnString_Value(entry_val)

    # debug
    # print(param_vals)
    return param_vals

def pack_matrix(_matrix):
    try:
        existing_data = json.load(open('Support_Files\output_data.json'))
    except:
        existing_data = {}
    if not("Data" in existing_data):
        existing_data["Data"]={"Pack":False}
    if existing_data["Data"]["Pack"]:
        existing_data["Data"]["Packed"].append(_matrix)
    else:
        existing_data["Data"]={"Pack":True,"Packed":[_matrix]}
    with open('Support_Files\output_data.json','w') as json_file:
        json.dump(existing_data, json_file, sort_keys=True, indent=4)

def store_data(id, vals, group, duplicates=True):
    pack_matrix = []
    try:
        with open('Support_Files\output_data.json') as json_file:
            existing_data = json.load(json_file)
            if "Pack" in existing_data["Data"].items():
                pack_matrix = existing_data["Data"]["Packed"]
            group_data={}
            if group in existing_data:
                group_data = existing_data[group]
            else:
                existing_data[group] = group_data
            if id in group_data:
                # id exists
                data = group_data[id]
                for val in vals:
                    if val in data:
                        if duplicates == True:
                            data.append(val)
                            # duplicate data
                            group_data[id] = data
                        else:
                            # avoiding duplicate
                            return
                    else:
                        # unique data
                        data.append(val)
                        group_data[id] = data
            else:
                group_data[id] = vals
                # create id
        existing_data[group]=group_data
        with open('Support_Files\output_data.json','w') as json_file:
            existing_data["Data"]["Packed"]=pack_matrix
            json.dump(existing_data, json_file, sort_keys=True, indent=4)
    except:
        data = {group:{id:vals}}
        # Create file
        with open('Support_Files\output_data.json','w') as json_file:
            json.dump(data, json_file)

def store_mp_data(store_data):
    import re
    try:
        with open('Support_Files\output_mp_data.json') as json_file:
            existing_data = json.load(json_file)

            for experiment in store_data:
                exp_id=experiment[0]
                if exp_id in existing_data:
                    print("Experiment exists")
                    for Data_n in experiment[1]:
                        if Data_n[0] in existing_data[exp_id]:
                            data = existing_data[exp_id][Data_n[0]]
                            if len(Data_n)>2 and Data_n[1][0] in data:
                                continue
                            data.append(Data_n[1][0])
                            existing_data[exp_id][Data_n[0]] = data
                        else:
                            existing_data[exp_id][Data_n[0]]=Data_n[1]
                else:
                    existing_data[exp_id]={}
                    for Data_n in experiment[1]:
                        existing_data[exp_id][Data_n[0]]=Data_n[1]
                check_length=min([len(existing_data[exp_id][ele]) for ele in existing_data[exp_id]])
                print(check_length)
                print(existing_data[exp_id])
                for id in existing_data[exp_id]:
                    while len(existing_data[exp_id][id]) > check_length:
                        print("list too long removing element")
                        del existing_data[exp_id][id][0]
        json.dump(existing_data, open('Support_Files\output_mp_data.json','w'), indent=4, sort_keys=True)
    except:
        json.dump(store_data, open('Support_Files\output_mp_data.json','w'), indent=4, sort_keys=True)
       
def display_data():
    # Not involved in neural network. Simply an easily accessible, modifiable function for inspecting data from network output json.
    from os import listdir
    from os.path import isfile, join
    name=[name for name in listdir("Support_Files") if isfile(join("Support_Files",name))]
    i=0
    for nm in name:
        i+=1
        print(str(i)+' - '+nm)
    choice=input("input:")
    with open('Support_Files\\'+name[int(choice)-1]) as json_file:
        data = json.load(json_file)
        print("Select graphing option:")
        print("1 - confusion matrices")
        print("2 - plot")
        choice=input("input:")
        if choice == "1":
            while choice.lower() != "quit":
                import seaborn
                import matplotlib.pyplot as plt
    
                for i in range(0,len(data["Data"]["conf_matrix"])):
                    
                    seaborn.set(color_codes=True)
                    plt.figure(1, figsize=(9, 9))
                    seaborn.set(font_scale=2)
                    plt.title("Predicted Label")
                    
                    ax = seaborn.heatmap(data["Data"]["conf_matrix"][i], annot=True, cmap="YlGnBu",fmt='d')
                    ax.set_xticklabels(["Healthy","Pneumonia","Covid-19"]) 
                    ax.set_yticklabels(["Healthy","Infected","Covid-19"])
                    ax.set(ylabel="True Label")
                    plt.savefig("./_comm/"+str(i))
                    plt.show()
            
                choice=input("input:")
                
 
        elif choice == "2":
            import seaborn as sb
            import matplotlib.pyplot as plt
            x = [240,480,720,960,1200]
            y = []
            for i in range(0,len(data["Data"]["val_loss"])):
                y.append(data["Data"]["val_loss"][i])
            plt.figure(figsize = (20,5))
            sb.lineplot(x = x, y = y, color='red', label = 'Loss')

            plt.title('Quantity of Data vs Training Time')
            plt.legend(loc = 'best')
            plt.xlabel("# of Images")
            plt.ylabel("Validation Loss")
            plt.show()
