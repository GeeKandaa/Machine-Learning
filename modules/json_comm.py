import argparse
import json
# from mpi4py import MPI
#     comm= MPI.COMM_WORLD
#     rank = comm.Get_rank()
# def mpi_get_param():
#     if rank == 0:
#         goahead = True
#         comm.send(goahead, dest=1)
#     else:
#         goahead=comm.recv(source=rank-1)
#         return get_param()

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
                    #print("hit:",entry," | ", entry_key,",",entry_val)
                    entry[entry_key] = None
                    continue
                if entry_val.lower() == "true":
                    #print("hit:",entry," | ", entry_key,",",entry_val)
                    entry[entry_key] = True
                    continue
                if entry_val.lower() == "false":
                    #print("hit:",entry," | ", entry_key,",",entry_val)
                    entry[entry_key] = False
                    continue
                if entry_val[0] == "[" and entry_val[-1] == "]":
                    entry[entry_key] = entry_val.strip("[]").split(",")
                try:
                    test = float(entry_val)
                    if test.is_integer:
                        entry[entry_key] = int(entry_val)
                    else:
                        entry[entry_key] = test
                    # print("hit:",entry," | ", entry_key,",",entry_val)
                except:
                    continue
                    # if entry_val.isnumeric():
                    #     entry[entry_key] = int(entry_val)
                    #     # print("hit:",entry," | ", entry_key,",",entry_val)

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

        param_vals["compile"] = {
            "loss":assigned["compile"]["loss"],
            "metrics":assigned["compile"]["metrics"],
            "loss_weights":assigned["compile"]["loss_weights"],
            "weighted_metrics":assigned["compile"]["weighted_metrics"],
            "run_eagerly":assigned["compile"]["run_eagerly"],
        }
        param_vals["iteration"] = {
            "#":assigned["iteration"]["#"]
        }
        param_vals["model"] = {
            "data":assigned["model"]["data"],
            "save":assigned["model"]["save"],
            "epoch":assigned["model"]["epoch"]
        }
    print(" _____________________________________________________\n|Non-Default Parameters:")
    print("|\n|     optimiser:",args.optimiser)
    print("|\n|     callback:",args.callback)
    print("|\n|     compile:",args.compile)
    print("|\n|     model:",args.model)
    print("|_____________________________________________________")

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

    # cont=input("print params.")
    # print(assigned)
    # print("______________________________________________")
    print(param_vals)
    # cont=input("END.")
    return param_vals

def store_data(id, vals, duplicates=True):
    try:
        with open('Support_Files\output_data.json') as json_file:
            existing_data = json.load(json_file)           
            if id in existing_data:
                print("id exists")
                data = existing_data[id]
                print("previous: ", data)
                for val in vals:
                    if val in data:
                        if duplicates == True:
                            data.append(val)
                            print("duplicate new: ", val)
                            existing_data[id] = data
                        else:
                            print("avoiding duplicate")
                            return
                    else:
                        print("non-dupe new: ", val)
                        data.append(val)
                        existing_data[id] = data
            else:
                print("1",existing_data)
                existing_data[id] = vals
                print("2 created id: ",id,":",existing_data[id])
            json.dump(existing_data, open('Support_Files\output_data.json','w'))
            print("3",existing_data)
    except:
        data = {id:vals}
        print("created file: ",data)
        with open('Support_Files\output_data.json','w') as json_file:
            json.dump(data, json_file)

def verify_data_length():
    with open('Support_Files\output_data.json') as json_file:
        existing_data = json.load(json_file) 
        check_length=min([len(existing_data[ele]) for ele in existing_data])
        print(check_length)
        print(existing_data)
        for id in existing_data:
            print(existing_data[id])
            while len(existing_data[id]) > check_length:
                print("list too long removing element")
                del existing_data[id][0]
    json.dump(existing_data, open('Support_Files\output_data.json','w'))
        
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
            json.dump(existing_data, open('Support_Files\output_mp_data.json','w'))
    except:
        data = {}
        for experiment in store_data:
            exp_id=experiment[0]
            data[exp_id]={}
            for Data_n in experiment[1]:
                data[exp_id][Data_n[0]]=Data_n[1]
            json.dump(data, open('Support_Files\output_mp_data.json','w'))


def display_graph(x,y):
    import os
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    with open('..\Support_Files\output_data.json') as json_file:
            data = json.load(json_file)
            if isinstance(y,list):
                fig, ax = plt.subplots(figsize = (20,5))
                ax.set(xlabel=x, ylabel=y[len(y)-1])
                for i in range(len(y)-1):
                    ax.plot(x,y[i],data=data,label=y[i],marker="x")
                ax.legend(loc='best')
                plt.show()
                return
            # ax.set(xlim=(-0.00001,0.01))
            plt.figure(figsize = (20,5))
            plt.title(x+' vs '+y)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.show()
            json_file.close()
            os.remove(json_file.name)
            ax = sb.lineplot(x = data[x], y = data[y], color='red')
            # ax.set(xlim=(-0.00001,0.01))
            plt.title(x+' vs '+y)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.show()
            json_file.close()
            # os.remove(json_file.name)
