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
            if entry_val == "None":
                print("hit:",entry," | ", entry_key,",",entry_val)
                entry[entry_key] = None
                continue
            if entry_val == "True":
                print("hit:",entry," | ", entry_key,",",entry_val)
                entry[entry_key] = True
                continue
            if entry_val == "False":
                print("hit:",entry," | ", entry_key,",",entry_val)
                entry[entry_key] = False
                continue
            if isinstance(entry_val, str):
                try:
                    entry[entry_key] = float(entry_val)
                    print("hit:",entry," | ", entry_key,",",entry_val)
                except:
                    if entry_val.isnumeric():
                        entry[entry_key] = int(entry_val)
                        print("hit:",entry," | ", entry_key,",",entry_val)
                continue

def get_param():
    parser =  argparse.ArgumentParser(description='Assign hyperparameters and run experiments.')
    subparsers = parser.add_subparsers(help="sub-command help")

    optimiser_parser = subparsers.add_parser('optimiser', help="optimiser help")
    parser.add_argument("--optimiser", type=pairs, nargs='+', help="define optimiser parameters as key-value pairs")

    callback_parser = subparsers.add_parser('callback', help="callback help")
    parser.add_argument("--callback", type=pairs, action='append', nargs='+', help="define callback parameters as key-value pairs")

    compile_parser = subparsers.add_parser('compile', help="compile help")
    parser.add_argument("--compile", type=pairs, nargs='+', help="define compile parameters as key-value pairs")

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

    print("\noptimiser:",args.optimiser)
    print("\ncallback:",args.callback)
    print("\ncompile:",args.compile)

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

    print(param_vals)

    for entry_key,entry_val in param_vals.items():
        UnString_Value(entry_val)

    cont=input("print params.")
    print(assigned)
    print("______________________________________________")
    print(param_vals)
    cont=input("END.")
    return param_vals