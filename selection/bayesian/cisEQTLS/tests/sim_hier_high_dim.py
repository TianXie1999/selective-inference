from __future__ import print_function
import argparse
import os
import sys
import numpy as np

# local import functions
from sim_ciseqtl import read_X, read_y, read_simes, read_signals, mkdir_p
from hierarchical_high_dimensional import hierarchical_inference, evaluate_hierarchical_results


def do_evaluation(args):
    sys.stderr.write("Evaluating lasso selection and inference results\n")
    s = int(args.seed)
    sel_type = str(args.seltype)
    idx_start = int(args.begin)
    idx_end = int(args.end)

    trial_dir = os.path.join(args.indir,"trial_"+str(s))

    sig_fname = os.path.join(trial_dir,"signals.txt")
    simes_fname = os.path.join(trial_dir,"simes_result.txt") 
    try:
        true_sig, snr, _ = read_signals(sig_fname)
    except OSError:
        sys.stderr.write('Cannot open: '+sig_fname+"\n")
    try:
        simes_sel = read_simes(simes_fname, full=True) 
    except OSError:
        sys.stderr.write('Cannot open: '+simes_fname+"\n")

    
    for i in xrange(idx_start,idx_end):
        num_true_sig = true_sig[i]
        sel = simes_sel[0][i]
        if not sel: 
            sys.stderr.write("Family "+str(i)+" was not selected by Simes\n")
            continue

        result_fname = os.path.join(trial_dir, "infs_"+sel_type, "sel_out_"+str(i)+".txt")
        if not os.path.isfile(result_fname):
            sys.stderr.write("Family "+str(i)+" was selected by Simes, but second stage selection was not complete\n")
            continue
        else:
            sys.stderr.write("Family "+str(i)+" was selected by Simes, and Lasso selected variables\n")
            result = np.loadtxt(result_fname) 

        X_fname = os.path.join(args.indir,"X_data","X_"+str(i)+".txt")
        try:
            X = read_X(X_fname)
        except OSError:
            sys.stderr.write('Cannot open: '+X_fname+"\n")
        # adjusted_coverage, unadjusted_coverage, FDR, power = evaluate_hierarchical_results(result, X, s, snr)
        res_eval = evaluate_hierarchical_results(result, X, num_true_sig, snr)
        print("{}\t{}\t{}\t{}".format(*res_eval))

def do_inference(args):
    sys.stderr.write("Running lasso selection and inference\n")
    np.random.seed(0)
    s = int(args.seed)
    i = int(args.geneid)
    sel_type = str(args.seltype)
    n_genes = int(args.ngenes)

    trial_dir = os.path.join(args.indir,"trial_"+str(s))
    res_dir = os.path.join(trial_dir,"infs_"+sel_type)
    mkdir_p(res_dir)

    X_f = os.path.join(args.indir,"X_data","X_"+str(i)+".txt")
    y_f = os.path.join(trial_dir,"y_data", "y_"+str(i)+".txt")
    # sime_fname = os.path.join(trial_dir,"simes_result.txt")
    s_f = os.path.join(trial_dir,"tmp","s_"+str(i)+".txt")
    sel_res = read_simes(s_f)
    sel = sel_res[0]

    if not sel: 
        sys.stderr.write("Family "+str(i)+" was not selected by Simes\n")
    else:
        sys.stderr.write("Family "+str(i)+" was selected by Simes\n")
        X = read_X(X_f)
        y = read_y(y_f)
        index = sel_res[1][i]
        idx_order = sel_res[2][i]
        sign = sel_res[3][i]
        simes_level = sel_res[4]
        rej = sel_res[5][i]

        n_sel = np.sum(np.array(sel_res[0]))
        n_samples, n_variables = X.shape

        pgenes = 1.0 * n_sel / n_genes 
        sys.stderr.write("Read X, y and eGene selection result data\n")
        sys.stderr.write("Number of samples: "+str(n_samples)+"\n")
        sys.stderr.write("Number of variables: "+str(n_variables)+"\n")
        sys.stderr.write("Proportion of selected genes: "+str(pgenes)+"\n")
        sys.stderr.write("Uniform Sime's simes_level: "+str(simes_level)+"\n")
        sys.stderr.write("Other params: "+str(index)+", "+str(idx_order)+", "+str(sign)+"\n")
        sys.stderr.write("Writing results to: "+res_dir+"\n")
        sys.stderr.write("Selection mode: "+sel_type+"\n")
        
        list_results = hierarchical_inference(X, y, index, simes_level, pgenes, J=rej, t_0=idx_order, T_sign=sign, selection_method=sel_type)
        if list_results is None:
            sys.stderr.write("Result is None (likely Lasso did not select any variables)\n")
            result_file = os.path.join(res_dir,"no_sel_"+str(i)+".txt")
            np.savetxt(0)
        else:
            if args.justsel: # do not do inference to save time
                sys.stderr.write("No infrence is performed.\n")
            else:
                result_file = os.path.join(res_dir,"sel_out_"+str(i)+".txt")
                np.savetxt(result_file,list_results)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="run hierarchical inference or evaluate its simulation results")
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('inference', help='') 
    command_parser.add_argument('-d', '--indir', required=True)
    command_parser.add_argument('-s', '--seed', required=True)
    command_parser.add_argument('-i', '--geneid', required=True)
    command_parser.add_argument('-g', '--ngenes', required=True)
    command_parser.add_argument('-t', '--seltype', required=True)
    command_parser.add_argument('-j', '--justsel', action='store_true', default=False, help="If true, ignore inference and just do selection")

    command_parser.set_defaults(func=do_inference)

    command_parser = subparsers.add_parser('evaluate', help='') 
    command_parser.add_argument('-d', '--indir', required=True)
    command_parser.add_argument('-s', '--seed', required=True)
    # command_parser.add_argument('-i', '--geneid', required=True)
    command_parser.add_argument('-b', '--begin', required=True)
    command_parser.add_argument('-e', '--end', required=True)
    command_parser.add_argument('-t', '--seltype', required=True)
    command_parser.set_defaults(func=do_evaluation)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


