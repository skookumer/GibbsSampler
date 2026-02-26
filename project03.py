

import random
import numpy as np
import seqlogo
from pprint import pprint

#import function for building sequence motif & idenfitying seqs matching to motif
from data_readers import *
import seq_ops
import motif_ops

from joblib import Parallel, delayed

from utils import utils
from sequence_database import sequence_box
from collections import Counter, defaultdict

from scipy.special import softmax
from sklearn.preprocessing import normalize

bases = ["A", "T", "C", "G", "N"]
encode_map = utils.init_base_encoding_map()


def GibbsMotifFinder(seqs=None, k=6, n_rows=3083497, subsample_size=1e10,
                     mode="norm", speed="fast", p_method="softmax", 
                     rtol=1e-5, atol=1e-10, max_iter=1024, seed=None, toprint=True, 
                     temp=0.1):
    '''
    Function to find a pfm from a list of strings using a Gibbs sampler
    
    Args: 
        seqs (str list): a list of sequences, not necessarily in same lengths
        k (int): the length of motif to find
        seed (int, default=None): seed for np.random
        n_rows: the number of rows to read from the file. **keep this at 3083497** for this version
        subsample_size: the size of the subsample for fast method **1e10 means no subsampling**
        mode: Not meaningful in this version
        speed: switch between pythonic and fast speed
        p_method: softmax is the only option
        rtol: relative tolerance for convergence
        atol: absolute tolerance for convergence
        max_iter: maximum number of iterations
        seed: the random seed for the algorithm
        toprint: (bool) print diagnostic statements
        temp: (float) temperature for the softmax function

    Returns:
        pfm (numpy array): dimensions are 4xlength
    '''

    '''
    -----------------------------------------------INITIALIZATION---------------------------------------
    '''

    # Use rng to make random samples/selections/numbers
    # Example: randint = rng.integer(1, 10)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    '''
    INITIALIZATION FUNCTIONS HERE:
    - load the flat sequence and indptr (indices)
    - initialize the random motifs and their corresponding indicies using a fast function
    - store all this in the sequence box object for future use
    - compute background frequencies (not in this version)
    '''

    seqs, indptr = utils.io_monster(mode)
    motifs, midx = utils.fast_init(seqs, n_rows, indptr, k)
    seq_box = sequence_box(indptr, seqs, midx, motifs, k)
    # bg = seq_box.get_bg()


    '''
    -----------------------------------------------ITERATION LOOP---------------------------------------
  
    Two speeds to pick from:
    - pythonic mode (understandable)
    - fast mode (uses object methods and numba functions for speed)
    '''
    i = 0
    converged = False
    if speed == "pythonic":

        #SLICING SUBSET
        mots = seq_box.motifs[:100]
        sample_list = []
        for j in range(len(mots)):
            sample_list.append(utils.decode_sequence(mots[j]))

        #ALTERNATE METHOD
        # mots = seq_box.get_str_list_format_motifs()

        while converged == False and i < max_iter:

            # choose one sequence from our sample list, remove and calculate pwm from remaining sequences
            mot_pick = np.random.randint(0, len(sample_list))
            sample_list.pop(mot_pick)
            pfm = motif_ops.build_pfm(sample_list, k)               #encodes and rebuilds the whole pfm each time
            pwm = motif_ops.build_pwm(pfm)

            # decode chosen sequence and get reverse complement
            seq = utils.decode_sequence(seq_box[mot_pick])
            rev_seq = seq_ops.reverse_complement(seq)

            # score each k-mer in forward sequence
            scores = []
            for x in range(len(seq) - k):
                score = motif_ops.score_kmer(seq[x:x+k], pwm)
                scores.append(score)

            # score each k-mer in reverse sequence
            for x in range(len(seq) - k):
                score = motif_ops.score_kmer(rev_seq[x:x+k], pwm)
                scores.append(score / temp)

            # choose "best" motif score using softmax calculation
            prob = softmax(scores)
            #print(prob)
            idx = np.random.choice(np.arange(len(scores)), p=prob)

            # use modulo to determine if chosen motif is on forward or reverse sequence
            quotient, remainder = divmod(idx, len(seq) - k - 1)
            if quotient == 0: # forward sequence
                new_motif = seq[remainder:remainder+k]
            else: # reverse sequence
                new_motif = rev_seq[remainder:remainder+k]

            # put the sequence we just scored back into the sample list
            seq_box.midx[mot_pick] = remainder
            sample_list.insert(mot_pick, new_motif)

            # check for convergence
            if i > 0:
                if np.allclose(pwm, pwm_old, ):
                    converged = True
                    pprint(pwm-pwm_old)

            # save pwm for next iteration
            pwm_old = pwm.copy()
            i += 1
            
    elif speed == "fast":
        
        if toprint:
            pfm = seq_box.get_pfm()
            pprint(pfm)
        if toprint:
            print(f"beginning fast iteration")

        j = 0
        
        while seq_box.check_remaining_frozen():                     #checks if all sequences have been sampled
            seq_box.unfreeze_random(subsample_size)                 #unfreezes the next random subsample for sampling and adds them to the pfm calculation
            pfm = seq_box.get_pfm(to_mask=True)                     #only build the pfm once
            converged=False
            i = 0
            while converged == False and i < max_iter:
                print(f"\rIteration {i}, subsample {j}", end="", flush=True)
                fwd_seq, selected_motif = seq_box.select_random_sampling_motif()    #select both motif and the sequence associated with it, store the index in the seq_box object
                pfm = utils.subtract_pfm(selected_motif, k, pfm)                    #just update the pfm by subtracting the current motif
                pwm = motif_ops.build_pwm(pfm)                                      #standard build_pwm function
                rev_seq = utils.fast_complement(fwd_seq)                            #get the reverse sequence
                fwd, rev = utils.fast_subdivide(fwd_seq, rev_seq, k)                #get all possible motifs in both sequences
                best_motif = utils.choose_best(fwd, rev, pwm, p_method)             #choose the best motif in either sequence
                pfm = utils.add_pfm(best_motif, k, pfm)                             #and re-add motif nucleotides to the pfm
                seq_box.update_motifs(best_motif)                                   #method update the seq_box object (as opposed to direct assignment)

                if i > 0:                                                           #check convergence
                    if np.allclose(pwm, pwm_old, rtol, atol):
                        converged = False
                pwm_old = pwm.copy()                                                #use .copy() to ensure no pointing in memory
                
                i += 1
            seq_box.reset_sampling_pool()                                           #set the indices available for sampling back to False, but keep the unfrozen indices for the pfm calculation
            j += 1

    if toprint:
        output = seq_box.get_str_list_format_motifs()                       #convert the encoded sequences to strings for printability
        pfm = seq_box.get_pfm()
        print()
        pprint(pfm)
        if i == max_iter:
                print("\nalgorithm did not converge :(((")

        print(f'\nAfter {i} iterations, final motif list: {output[:20]}')
    
    pfm = normalize(pfm.T, norm="l1", axis=1)                       #included this here to avoid normalizing and transposing outside the function
    return pfm                                                      #output is ready for seqlogo

    


def consensus(results, top_k=20):                   #outdated consensus function
    votes = Counter()
    for result in results:
        votes.update(Counter(result))
    return votes.most_common(top_k)

def run_parallel(n_runs = 32, **params):            #option to run the algorithm in parallel and aggregate results
    results = Parallel(n_jobs=-1)([delayed(GibbsMotifFinder)(**params) for _ in range(n_runs)])
    return np.sum(results, axis=1)


#code to run in parallel
# result = run_parallel(k=6, speed="fast", max_iter=10, subsample_size = 1e10, toprint=True)

# Making a fake PWM
# random_ppm = np.random.dirichlet(np.ones(4), size=6)
# print(random_ppm)
# ppm = seqlogo.Ppm(random_ppm)

if __name__=="__main__":
    # Run the gibbs sampler:
    pfm = GibbsMotifFinder(k=6, speed="fast", max_iter=6e6, toprint=True, rtol=1e-5, atol=1e-10, subsample_size=1e10)

    ppm = seqlogo.CompletePm(pfm = pfm)
    # Plot the final pfm that is generated: 
    seqlogo.seqlogo(ppm, ic_scale=False, format="png", size="large", filename="test.png")


