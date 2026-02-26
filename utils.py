import bamnostic as bs
from pathlib import Path
import numpy as np
from scipy.special import softmax 

from numba import jit, njit

home = Path(__file__).parent
decode_map = np.array(['A', 'C', 'G', 'T', 'N'])
complement_map = np.array([3, 2, 1, 0, 4])

class utils:

    def io_monster(mode, filename="SRR9090854.subsampled_5pct.bam"):
        '''
        Docstring for io_monster
        
        :param mode: Specifies whether to load reads from file or to load and save as npz for fast retreival
        '''

        def read_encode_bam():
            seqs = [read.seq for read in bs.AlignmentFile(bam_path)]
            lengths = np.array([len(seq) for seq in seqs], dtype=np.int32)      #store the indices of the reads so we can retrieve them later
            all_bytes = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)   #flatten the whole genome into a byte array (strings -> bytes)
            encode_map = utils.init_base_encoding_map()
            seqs = utils.encode_sequences(all_bytes, encode_map)                #use the fast numba function to encode the bytes as nucleotides (bytes -> encodings)
            indptr = np.zeros(len(lengths) + 1, dtype=np.int32)                 #build indptr (map of indices in flat array)
            indptr[1:] = np.cumsum(lengths)                                     #get the fenceposts of all sequences in the flat array
            np.savez_compressed(npz_path, seqs=seqs, indptr=indptr)


        npz_name = "processed_data.npz"
        bam_path = home / "data" / filename
        npz_path = home / "data" / npz_name

        if mode == "test":                                                  #use a test file if testmode specified
            print("loading bam test data query sequences")
            bam = bs.AlignmentFile(bs.example_bam, 'rb')
            seqs = [seq.query_sequence for seq in bam]
        elif mode == "vanilla":
            print("loading directly from bam (SLOW)")
            bam = bs.AlignmentFile(bs.example_bam, 'rb')
            seqs = [seq.query_sequence for seq in bam]
            return seqs
        else:                                                               #If n_rows != 0, use a text file containing the specified number of rows
            if not npz_path.exists():                                       #text file does not exist
                print(f"{npz_name} does not exist, get ready for bammage. reading from bam.")
                read_encode_bam()
            else:
                print(f"{npz_name} exists, reading from path")
            npz = np.load(npz_path)
            print("data ready :)")
            return npz["seqs"], npz["indptr"]
        return seqs, None
    
    def init_base_encoding_map():
        '''
        Returns mapping for binary/ASCII (not sure) encoded strings to numbers
        '''
        base_map = np.zeros(256, dtype=np.int8)
        base_map[ord('A')] = 0
        base_map[ord('C')] = 1
        base_map[ord('G')] = 2
        base_map[ord('T')] = 3
        base_map[ord('N')] = 4
        return base_map
    
    def decode_sequence(seq):
        '''        
        :param seq: takes an encoded sequence and converts it back to a string with decode map
        '''
        return "".join(decode_map[seq])

    
    def seq_to_array(seq, base_map):
        '''        
        :param seq: converts a string sequence to an encoded array
        :param base_map: the encoding map initialized with the encoding map method
        '''
        seq_array = np.frombuffer(seq.encode(), dtype=np.int8)
        indices = base_map[seq_array]
        return indices
    
    '''
    The rest are numba functions for fast iterating through numpy arrays
    '''
    
    @njit
    def encode_sequences(all_bytes, base_map):
        mdata = np.empty(len(all_bytes), dtype=np.int8)
        for i in range(len(all_bytes)):
            mdata[i] = base_map[all_bytes[i]]
        return mdata
    
    @njit
    def fast_init(flat_seqs, n_rows, indptr, k):
        motifs = np.zeros((n_rows, k), dtype=np.int8)                   #init array for motif storage
        idxs = np.zeros(n_rows, dtype=np.int32)                         #init array for motif indices
        for i in range(n_rows):
            idx = np.random.randint(indptr[i], indptr[i+1] - k + 1)     #choose random index
            motifs[i] = flat_seqs[idx:idx+k]                            #get the motif
            idxs[i] = idx + indptr[i]                                   #save the chosen motif index at its position within that sequence
        return motifs, idxs
    
    @njit
    def build_pfm_fast(motifs, k, n_rows, n_bases=5, slice_last_row=True):
        pfm = np.zeros((n_bases, k), dtype=np.int32)
        for i in range(n_rows):
            for j in range(k):
                pfm[motifs[i, j], j] += 1
        if slice_last_row:
            pfm[0, :] += pfm[-1, :]
            return pfm[:-1, :]
        return pfm
    
    @njit
    def subtract_pfm(motif, k, pfm):            #simple update the pfm
            for j in range(k):
                pfm[motif[j], j] -= 1
            return pfm
    
    @njit
    def add_pfm(motif, k, pfm):            #simple update the pfm
        for j in range(k):
            pfm[motif[j], j] += 1
        return pfm
    
    @njit
    def fast_complement(seq):               #reverse sequences
        return complement_map[seq[::-1]]
    
    @njit
    def fast_subdivide(seq, rev_seq, k):                            #splits the fwd and rev sequences into all possible motifs
        motif_indices = len(seq) - k
        fwd_motifs = np.zeros((motif_indices, k), dtype=np.int8)
        rev_motifs = np.zeros((motif_indices, k), dtype=np.int8)
        for i in range(len(seq) - k):
            fwd_motifs[i] = seq[i:i+k]
            rev_motifs[i] = rev_seq[i:i+k]
        return fwd_motifs, rev_motifs

    @njit
    def fast_score(motif_array, pwm):                                   #takes the scoring pwm and basically does what marcus's does
        scores = np.zeros(len(motif_array), dtype=np.float64)
        for i in range(len(motif_array)):
            motif = motif_array[i]
            score = 0
            for j in range(len(motif)):
                score += pwm[motif[j], j]
            scores[i] = score
        return scores

    def choose_best(fwd, rev, pwm, p_method, temp=0.1):                 #scoring function that can take a variety of probability methods
        methods = {"softmax": lambda x: softmax(x / temp)}              #we chose softmax because this is common in ML and LLMs
        fwd_score = utils.fast_score(fwd, pwm)
        rev_score = utils.fast_score(rev, pwm)
        score_dist = np.concatenate([fwd_score, rev_score], axis=0)     #iffy choice to concatenate both scores into one distribution
        if p_method not in methods:
            raise ValueError("improper probability method")
        p_dist = methods[p_method](score_dist)
        all_motifs = np.concatenate([fwd, rev], axis=0)
        idx =  np.random.choice(len(all_motifs), p=p_dist)
        return all_motifs[idx]



        