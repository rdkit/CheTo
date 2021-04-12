#
#  Copyright (c) 2019, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Created by Nadine Schneider, May 2019


import pandas as pd
import numpy as np

import ChemTopicModel
from ChemTopicModel import chemTopicModel, utilsEvaluation, drawTopicModel

print('\n----------------------------------------------------------')
print('----------------------------------------------------------')
print('-------------------   CHETO   ----------------------------')
print('-----------   Chemical topic modeling   ------------------')
print('----------------------------------------------------------')
print('----------------------------------------------------------\n\n')

import time
print(time.asctime())

import sklearn
from rdkit import rdBase
print('RDKit version: ',rdBase.rdkitVersion)
print('Pandas version:', pd.__version__)
print('Scikit-Learn version:', sklearn.__version__)
print('Numpy version:', np.__version__)
print(ChemTopicModel.__file__)

print('\n----------------------------------------------------------\n')

import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Please specify the path to your data file. Required format: csv; first column needs to contain a SMILES of your molecule, the following columns can be different labels for the data.')
    parser.add_argument('--numTopics', type=int, required=True, help='Please specify the number of topics for your model.')
    parser.add_argument('--fragMethod', type=str, default='Morgan', choices=['Morgan', 'RDK', 'Brics'], help='Please select your fragmentation method. Default: Morgan')
    parser.add_argument('--sizeSampleDataSet', type=float, default=1.0, 
                        help='Please choose a ratio between 0.0 and 1.0. Default: 1.0; for large datasets a value of 0.1 is recommended.')
    parser.add_argument('--rareThres', type=float, default=0.001, 
                        help='Please choose a threshold between 0.0 and 1.0. Default: 0.001; for small datasets a value of 0.01 is recommended.')
    parser.add_argument('--commonThres', type=float, default=0.1, help='Please choose a threshold between 0.0 and 1.0. Default: 0.1')
    parser.add_argument('--njobsFrag', type=int, default=1, help='Please specify the number of jobs used to fragment the molecules. Default: 1')
    parser.add_argument('--njobsLDA', type=int, default=1, help='Please specify the number of jobs used to fragment the molecules. Default: 1')
    parser.add_argument('--maxIterOpt', type=int, default=10, help='Please specify the number of iterations for the LDA optimization. Default: 10')
    parser.add_argument('--outfilePrefix', type=str, default='tm_', help='Please specify a filename to store the model.')
    parser.add_argument('--chunksize', type=int, default=1000, help='Please specify the chunksize for online training. Default: 1000')
    parser.add_argument('--lowPrec', type=bool, default=0, help='Choose a lower precision if you expect your model to be huge. Default: False')
    parser.add_argument('--ratioCmpdsMB', type=float, default=1.0, help='Choose the number of cmpds the model will be build on. Default: 1.0')
    args = parser.parse_args()

    print('---> Reading data')
    datafile = args.data
    data = pd.read_csv(datafile)
    
    data = data.sample(frac=1.0,random_state=np.random.RandomState(42))
    data.reset_index(drop=True,inplace=True)
    data.to_csv(datafile+'.shuffled',index=False)

    seed=57
    tm=chemTopicModel.ChemTopicModel(sizeSampleDataSet=args.sizeSampleDataSet, fragmentMethod=args.fragMethod, 
                                     rareThres=args.rareThres, commonThres=args.commonThres, randomState=seed, 
                                     n_jobs=args.njobsFrag, chunksize=args.chunksize)

    tm.loadData(data)

    print("---> Generating fragments")
    stime = time.time()
    tm.generateFragments()
    print("Time:", time.time()-stime)
    print("Size fragment matrix ", tm.fragM.shape)

    print("---> Fitting topic model")
    stime = time.time()    
    tm.fitTopicModel(args.numTopics, max_iter=args.maxIterOpt, nJobs=args.njobsLDA, sizeFittingDataset=args.ratioCmpdsMB)
    print("Time:", time.time()-stime)
    print("---> Transforming topic model")
    stime = time.time()    
    tm.transformDataToTopicModel(lowerPrecision=args.lowPrec)
    print("Time:", time.time()-stime)
    
    print("---> Saving topic model")
    # you need protocol 4 to save large files (> 4GB), this is only possible with pyhton version > 3.4
    with open(args.outfilePrefix+'.pkl', 'wb') as fp:
        pickle.dump(tm, fp, protocol=4)
    
    print('---> DONE. Enjoy your model!')
    
main()