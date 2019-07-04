#
#  Copyright (c) 2016, Novartis Institutes for BioMedical Research Inc.
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
# Created by Nadine Schneider, April 2019


from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from ChemTopicModel import chemTopicModel


def rankInterestingTopics(TM, minMaxProb=0.6, topXfrags=10):
    #ratio of high prob mols
    numMols, numTopics = TM.documentTopicProbabilities.shape
    fracHighProbMols=[]
    relTopicSize=[]
    absTopicSize=[]
    for i in range(numTopics):
        #find all molecules that have maximum topic probability for topic i
        subM = TM.documentTopicProbabilities[np.where(TM.documentTopicProbabilities.argmax(axis=1) == i)]
        numMolsMaxProb,_ = subM.shape
        if numMolsMaxProb > 0:
            relTopicSize.append(numMolsMaxProb/numMols)
            #get the fraction of molecules with a probability of at least minMaxProb
            numHighProbMols = np.where(subM.max(axis=1) >= minMaxProb)[0].shape[0]
            fracHighProbMols.append(numHighProbMols/numMolsMaxProb)
            absTopicSize.append(numMolsMaxProb)
        else:
            relTopicSize.append(0.0)
            absTopicSize.append(0.0)
            fracHighProbMols.append(0.0)
    
    fragsprob=TM.getTopicFragmentProbabilities()
    probTopXFrags = [sum(sorted(fragsprob[k], reverse=True)[:topXfrags]) for k in range(numTopics)]
    numFrags=[len(Counter(i.astype('float16')))-1 for i in fragsprob]
    minfrag = min(numFrags)
    maxfrag = max(numFrags)
    normNumFrags=[(x-minfrag)/(maxfrag-minfrag) for x in numFrags]
    
    tmpResult = list(zip(list(range(numTopics)),fracHighProbMols,absTopicSize,relTopicSize,normNumFrags,probTopXFrags))   
    result = pd.DataFrame(tmpResult, columns=['Topic Idx', 'fraction high prob. mols', 
                                              'abs. topic size', 'rel. topic size', 
                                              'rel. num. relevant frags', 'sum prob. top 5 frags'])
    return result