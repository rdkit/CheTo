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
# Created by Nadine Schneider, September 2016


from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import operator

# continous Tanimoto coefficient
def _calcContTanimoto(x,y):
    m=np.array([x,y])
    minsum = m.min(0).sum()
    maxsum = m.max(0).sum()
    if maxsum > 0:
        return minsum/maxsum
    else:
        return 0

# continous Tanimoto coefficient matrix for a count-based FP matrix
def calcContTanimotoDistMatrix(fragMatrix):
    from scipy.spatial.distance import squareform,pdist
    dists = pdist(fragMatrix, lambda u, v: _calcContTanimoto(u,v))
    return squareform(dists)

# calculate recall and precision for a topic model based on a given label
def generateStatistics(topicModel, idLabelToUse=1):

    label_topics=defaultdict(list)
    topics_label=defaultdict(list)
    numMolsPerLabel=Counter(topicModel.moldata['label_'+str(idLabelToUse)].tolist())
    numDocs, numTopics = topicModel.documentTopicProbabilities.shape
    for i in range(0, numDocs):
        label = topicModel.moldata['label_'+str(idLabelToUse)][i]
        maxTopic=np.argmax(topicModel.documentTopicProbabilities[i])
        label_topics[label].append(maxTopic)
        topics_label[maxTopic].append(label)
    label_topics2=defaultdict(dict)
    for tid,topics in label_topics.items():
        label_topics2[tid]=Counter(topics)
    topics_label2=defaultdict(dict)
    for topicid,tid in topics_label.items():
        topics_label2[topicid]=Counter(tid)

    data=[]
    for label,topics in label_topics2.items():
        tsorted = sorted(topics.items(), key=operator.itemgetter(1),reverse=True)
        maxTopic = tsorted[0][0]
        numMolsMaxTopic = tsorted[0][1]
        numMols = numMolsPerLabel[label]
        precisionMT = numMolsMaxTopic/float(sum(topics_label2[maxTopic].values()))
        recallMT = numMolsMaxTopic/float(numMols)
        F1 = 2 * (precisionMT * recallMT) / (precisionMT + recallMT)
        data.append([label, numMols, len(topics.keys()), int(maxTopic), recallMT, precisionMT, F1])
    data=pd.DataFrame(data, columns=['label','# mols','# topics','main topic ID','recall in main topic',\
                                     'precision in main topic', 'F1'])
    data = data.sort_values(['main topic ID'])
    overall=['Median']
    overall.extend(data[['# mols','# topics']].median().values.tolist())
    overall.append('-')
    overall.extend(data[['recall in main topic','precision in main topic', 'F1']].median().values.tolist())
    data.loc[len(data)] = overall
    return data
