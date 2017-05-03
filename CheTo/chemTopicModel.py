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
# Created by Nadine Schneider, June 2016


import random
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals.joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries, BRICS

from ChemTopicModel import utilsFP

#### FRAGMENT GENERATION ####################

def _prepBRICSSmiles(m):
    smi = Chem.MolToSmiles(m,isomericSmiles=True, allHsExplicit=True, allBondsExplicit=True)
    # delete the connection ids
    smi = re.sub(r"\[\d+\*\]", "[*]", smi)
    order = eval(m.GetProp("_smilesAtomOutputOrder"))
    # make the smiles more descriptive, add properties
    return utilsFP.writePropsToSmiles(m,smi,order)

def _generateFPs(mol,fragmentMethod='Morgan'):
    aBits={}
    fp=None
    # circular Morgan fingerprint fragmentation, we use a simple invariant than ususal here
    if fragmentMethod=='Morgan':
        tmp={}
        fp = AllChem.GetMorganFingerprint(mol,radius=2,invariants=utilsFP.generateAtomInvariant(mol),bitInfo=tmp)
        aBits = utilsFP.getMorganEnvironment(mol, tmp, fp=fp, minRad=2)
        fp = fp.GetNonzeroElements()
    # path-based RDKit fingerprint fragmentation
    elif fragmentMethod=='RDK':
        fp = AllChem.UnfoldedRDKFingerprintCountBased(mol,maxPath=5,minPath=3,bitInfo=aBits)
        fp = fp.GetNonzeroElements()
    # get the final BRICS fragmentation (= smallest possible BRICS fragments of a molecule)
    elif fragmentMethod=='Brics':
        fragMol=BRICS.BreakBRICSBonds(mol)
        propSmi = _prepBRICSSmiles(fragMol)
        fp=Counter(propSmi.split('.'))
    else:
        print("Unknown fragment method")
    return fp, aBits

# this function is not part of the class due to parallelisation
# generate the fragments of a molecule, return a map with moleculeID and fragment dict
def _generateMolFrags(datachunk, vocabulary, fragmentMethod, fragIdx=None):
    if fragIdx is None and fragmentMethod == 'Brics':
        return
    result={}
    for idx, smi in datachunk:
        mol = Chem.MolFromSmiles(str(smi))
        if mol == None:
            continue
        fp,_=_generateFPs(mol,fragmentMethod=fragmentMethod)
        if fp is None:
            continue
        tmp={}
        for k,v in fp.items():
            if k not in vocabulary:
                continue
            # save memory: for BRICS use index instead of long complicated SMILES
            if fragmentMethod == 'Brics':
                tmp[fragIdx[k]]=v
            else:
                tmp[k]=v
        result[idx]=tmp
    return result

########### chemical topic modeling class ###################
class ChemTopicModel:
      
    # initialisation chemical topic model
    def __init__(self, fragmentMethod = 'Morgan', randomState=42, sizeSampleDataSet=0.1, rareThres=0.001, 
                 commonThres=0.1, verbose=0, n_jobs=1, chunksize=1000, learningMethod='batch'):
        self.fragmentMethod = fragmentMethod
        self.seed = randomState
        self.sizeSampleDataSet = sizeSampleDataSet
        self.rareThres = rareThres
        self.commonThres = commonThres
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.learningMethod = learningMethod    
    
    # generate the fragments used for the model, exclude rare and common fragments depending on a threshold
    def _generateFragmentVocabulary(self,molSample):
        fps=defaultdict(int)
        # collect fragments from a sample of the dataset 
        for smi in molSample:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            fp,_=_generateFPs(mol,fragmentMethod=self.fragmentMethod)
            if fp is None:
                continue
            for bit in fp.keys():
                fps[bit]+=1
        # filter rare and common fragments
        fragOcc = np.array(list(fps.values()))
        normFragOcc = fragOcc/float(len(molSample))
        ids = normFragOcc > self.commonThres
        normFragOcc[ids] = 0
        ids = normFragOcc < self.rareThres
        normFragOcc[ids] = 0
        keys = list(fps.keys())
        self.vocabulary = sorted(n for n,i in zip(keys,normFragOcc) if i != 0)
        self.fragIdx=dict((i,j) for j,i in enumerate(self.vocabulary))
        if self.verbose:
            print('Created vocabulary, size: {0}, used sample size: {1}'.format(len(self.vocabulary),len(molSample)))
    
    # generate the fragment templates important for the visualisation of the topics later
    def _generateFragmentTemplates(self,molSample):
        fragTemplateDict=defaultdict(list)
        voc=set(self.vocabulary)
        if not len(self.vocabulary):
            print('Please generate your vocabulary first')
            return
        sizeVocabulary=len(self.vocabulary)
        for n,smi in enumerate(molSample):
            mol = Chem.MolFromSmiles(str(smi))
            if mol == None:
                continue
            fp,aBits=_generateFPs(mol,fragmentMethod=self.fragmentMethod)
            if fp is None:
                continue                
            for k,v in fp.items():
                if k not in voc or k in fragTemplateDict:
                    continue
                # save memory: for brics use index instead of long complicated smarts
                if self.fragmentMethod in ['Brics','BricsAll']:
                    fragTemplateDict[self.fragIdx[k]]=['', []]
                else:
                    fragTemplateDict[k]=[smi, aBits[k][0]]
            if len(fragTemplateDict) == sizeVocabulary:
                break
        tmp = [[k,v[0],v[1]] for k,v in fragTemplateDict.items()]
        self.fragmentTemplates = pd.DataFrame(tmp,columns=['bitIdx','templateMol','bitPathTemplateMol'])
        if self.verbose:
            print('Created fragment templates', self.fragmentTemplates.shape)
    
    # generate fragments for the whole dataset
    def _generateFragments(self):
        voc=set(self.vocabulary)
        fpsdict = dict([(idx,{}) for idx in self.moldata.index])
        nrows = self.moldata.shape[0]
        counter = 0
        with Parallel(n_jobs=self.n_jobs,verbose=self.verbose) as parallel:
            while counter < nrows:
                nextChunk = min(counter+(self.n_jobs*self.chunksize),nrows)
                result = parallel(delayed(_generateMolFrags)(mollist, voc,
                                                    self.fragmentMethod, 
                                                    self.fragIdx)
                                   for mollist in self._produceDataChunks(counter,nextChunk,self.chunksize))
                for r in result:
                    counter+=len(r)
                    fpsdict.update(r)            
        self.moldata['fps'] = np.array(sorted(fpsdict.items()))[:,1]                
    
    # construct the molecule-fragment matrix as input for the LDA algorithm 
    def _generateFragmentMatrix(self):
        fragM=[]
        vsize=len(self.vocabulary)
        for n,fps in enumerate(self.moldata['fps']):
            # we only use 8 bit integers for the counts to save memory
            t=np.zeros((vsize,),dtype=np.uint8)
            for k,v in fps.items():
                idx = k
                if self.fragmentMethod in ['Morgan', 'RDK']:
                    idx = self.fragIdx[k]
                if v > 255:
                    print("WARNING: too many fragments of type {0} in molecule {1}".format(k,self.moldata['smiles'][len(fragM)]))
                    t[idx]=255
                else:
                    t[idx]=v
            fragM.append(t)
        self.fragM = np.array(fragM)
        
    # helper functions for parallelisation    
    def _produceDataChunks(self,start,end,chunksize):
        for start in range(start,end,chunksize):
            end=min(self.moldata.shape[0],start+chunksize)
            yield list(zip(self.moldata[start:end].index, self.moldata[start:end]['smiles']))   
                
    def _generateMatrixChunks(self, start,end,chunksize=10000):
        for start in range(start,end,chunksize):
            end=min(self.fragM.shape[0],start+chunksize)
            yield self.fragM[start:end,:], start

    ############# main functions #####################################S

    # load the data (molecule table in SMILES format (required) and optionally some lables for the molecules)
    def loadData(self, inputDataFrame):
        self.moldata = inputDataFrame
        oriLabelNames = list(self.moldata.columns)
        self.oriLabelNames = oriLabelNames[1:]       
        self.moldata.rename(columns=dict(zip(oriLabelNames, ['smiles']+['label_'+str(i) for i in range(len(oriLabelNames)-1)])),
                            inplace=True)
        
    def generateFragments(self):
        # set a fixed seed due to order dependence of the LDA method --> the same data should get the same results
        sample = self.moldata.sample(frac=self.sizeSampleDataSet,random_state=np.random.RandomState(42))
        self._generateFragmentVocabulary(sample['smiles'])
        self._generateFragmentTemplates(sample['smiles'])
        self._generateFragments()
        self._generateFragmentMatrix()
        
    # it is better use these functions instead of buildTopicModel if the dataset is larger   
    def fitTopicModel(self, numTopics, max_iter=100, **kwargs):

        self.lda = LatentDirichletAllocation(n_topics=numTopics,learning_method=self.learningMethod,random_state=self.seed,
                                             n_jobs=1, max_iter=max_iter, batch_size=self.chunksize, **kwargs)
        if self.fragM.shape[0] > self.chunksize:
            # fit the model in chunks
            self.lda.learning_method = 'online'
            self.lda.fit(self.fragM)
        else:
            self.lda.fit(self.fragM)

    def transformDataToTopicModel(self):
        
        try:
            self.lda
        except:
            raise ValueError('No topic model is available')

        if self.fragM.shape[0] > self.chunksize:
            # after fitting transform the data to our model
            for chunk in self._generateMatrixChunks(0,self.fragM.shape[0],chunksize=self.chunksize):
                resultLDA = self.lda.transform(chunk[0])
                # here using a 16bit float instead of the 64bit float would save memory and might be enough precision. Test that later!!
                if chunk[1] > 0:
                    self.documentTopicProbabilities = np.concatenate((self.documentTopicProbabilities,
                                                                 resultLDA/resultLDA.sum(axis=1,keepdims=1)), axis=0)
                else:
                    self.documentTopicProbabilities = resultLDA/resultLDA.sum(axis=1,keepdims=1)
        else:
            resultLDA = self.lda.transform(self.fragM)
            self.documentTopicProbabilities = resultLDA/resultLDA.sum(axis=1,keepdims=1)
            
    
    # use this if the dataset is small- to medium-sized   
    def buildTopicModel(self, numTopics, max_iter=100, **kwargs):
        
        self.fitTopicModel(numTopics, max_iter=max_iter, **kwargs)
        self.transformDataToTopicModel()
            
        
    def getTopicFragmentProbabilities(self):
                
        try:
            self.lda
        except:
            raise ValueError('No topic model is available')
        return self.lda.components_/self.lda.components_.sum(axis=1,keepdims=1)
        
