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


from rdkit import Chem
from rdkit.Chem import rdqueries
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import display,HTML,SVG

from collections import defaultdict
import operator
import seaborn as sns
import numpy as np

from ChemTopicModel import utilsDrawing, drawFPBits, chemTopicModel

# get the topic fragment probabilites per atom for highlighting 
def _getAtomWeights(mol, molID, topicID, topicModel):

    weights=[0]*mol.GetNumAtoms()
    # ignore "wildcard atoms" in BRICS fragments
    q = rdqueries.AtomNumEqualsQueryAtom(0)
    # get all fragments of a certain molecule
    _,aBits=chemTopicModel._generateFPs(mol, topicModel.fragmentMethod)
    fp=topicModel.moldata.loc[molID,'fps']
    probs = topicModel.getTopicFragmentProbabilities()
    nTopics, nFrags = probs.shape
    # use the max probability of a fragment associated with a certain topic 
    # to normalize the fragment weights
    maxWeightTopic = max(probs[topicID])
    r = 0.0
    # calculate the weight of an atom concerning a certain topic
    for bit in fp.keys():
        try:
            idxBit = bit
            if topicModel.fragmentMethod in ['Morgan', 'RDK']:
                idxBit = topicModel.fragIdx[bit]
        except:
            continue
        try:
            r = probs[topicID,idxBit]
        except:
            continue
        if r <= 1./nFrags:
            continue
        # Morgan/RDK fingerprints
        if topicModel.fragmentMethod in ['Morgan', 'RDK'] and bit in aBits:
            paths = aBits[bit]
            for p in paths:
                for b in p:
                    bond = mol.GetBondWithIdx(b)
                    # for overlapping fragments take the highest weight for the atom 
                    weights[bond.GetBeginAtomIdx()]=max(r,weights[bond.GetBeginAtomIdx()])
                    weights[bond.GetEndAtomIdx()]=max(r,weights[bond.GetEndAtomIdx()])
        elif topicModel.fragmentMethod.startswith('Brics'):
            # BRICS fragments
            submol = Chem.MolFromSmarts(topicModel.vocabulary[idxBit])
            ignoreWildcards = [i.GetIdx() for i in list(submol.GetAtomsMatchingQuery(q))]
            matches = mol.GetSubstructMatches(submol)
            for m in matches:
                for n,atomidx in enumerate(m):
                    if n in ignoreWildcards:
                        continue
                    # for overlapping fragments take the highest weight for the atom, this not happen for BRICS though :) 
                    weights[atomidx]=max(r,weights[atomidx])
    atomWeights = np.array(weights)
    return atomWeights,maxWeightTopic

# hightlight a topic in a molecule
def drawTopicWeightsMolecule(mol, molID, topicID, topicModel, molSize=(450,200), kekulize=True,\
                             baseRad=0.1, color=(.9,.9,.9), fontSize=0.9):

    # get the atom weights
    atomWeights,maxWeightTopic=_getAtomWeights(mol, molID, topicID, topicModel)
    atRads={}
    atColors={}      
    
    # color the atoms and set their highlight radius according to their weight
    if np.sum(atomWeights) > 0:
        for at,score in enumerate(atomWeights):
            atColors[at]=color
            atRads[at]=max(atomWeights[at]/maxWeightTopic, 0.0) * baseRad

            if atRads[at] > 0 and atRads[at] < 0.2:
                atRads[at] = 0.2

    mc = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=kekulize)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.SetFontSize(fontSize)
    drawer.DrawMolecule(mc,highlightAtoms=atColors.keys(),
                        highlightAtomColors=atColors,highlightAtomRadii=atRads,
                        highlightBonds=[])
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')

# generates all svgs of molecules belonging to a certain topic and highlights this topic within the molecule
def generateMoleculeSVGsbyTopicIdx(topicModel, topicIdx, idsLabelToShow=[0], topicProbThreshold = 0.5, baseRad=0.5,\
                                    molSize=(250,150),color=(.0,.0, 1.),maxMols=100, fontSize=0.9):
    svgs=[]
    namesSVGs=[]
    numDocs, numTopics = topicModel.documentTopicProbabilities.shape
    
    if topicIdx >= numTopics:
        return "Topic not found"
    molset = topicModel.documentTopicProbabilities[:,topicIdx].argsort()[::-1][:maxMols]
    for doc in molset:
        if topicModel.documentTopicProbabilities[doc,topicIdx] >= topicProbThreshold:
            data = topicModel.moldata.iloc[doc]
            smi = data['smiles']
            name = ''
            for idx in idsLabelToShow:
                name += str(data['label_'+str(idx)])
                name += ' | '
            mol = Chem.MolFromSmiles(smi)
            topicProb = topicModel.documentTopicProbabilities[doc,topicIdx]   
            svg = drawTopicWeightsMolecule(mol, doc, topicIdx, topicModel, molSize=molSize, baseRad=baseRad, color=color, fontSize=fontSize)
            svgs.append(svg)
            maxTopicID= np.argmax(topicModel.documentTopicProbabilities[doc,:])
            namesSVGs.append(str(name)+"(p="+str(round(topicProb,2))+")")
    if not len(svgs):
        #print('No molecules can be drawn')
        return [],[]
    
    return svgs, namesSVGs

# generates all svgs of molecules having a certain label attached and highlights most probable topic within the molecule
def generateMoleculeSVGsbyLabel(topicModel, label, idLabelToMatch=0, baseRad=0.5, molSize=(250,150),maxMols=100):
    
    data = topicModel.moldata.loc[topicModel.moldata['label_'+str(idLabelToMatch)] == label]
    
    if not len(data):
        return "Label not found"
    
    svgs=[]
    namesSVGs=[]
    numDocs, numTopics = topicModel.documentTopicProbabilities.shape
    colors = sns.husl_palette(numTopics, s=.6) 

    topicIdx = np.argmax(topicModel.documentTopicProbabilities[data.index,:],axis=1) 
    topicProb = np.amax(topicModel.documentTopicProbabilities[data.index,:],axis=1)
    topicdata = list(zip(data.index, topicIdx, topicProb))
    topicdata_sorted = sorted(topicdata, key=operator.itemgetter(2), reverse=True)
    
    for idx,tIdx,tProb in topicdata_sorted[:maxMols]:
        mol = Chem.MolFromSmiles(data['smiles'][idx])
        color = tuple(colors[tIdx]) 
        svg = drawTopicWeightsMolecule(mol, idx, tIdx, topicModel, molSize=molSize, baseRad=baseRad, color=color)
        svgs.append(svg)
        namesSVGs.append(str("Topic "+str(tIdx)+" | (p="+str(round(tProb,2))+")"))

    return svgs, namesSVGs

### draw mols by label, highlight different topics

# draws molecules of a certain label in a html table and highlights the most probable topic
def drawMolsByLabel(topicModel, label, idLabelToMatch=0, baseRad=0.5, molSize=(250,150),\
                    numRowsShown=3, tableHeader='', maxMols=100):
        
    result = generateMoleculeSVGsbyLabel(topicModel, label, idLabelToMatch=idLabelToMatch,baseRad=baseRad,\
                                                              molSize=molSize, maxMols=maxMols)
    if len(result)  == 1:
        print(result)
        return
        
    svgs, namesSVGs = result
    finalsvgs = []
    for svg in svgs:
        # make the svg scalable
        finalsvgs.append(svg.replace('<svg','<svg preserveAspectRatio="xMinYMin meet" viewBox="0 0 '+str(molSize[0])\
                                     +' '+str(molSize[1])+'"'))
    
    return display(HTML(utilsDrawing.drawSVGsToHTMLGrid(finalsvgs[:maxMols],cssTableName='overviewTab',tableHeader='Molecules of '+str(label),
                                                 namesSVGs=namesSVGs[:maxMols], size=molSize, numRowsShown=numRowsShown, numColumns=4)))

# produces a svg grid of the molecules of a certain label and highlights the most probable topic
def generateSVGGridMolsByLabel(topicModel, label, idLabelToMatch=0, baseRad=0.5, molSize=(250,150),svgsPerRow=4):
    
    result = generateMoleculeSVGsbyLabel(topicModel, label, idLabelToMatch=idLabelToMatch, baseRad=baseRad, molSize=molSize)
    
    if len(result)  == 1:
        print(result)
        return
        
    svgs, namesSVGs, labelName = result
    
    svgGrid = utilsDrawing.SvgsToGrid(svgs, namesSVGs, svgsPerRow=svgsPerRow, molSize=molSize)
    
    return svgGrid

#### draw mols by topic

# draws molecules belonging to a certain topic in a html table and highlights this topic within the molecules
def drawMolsByTopic(topicModel, topicIdx, idsLabelToShow=[0], topicProbThreshold = 0.5, baseRad=0.5, molSize=(250,150),\
                    numRowsShown=3, color=(.0,.0, 1.), maxMols=100, fontSize=0.9):
    result = generateMoleculeSVGsbyTopicIdx(topicModel, topicIdx, idsLabelToShow=idsLabelToShow, \
                                             topicProbThreshold = topicProbThreshold, baseRad=baseRad,\
                                             molSize=molSize,color=color, maxMols=maxMols,fontSize=fontSize)
    if len(result)  == 1:
        print(result)
        return
        
    svgs, namesSVGs = result
    finalsvgs = []
    for svg in svgs:
        # make the svg scalable
        finalsvgs.append(svg.replace('<svg','<svg preserveAspectRatio="xMinYMin meet" viewBox="0 0 '+str(molSize[0])\
                                     +' '+str(molSize[1])+'"'))
 
    tableHeader = 'Molecules in topic '+str(topicIdx)+' (sorted by decending probability)'
    
    return display(HTML(utilsDrawing.drawSVGsToHTMLGrid(finalsvgs[:maxMols],cssTableName='overviewTab',tableHeader=tableHeader,\
                                                 namesSVGs=namesSVGs[:maxMols], size=molSize, numRowsShown=numRowsShown, numColumns=4)))

# produces a svg grid of the molecules belonging to a certain topic and highlights this topic within the molecules
def generateSVGGridMolsbyTopic(topicModel, topicIdx, idsLabelToShow=[0], topicProbThreshold = 0.5, baseRad=0.5, \
                                 molSize=(250,150), svgsPerRow=4, color=(1.,1.,1.)):
    
    result = generateMoleculeSVGsbyTopicIdx(topicModel, topicIdx, idsLabelToShow=idsLabelToShow, \
                                             topicProbThreshold = topicProbThreshold, baseRad=baseRad,\
                                             molSize=molSize,color=color)
    if len(result)  == 1:
        print(result)
        return
        
    svgs, namesSVGs = result
    
    svgGrid = utilsDrawing.SvgsToGrid(svgs, namesSVGs, svgsPerRow=svgsPerRow, molSize=molSize)
    
    return svgGrid

####### Fragments ##########################

# generates svgs of the fragments related to a certain topic
def generateTopicRelatedFragmentSVGs(topicModel, topicIdx, n_top_frags=10, molSize=(100,100),\
                                     svg=True, prior=-1.0, fontSize=0.9):
    svgs=[]
    probs = topicModel.getTopicFragmentProbabilities()
    numTopics, numFragments = probs.shape
    if prior < 0:
        prior = 1./numFragments
    # only consider the top n fragments
    for i in probs[topicIdx,:].argsort()[::-1][:n_top_frags]:
        if probs[topicIdx,i] > prior:
            bit = topicModel.vocabulary[i]
            # draw the bits using the templates
            if topicModel.fragmentMethod in ['Morgan', 'RDK']:
                templMol = topicModel.fragmentTemplates.loc[topicModel.fragmentTemplates['bitIdx'] == bit]['templateMol'].item()
                pathTemplMol = topicModel.fragmentTemplates.loc[topicModel.fragmentTemplates['bitIdx'] == bit]['bitPathTemplateMol'].item()
                if svg:
                    svgs.append(drawFPBits.drawFPBit(templMol,pathTemplMol,molSize=molSize, fontSize=fontSize))
                else:
                    svgs.append(drawFPBits.drawFPBitPNG(templMol,pathTemplMol,molSize=molSize))
            else:
                if svg:
                    svgs.append(drawFPBits.drawBricsFrag(bit,molSize=molSize))
                else:
                    svgs.append(drawFPBits.drawBricsFragPNG(bit,molSize=molSize))                   
    return svgs

# draw the svgs of the fragments related to a certain topic in a html table
def drawFragmentsbyTopic(topicModel, topicIdx, n_top_frags=10, numRowsShown=4, cssTableName='fragTab', \
                         prior=-1.0, numColumns=4, tableHeader='',fontSize=0.9):    
    
    scores = topicModel.getTopicFragmentProbabilities()
    numTopics, numFragments = scores.shape
    if prior < 0:
        prior = 1./numFragments
    svgs=generateTopicRelatedFragmentSVGs(topicModel, topicIdx, n_top_frags=n_top_frags, prior=prior,fontSize=fontSize)
    namesSVGs = list(map(lambda x: "Score %.2f" % x, \
                    filter(lambda y: y > prior, sorted(scores[topicIdx,:], reverse=True)[:n_top_frags])))
    if tableHeader == '':
        tableHeader = "Topic "+str(topicIdx)
    return display(HTML(utilsDrawing.drawSVGsToHTMLGrid(svgs,tableHeader=tableHeader,cssTableName=cssTableName,\
                                                 namesSVGs=namesSVGs,size=(120,100),numRowsShown=numRowsShown,\
                                                 numColumns=numColumns)))

