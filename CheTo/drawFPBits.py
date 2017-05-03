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
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

import numpy as np
import re

def _drawFPBit(smi,bitPath,molSize=(150,150),kekulize=True,baseRad=0.05,svg=True,**kwargs):
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)

    # get the atoms for highlighting
    atomsToUse=[]
    for b in bitPath:
        atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
    atomsToUse = list(set(atomsToUse))
    
    #  enlarge the environment by one further bond
    enlargedEnv=[]
    for atom in atomsToUse:
        a =  mol.GetAtomWithIdx(atom)
        for b in a.GetBonds():
            bidx=b.GetIdx()
            if bidx not in bitPath:
                enlargedEnv.append(bidx)
    enlargedEnv = list(set(enlargedEnv))
    enlargedEnv+=bitPath
    
    # set the coordinates of the submol based on the coordinates of the original molecule
    amap={}
    submol = Chem.PathToSubmol(mol,enlargedEnv,atomMap=amap)
    rdDepictor.Compute2DCoords(submol)
    conf = submol.GetConformer(0)
    confOri =  mol.GetConformer(0)
    for i1,i2 in amap.items():
        conf.SetAtomPosition(i2,confOri.GetAtomPosition(i1))
        
    envSubmol=[]
    for i1,i2 in amap.items():
        for b in bitPath:
            beginAtom=amap[mol.GetBondWithIdx(b).GetBeginAtomIdx()]
            endAtom=amap[mol.GetBondWithIdx(b).GetEndAtomIdx()]
            envSubmol.append(submol.GetBondBetweenAtoms(beginAtom,endAtom).GetIdx())
            
    # Drawing
    if svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
        
    
    drawopt=drawer.drawOptions()
    drawopt.continuousHighlight=False
    
    # color all atoms of the submol in gray which are not part of the bit
    # highlight atoms which are in rings
    color = (.9,.9,.9)
    atomcolors,bondcolors={},{}
    highlightAtoms,highlightBonds=[],[]
    
    for aidx in amap.keys():
        if aidx in atomsToUse:
            if mol.GetAtomWithIdx(aidx).GetIsAromatic():
                atomcolors[amap[aidx]]=(0.9,0.9,0.2)
                highlightAtoms.append(amap[aidx])
            elif mol.GetAtomWithIdx(aidx).IsInRing():
                atomcolors[amap[aidx]]=(0.8,0.8,0.8)
                highlightAtoms.append(amap[aidx])
        else:
            drawopt.atomLabels[amap[aidx]]='*'
            submol.GetAtomWithIdx(amap[aidx]).SetAtomicNum(1)
    for bid in submol.GetBonds():
        bidx=bid.GetIdx()
        if bidx not in envSubmol:
            bondcolors[bidx]=color
            highlightBonds.append(bidx)

    drawer.DrawMolecule(submol,highlightAtoms=highlightAtoms,highlightAtomColors=atomcolors,
                    highlightBonds=highlightBonds,highlightBondColors=bondcolors,
                    **kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def drawFPBitPNG(smi,bitPath,molSize=(150,150),kekulize=True,baseRad=0.05,**kwargs):   
    return _drawFPBit(smi,bitPath,molSize=molSize,kekulize=kekulize,baseRad=baseRad, svg=False,**kwargs)
    
def drawFPBit(smi,bitPath,molSize=(150,150),kekulize=True,baseRad=0.05,**kwargs):   
    svg = _drawFPBit(smi,bitPath,molSize=molSize,kekulize=kekulize,baseRad=baseRad,**kwargs)
    return svg.replace('svg:','')

def _drawBricsFrag(smi,molSize=(150,150),kekulize=True,baseRad=0.05,svg=True,**kwargs):
    
    # delete smarts specific syntax from the pattern
    smi = re.sub(r"\;R\d?\;D\d+", "", smi)
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    mc = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=kekulize)
            
    # Drawing
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    if not svg:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def drawBricsFragPNG(smi,molSize=(150,150),kekulize=True,baseRad=0.05,**kwargs):
    return _drawBricsFrag(smi,molSize=molSize,kekulize=kekulize,baseRad=baseRad,svg=False,**kwargs)

def drawBricsFrag(smi,molSize=(150,150),kekulize=True,baseRad=0.05,**kwargs):
    svg = _drawBricsFrag(smi,molSize=molSize,kekulize=kekulize,baseRad=baseRad,**kwargs)
    return svg.replace('svg:','')
