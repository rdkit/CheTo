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
from rdkit.Chem import AllChem

import copy
from collections import defaultdict
import hashlib

def getMorganEnvironment(mol, bitInfo, fp=None, minRad=0):
    """

    >>> m = Chem.MolFromSmiles('CC(O)C')
    >>> bi = {}
    >>> fp = AllChem.GetMorganFingerprintAsBitVect(m,2,2048,bitInfo=bi)
    >>> getMorganEnvironment(m,bi)
    defaultdict(<class 'list'>, {1: [[]], 227: [[1]], 283: [[0], [2]], 709: [[0, 1, 2]], 807: [[]], 1057: [[], []]})
    >>> getMorganEnvironment(m,bi,minRad=1)
    defaultdict(<class 'list'>, {227: [[1]], 283: [[0], [2]], 709: [[0, 1, 2]]})
    >>> list(fp.GetOnBits())
    [1, 227, 283, 709, 807, 1057]
    >>> getMorganEnvironment(m,bi,minRad=1,fp=fp)
    defaultdict(<class 'list'>, {227: [[1]], 283: [[0], [2]], 709: [[0, 1, 2]]})
    >>> list(fp.GetOnBits())
    [227, 283, 709]

    """
    bitPaths=defaultdict(list)
    for bit,info in bitInfo.items():
        for atomID,radius in info:
            if radius < minRad:
                if fp != None:
                    fp[bit]=0
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
            bitPaths[bit].append(list(env))
    return bitPaths

def _includeRingMembership(s, n, noRingAtom=False):
    r=';R]'
    if noRingAtom:
        r=';R0]'
    d="]"
    return r.join([d.join(s.split(d)[:n]),d.join(s.split(d)[n:])])

def _includeDegree(s, n, d):
    r=';D'+str(d)+']'
    d="]"
    return r.join([d.join(s.split(d)[:n]),d.join(s.split(d)[n:])])

def writePropsToSmiles(mol,smi,order):
    """

    >>> writePropsToSmiles(Chem.MolFromSmiles('Cc1ncccc1'),'[cH]:[n]:[c]-[CH3]',(3,2,1,0))
    '[cH;R;D2]:[n;R;D2]:[c;R;D3]-[CH3;R0;D1]'

    """
    finalsmi = copy.deepcopy(smi)
    for i,a in enumerate(order,1):
        atom = mol.GetAtomWithIdx(a)
        if not atom.GetAtomicNum():
            continue
        finalsmi = _includeRingMembership(finalsmi, i, noRingAtom = not atom.IsInRing())
        finalsmi = _includeDegree(finalsmi, i, atom.GetDegree())
    return finalsmi

def getSubstructSmi(mol,env,propsToSmiles=True):
    """

    >>> getSubstructSmi(Chem.MolFromSmiles('Cc1ncccc1'),((0,1,2)))
    '[cH;R;D2]:[n;R;D2]:[c;R;D3]-[CH3;R0;D1]'

    """
    atomsToUse=set()
    if not len(env):
        return ''
    for b in env:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    # no isomeric smiles since we don't include that in the fingerprints
    smi = Chem.MolFragmentToSmiles(mol,atomsToUse,isomericSmiles=False,
                                   bondsToUse=env,allHsExplicit=True, allBondsExplicit=True)
    if propsToSmiles:
        order = eval(mol.GetProp("_smilesAtomOutputOrder"))
        smi = writePropsToSmiles(mol,smi,order)
    return smi

def generateAtomInvariant(mol):
    """

    >>> generateAtomInvariant(Chem.MolFromSmiles("Cc1ncccc1"))
    [346999948, 3963180082, 3525326240, 2490398925, 2490398925, 2490398925, 2490398925]

    """
    num_atoms = mol.GetNumAtoms()
    invariants = [0]*num_atoms
    for i,a in enumerate(mol.GetAtoms()):
        descriptors=[]
        descriptors.append(a.GetAtomicNum())
        descriptors.append(a.GetTotalDegree())
        descriptors.append(a.GetTotalNumHs())
        descriptors.append(a.IsInRing())
        descriptors.append(a.GetIsAromatic())
        invariants[i]=int(hashlib.sha256(str(descriptors).encode('utf-8')).hexdigest(),16)& 0xffffffff
    return invariants


#------------------------------------
#
#  doctest boilerplate
#
def _test():
    import doctest, sys
    return doctest.testmod(sys.modules["__main__"])


if __name__ == '__main__':
    import sys
    failed, tried = _test()
    sys.exit(failed)
