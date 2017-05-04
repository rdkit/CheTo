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


import numpy as np
import pandas as pd
import copy
import re
from rdkit.Chem import PandasTools
from IPython.display import SVG

# generate an HTML table of the svg images to visulize them nicely in the Jupyter notebook 
PandasTools.RenderImagesInAllDataFrames(images=True)
def drawSVGsToHTMLGrid(svgs, cssTableName='default', tableHeader='', namesSVGs=[], size=(150,150), numColumns=4, numRowsShown=2, noHeader=False):
    rows=[]
    names=copy.deepcopy(namesSVGs)
    rows = [SVG(i).data for i in svgs]
    d=int(len(rows)/numColumns)
    x=len(rows)%numColumns
    if x > 0:
        rows+=['']*(numColumns-x)
        d+=1
        if len(names)>0:
            names+=['']*(numColumns-x)
    rows=np.array(rows).reshape(d,numColumns)
    finalRows=[]
    if len(names)>0:
        names = np.array(names).reshape(d,numColumns)
        for r,n in zip(rows,names):
            finalRows.append(r)
            finalRows.append(n)
        d*=2
    else:
        finalRows=rows

    headerRemove = int(max(numColumns,d))
    df=pd.DataFrame(finalRows)

    style = '<style>\n'
    style += 'table.'+cssTableName+' { border-collapse: collapse; border: none;}\n'
    style += 'table.'+cssTableName+' tr, table.'+cssTableName+' th, table.'+cssTableName+' td { border: none;}\n'
    style += 'table.'+cssTableName+' td { width: '+str(size[0])+'px; max-height: '+str(size[1])+'px; background-color: white; text-align:center;}\n'
    if noHeader:
        style += 'table.'+cssTableName+' th {  width: '+str(size[0])+'px; max-height: 0px; background-color: white;}\n'
    else:
        style += 'table.'+cssTableName+' th { color: #ffffff; background-color: #848482; text-align: center;}\n'
        style += '.headline { color: #ffffff; background-color: #848482; text-align: center; font-size: 18px;\
        font-weight: bold; padding: 10px 10px 10px 10px}\n'
    style += '</style>\n'
    if not noHeader:
        style += '<div class="headline">'+str(tableHeader)+'</div>\n'
    style += '<div id="" style="overflow-y:scroll; overflow-x:hidden; max-height:'+str(size[1]*numRowsShown+size[1]/2)+'px; background-color: white; border:1px solid grey">\n'
    dfhtml=style+df.to_html()+'\n</div>\n'
    dfhtml=dfhtml.replace('class="dataframe"','class="'+cssTableName+'"')
    dfhtml=dfhtml.replace('<th></th>','')
    for i in range(0,headerRemove):
        dfhtml=dfhtml.replace('<th>'+str(i)+'</th>','')
    return dfhtml

# build an svg grid image to print
def SvgsToGrid(svgs, labels, svgsPerRow=4,molSize=(250,150),fontSize=12):
    
    matcher = re.compile(r'^(<.*>\n)(<rect .*</rect>\n)(.*)</svg>',re.DOTALL) 
    hdr='' 
    ftr='</svg>' 
    rect='' 
    nRows = len(svgs)//svgsPerRow 
    if len(svgs)%svgsPerRow : nRows+=1 
    blocks = ['']*(nRows*svgsPerRow)
    labelSizeDist = fontSize*5
    fullSize=(svgsPerRow*(molSize[0]+molSize[0]/10.0),nRows*(molSize[1]+labelSizeDist))
    print(fullSize)

    count=0
    for svg,name in zip(svgs,labels):
        h,r,b = matcher.match(svg).groups()
        if not hdr: 
            hdr = h.replace("width='"+str(molSize[0])+"px'","width='%dpx'"%fullSize[0])
            hdr = hdr.replace("height='"+str(molSize[1])+"px'","height='%dpx'"%fullSize[1])
        if not rect: 
            rect = r
        legend = '<text font-family="sans-serif" font-size="'+str(fontSize)+'px" text-anchor="middle" fill="black">\n'
        legend += '<tspan x="'+str(molSize[0]/2.)+'" y="'+str(molSize[1]+fontSize*2)+'">'+name.split('|')[0]+'</tspan>\n'
        if len(name.split('|')) > 1:
            legend += '<tspan x="'+str(molSize[0]/2.)+'" y="'+str(molSize[1]+fontSize*3.5)+'">'+name.split('|')[1]+'</tspan>\n'
        legend += '</text>\n'
        blocks[count] = b + legend
        count+=1

    for i,elem in enumerate(blocks): 
        row = i//svgsPerRow 
        col = i%svgsPerRow 
        elem = rect+elem 
        blocks[i] = '<g transform="translate(%d,%d)" >%s</g>'%(col*(molSize[0]+molSize[0]/10.0),row*(molSize[1]+labelSizeDist),elem) 
    res = hdr + '\n'.join(blocks)+ftr 
    return res 
