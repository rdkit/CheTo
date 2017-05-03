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
# Created by Nadine Schneider, December 2016


from ipywidgets import *
from IPython.display import display, clear_output
from ChemTopicModel import drawTopicModel, chemTopicModel

# allows choosing of topic colors
from matplotlib.colors import hex2color, rgb2hex, cnames
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import numpy as np
from collections import defaultdict

# some nice interactive bokeh plots
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import show, figure
from bokeh.io import output_notebook

# outputy bokeh plots within the notebook
output_notebook()
# use seaborn style
sns.set()
# for encoding the png images
import base64

def to_base64(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

# main GUI
def TopicModel():   
    
    def buildModel(sender):
        clear_output()
        showTopicButton.disabled = True
        showMoleculesButton.disabled = True
        saveAsButton.disabled = True
        saveAsButton2.disabled = True
        statsButton.disabled = True
        statsButton2.disabled = True
        progressBar = widgets.FloatProgress(min=0, max=100, width='300px', margin='10px 5px 10px 10px')
        labelProgressBar.value = 'Loading data'
        display(progressBar)
        
        filename = dataSetSelector.value
        if filename == '':
            print('No data set specified, please check your input.')
            progressBar.close()
            return
        try:
            data = pd.read_csv(filename)
            labelProgressBar.value = 'Generating fragments (may take several minutes for larger data sets)'
            progressBar.value +=33
        except:
            progressBar.value +=100
            labelProgressBar.value = 'Reading data failed'
            print('Invalid data file, please check your file.')
            progressBar.close()
            return
        try:
            starttime =  time.time()
            topicModel=chemTopicModel.ChemTopicModel(fragmentMethod=fragmentmethodSelector.value, rareThres=rareFilterSelector.value, commonThres=commonFilterSelector.value)
            topicModel.loadData(data)
            topicModel.generateFragments()
            labelProgressBar.value = 'Building the model (may take several minutes for larger data sets and many topics)'
            progressBar.value +=33
            topicModel.buildTopicModel(numTopicSelector.value)
            finaltime = time.time() - starttime
            progressBar.value +=34
            labelProgressBar.value = 'Finished model successfully in %.3f sec'%finaltime

            # Update parameters,dropdown options etc.
            labelSelector.options = topicModel.oriLabelNames
            labelSelector2.options = topicModel.oriLabelNames
            labelSelector2a.options = topicModel.oriLabelNames
            numDocs, numTopics = topicModel.documentTopicProbabilities.shape
            labelID = topicModel.oriLabelNames.index(labelSelector2.value)
            labelSelector3.options = sorted(list(set(topicModel.moldata['label_'+str(labelID)])))
            labelSelector3a.options = sorted(list(set(topicModel.moldata['label_'+str(labelID)])))
            topicSelector.max=numTopics
            params['labels'] = topicModel.oriLabelNames
            params['topicModel'] = topicModel
            params['colors'] = sns.husl_palette(numTopics, s=.6)
            params['numTopics'] = numTopics
            showTopicButton.disabled = False
            showMoleculesButton.disabled = False
            saveAsButton.disabled = False
            saveAsButton2.disabled = False
            statsButton.disabled = False
            statsButton2.disabled = False
            progressBar.close()
        except:
            progressBar.value +=100
            labelProgressBar.value = 'Model building failed'
            print('Topic model could not be built.')
            return
        
    _tooltipstr="""
                <div>
                    <div>
                        <span style="font-size: 17px;">Topic $index</span><br>
                        <span style="font-size: 12px;">Top 3 fragments</span>
                    </div>
                    <div style="display: flex">
                        <figure style="text-align: center">
                            <img
                            src="@desc1" height="20" 
                            style="float: left; margin: 0px 5px 5px 0px;"
                            border="2"
                            ></img>
                            <figcaption> [@desc4] </figcaption>
                        </figure>
                        <figure style="text-align: center">
                            <img
                            src="@desc2" height="20" 
                            style="float: center; margin: 0px 5px 5px 0px; "
                            border="2"
                            ></img>
                            <figcaption> [@desc5] </figcaption>
                        </figure>
                        <figure style="text-align: center">
                            <img
                            src="@desc3" height="20" 
                            style="float: right; margin: 0px 5px 5px 0px; "
                            border="2"
                            ></img>
                            <figcaption> [@desc6] </figcaption>
                        </figure>
                    </div>
                </div>
                      """
    
    def _getToolTipImages(topicModel, numTopics, nTopFrags):
        
        tmp=[]
        name=[]
        scores=topicModel.getTopicFragmentProbabilities()
        for i in range(0,numTopics):
            try:
                imgs = drawTopicModel.generateTopicRelatedFragmentSVGs(topicModel, i, n_top_frags=nTopFrags,molSize=(100,80),svg=False)
                t = [to_base64(i) for i in imgs]
                if len(t) < nTopFrags:
                    for j in range(len(t),nTopFrags):
                        t.append('')
                tmp.append(t)
            except:
                pass
            names = list(map(lambda x: "Score %.2f" % x, filter(lambda y: y > 0.0, sorted(scores[i,:], reverse=True)[:nTopFrags])))
            name.append(names)
        name = np.array(name)
        
        edges = np.arange(numTopics+1)
        if len(tmp)  == 0:
            tmp=[['','','']]*numTopics
        tmp = np.array(tmp)
        return name,tmp,edges
    
    def calcOverallStatistics(sender):
        clear_output()
        labelProgressBar.value=''
        topicModel =  params['topicModel']
        numDocs, numTopics = topicModel.documentTopicProbabilities.shape
        topicDocStats=[0]*numTopics
        for doc in range(0,numDocs):
            topicDocStats[np.argmax(topicModel.documentTopicProbabilities[doc,:])]+=1
        topicDocStatsNorm=np.array(topicDocStats).astype(float)/numDocs
               
        name,tmp,edges = _getToolTipImages(topicModel, numTopics, 3)

        source = ColumnDataSource( data = dict( y = topicDocStatsNorm, l = edges[ :-1 ], r = edges[ 1: ], desc1 = tmp[:,0], \
                                              desc2 = tmp[:,1], desc3 = tmp[:,2], desc4 = name[:,0], \
                                              desc5 = name[:,1], desc6 = name[:,2]))

        hover=HoverTool()
        hover.tooltips= _tooltipstr     
               
        p = figure(width=800, height=400, tools=[hover], toolbar_location=None, title="Overall topic distribution")

        p.quad( top = 'y', bottom = 0, left = 'l', right = 'r', 
                 fill_color = "#036564", line_color = "#033649", source = source ) 
        
        p.xaxis.axis_label = "Topics"
        p.yaxis.axis_label = "% molecules per topic"
        p.xaxis.minor_tick_line_color = None
        show(p)

    def calcSubsetStatistics(sender):
        clear_output()
        labelProgressBar.value=''
        topicModel =  params['topicModel']
        label = labelSelector3a.value
        labelID = params['labels'].index(labelSelector2a.value)
        numDocs, numTopics = topicModel.documentTopicProbabilities.shape
        
        data = topicModel.moldata.loc[topicModel.moldata['label_'+str(labelID)] == label]
        topicProfile  = np.zeros((numTopics,), dtype=np.int)
        
        for idx in data.index:
            topicProfile = np.sum([topicProfile, topicModel.documentTopicProbabilities[idx]], axis=0)  
        topicProfileNorm=np.array(topicProfile).astype(float)/data.shape[0]

        name,tmp,edges = _getToolTipImages(topicModel, numTopics, 3)

        source = ColumnDataSource( data = dict( y = topicProfileNorm, l = edges[ :-1 ], r = edges[ 1: ], desc1 = tmp[:,0], \
                                              desc2 = tmp[:,1], desc3 = tmp[:,2], desc4 = name[:,0], \
                                              desc5 = name[:,1], desc6 = name[:,2]))

        hover=HoverTool()
        hover.tooltips= _tooltipstr 
               
        p = figure(width=800, height=400, tools=[hover],toolbar_location=None, title="Topic profile for "+str(label))

        p.quad( top = 'y', bottom = 0, left = 'l', right = 'r', 
                 fill_color = "#036564", line_color = "#033649", source = source ) 

        p.xaxis.axis_label = "Topics"
        p.yaxis.axis_label = "Mean probability of topics"
        p.xaxis.minor_tick_line_color = None
        show(p)
        

    def showTopic(sender):
        topicModel =  params['topicModel']
        clear_output()
        labelProgressBar.value=''
        topicID = topicSelector.value
        labelID = params['labels'].index(labelSelector.value)
        if chooseColor.value:
            c = colorSelector.value
            if not c.startswith('#'):
                c =  cnames[c]
            hex_color = c
            rgb_color = hex2color(hex_color)
        else:
            rgb_color = tuple(params['colors'][topicSelector.value])
            colorSelector.value = rgb2hex(rgb_color)

        temp=None
        if topicID == '' or labelID == '':
            print("Please check your input")
        else:
            drawTopicModel.drawFragmentsbyTopic(topicModel, topicID, n_top_frags=20, numRowsShown=1.2,\
                                         numColumns=8, tableHeader='Top fragments of topic '+str(topicID))

            drawTopicModel.drawMolsByTopic(topicModel, topicID, idsLabelToShow=[labelID], topicProbThreshold = 0.1, baseRad=0.9,\
                                    numRowsShown=3, color=rgb_color)
                        
    def showMolecules(sender):
        topicModel =  params['topicModel']
        clear_output()
        labelProgressBar.value=''
        label = labelSelector3.value
        labelID = params['labels'].index(labelSelector2.value)

        if label == '' or labelID == '':
            print("Please check your input")
        else:
            drawTopicModel.drawMolsByLabel(topicModel, label, idLabelToMatch=labelID, baseRad=0.9, \
                                    molSize=(250,150), numRowsShown=3)
            

    def saveTopicAs(sender):
        topicModel =  params['topicModel']
        topicID = topicSelector.value
        labelID = params['labels'].index(labelSelector.value)
        path = filePath.value

        if chooseColor.value:
            c = colorSelector.value
            if not c.startswith('#'):
                c =  cnames[c]
            hex_color = c
            rgb_color = hex2color(hex_color)
        else:
            rgb_color = tuple(params['colors'][topicSelector.value])
            colorSelector.value = rgb2hex(rgb_color)

        temp=None
        if topicID == '' or labelID == '':
            print("Please check your input")
        else:
            svgGrid = drawTopicModel.generateSVGGridMolsbyTopic(topicModel, 0, idLabelToShow=labelID, topicProbThreshold = 0.1,
                                                         baseRad=0.9, color=rgb_color)
            with open(path+'.svg','w') as out:
                out.write(svgGrid)
            print("Saved topic image to: "+os.getcwd()+'/'+path+'.svg')

                
    def saveMolSetAs(sender):
        topicModel =  params['topicModel']
        if topicModel == None:
            print('No topic model available, please build a valid model first.')
            return

        path = filePath2.value
        label = labelSelector3.value
        labelID = params['labels'].index(labelSelector2.value)

        if label == '' or labelID == '':
            print("Please check your input")
        else:
            svgGrid = drawTopicModel.generateSVGGridMolsByLabel(topicModel, label, idLabelToMatch=labelID, baseRad=0.9)
            
            with open(path+'.svg','w') as out:
                out.write(svgGrid)
            print("Saved molecule set image to: "+os.getcwd()+'/'+path+'.svg')
                

    def getMolLabels(labelName):
        topicModel =  params['topicModel']
        try:
            labelID = params['labels'].index(labelName)
            return list(set(topicModel.moldata['label_'+str(labelID)]))
        except:
            return []
    
    def selectMolSet(sender):
        labelSelector3.options = sorted(getMolLabels(labelSelector2.value))
        
    def selectMolSeta(sender):
        labelSelector3a.options = sorted(getMolLabels(labelSelector2a.value))
        
    def topicColor(sender):
        rgb_color = tuple(params['colors'][topicSelector.value])
        colorSelector.value = rgb2hex(rgb_color)          

        
    # init values
    params=dict([('labels',[]),('numTopics',50),('colors',sns.husl_palette(20, s=.6)),('topicModel',None),('rareThres',0.001),('commonThres',0.1)])    
          
    labelProgressBar = widgets.Label(value='')

        
    ########### Model building widgets
    # widgets
    dataSetSelector = widgets.Text(description='Data set:',value='data/datasetA.csv', width='450px', margin='10px 5px 10px 10px')
    numTopicSelector = widgets.IntText(description='Number of topics', width='200px', value=params['numTopics'],\
                                       margin='10px 5px 10px 10px')
    rareFilterSelector = widgets.BoundedFloatText(min=0,max=1.0,description='Threshold rare fragments', width='200px', value=params['rareThres'], margin='10px 5px 10px 10px')
    commonFilterSelector = widgets.BoundedFloatText(min=0,max=1.0,description='Threshold common fragments', width='200px', value=params['commonThres'], margin='10px 5px 10px 10px')
    fragmentmethodSelector = widgets.Dropdown(options=['Morgan', 'RDK', 'Brics'], description='Fragment method:',\
                                              width='200px',margin='10px 5px 10px 10px')  
    doItButton = widgets.Button(description="Build model", button_style='danger', width='300px', margin='10px 5px 10px 10px')
    
    # actions
    labels = doItButton.on_click(buildModel)
    
    # layout widgets
    set1 = widgets.HBox()
    set1.children = [dataSetSelector]
    set2 = widgets.HBox()
    set2.children = [numTopicSelector, fragmentmethodSelector]
    set2a = widgets.HBox()
    set2a.children = [rareFilterSelector, commonFilterSelector]
    set3 = widgets.HBox()
    set3.children = [doItButton]
    finalLayout = widgets.VBox()
    finalLayout.children = [set1, set2, set2a, set3]

    ########### Model statistics widget
    statsButton = widgets.Button(description="Show overall topic distribution", disabled=True, button_style='danger',\
                                 width='300px', margin='10px 5px 10px 10px')
    labelSelector2a = widgets.Dropdown(options=params['labels'], description='Label:', width='300px', margin='10px 5px 10px 10px')
    init = labelSelector2a.value
    labelSelector3a = widgets.Dropdown(options=getMolLabels(init), description='Molecule set:', width='300px', margin='10px 5px 10px 10px')  
    statsButton2 = widgets.Button(description="Show topic profile by label", disabled=True, button_style='danger',\
                                  width='300px', margin='10px 5px 10px 10px')
    
    # actions
    statsButton.on_click(calcOverallStatistics)
    statsButton2.on_click(calcSubsetStatistics)
    labelSelector2a.observe(selectMolSeta)
    
    # layout
    statsLayout = widgets.HBox()
    statsLayout.children = [statsButton]
    statsLayout2 = widgets.HBox()
    statsLayout2.children = [labelSelector2a, labelSelector3a, statsButton2]
    finalLayoutStats= widgets.VBox()
    finalLayoutStats.children = [statsLayout, statsLayout2]
    
    ########### Model exploration widgets
    # choose topic tab
    labelSelector = widgets.Dropdown(options=params['labels'], description='Label to show:', width='300px', margin='10px 5px 10px 10px')  
    topicSelector = widgets.BoundedIntText(description="Topic to show", min=0, max=params['numTopics']-1, width='200px',\
                                           margin='10px 5px 10px 10px')
    lableChooseColor = widgets.Label(value='Define topic color',margin='10px 5px 10px 10px')
    chooseColor = widgets.Checkbox(value=False,margin='10px 5px 10px 10px')
    showTopicButton = widgets.Button(description="Show the topic", button_style='danger',disabled=True,\
                                     width='200px', margin='10px 5px 10px 10px')
    # choose molecules tab
    labelSelector2 = widgets.Dropdown(options=params['labels'], description='Label:', width='300px', margin='10px 5px 10px 10px')
    init = labelSelector2.value
    labelSelector3 = widgets.Dropdown(options=getMolLabels(init), description='Molecule set:', width='300px', margin='10px 5px 10px 10px')  
    showMoleculesButton = widgets.Button(description="Show the molecules", button_style='danger',disabled=True, width='200px',\
                                         margin='10px 5px 10px 10px')
    # choose color tab
    colorSelector = widgets.ColorPicker(concise=False, description='Topic highlight color', value='#e0e3e4',width='200px', \
                                        margin='10px 5px 10px 10px')
    # save as tab
    filePath = widgets.Text(description="Save file as:", width='450px', margin='10px 5px 10px 10px')
    saveAsButton = widgets.Button(description="Save topic image", button_style='info',disabled=True,width='200px',\
                                  margin='10px 5px 10px 10px')
    filePath2 = widgets.Text(description="Save file as:", width='450px', margin='10px 5px 10px 10px')
    saveAsButton2 = widgets.Button(description="Save molecule set image", button_style='info',disabled=True,width='200px', \
                                   margin='10px 5px 10px 10px')
                
    # actions             
    showTopicButton.on_click(showTopic)
    saveAsButton.on_click(saveTopicAs)
    saveAsButton2.on_click(saveMolSetAs)
    showMoleculesButton.on_click(showMolecules)
    labelSelector2.observe(selectMolSet)
    
    # layout widgets
    tab1 = widgets.HBox()
    tab1.children = [topicSelector, labelSelector, lableChooseColor, chooseColor, showTopicButton]
    tab2 = widgets.HBox()
    tab2.children = [labelSelector2, labelSelector3, showMoleculesButton]
    tab3 = widgets.HBox()
    tab3.children = [colorSelector]
    tab4a = widgets.HBox()
    tab4a.children = [filePath, saveAsButton]
    tab4b = widgets.HBox()
    tab4b.children = [filePath2, saveAsButton2]
    tab4 = widgets.VBox()
    tab4.children = [tab4a, tab4b]

    children = [tab1, tab2, tab3, tab4]
    tabs = widgets.Tab(children=children)
    tabs.set_title(0,'Topic to explore')
    tabs.set_title(1,'Molecule set to explore')
    tabs.set_title(2,'Choose color')
    tabs.set_title(3,'Save images as')
    
    accordion = widgets.Accordion(children=[finalLayout, finalLayoutStats, tabs])
    accordion.set_title(0, 'Build Topic model')
    accordion.set_title(1, 'Statistics Topic model')
    accordion.set_title(2, 'Explore Topic model')
    display(accordion)
    display(labelProgressBar)