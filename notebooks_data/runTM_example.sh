# default learning_rate and default learning_offset
python runTopicModel.py --data chembl23_mols.csv.shuffled --rareThres 0.0005 --njobsFrag 10 --numTopics 100 --sizeSampleDataSet 0.1 --outfilePrefix tm_chembl23_100 --maxIterOpt 50 --chunksize 85000 --lowPrec 1 > chembl23_100.log &
