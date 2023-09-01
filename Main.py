from NN4 import Model4
from NN3 import Model
import numpy as np




#
# data = Pipeline.Pipeline("training.csv")


#nw = Model(20,10,10)
#nw.SGD("shuffled_training.csv",alpha=.8)
#nw.ADAM("shuffled_training.csv", alpha=.08, batch_size=1)


nw = Model(14,7,7)
#nw.ADAM("shuffledRevisedData.csv",alpha=.05, split = .8, batch_size=1)
nw.SGD("shuffledRevisedData.csv",alpha=.035, split = .8, batch_size=1)



#nw.mini_batch("shuffled_training.csv", float(.05), 10)
#else:
#    nw = Model(20, 24)
    


