from NN import Model
import re
import Pipeline


#data = Pipeline.Pipeline("training.csv")



print("Train: 0")
print("Predict: 1")
print(" ")
choice = input()
if int(choice) == 0:
    nw = Model(20, 24)
    lr = input("Enter a learning rate: ")
    nw.train("training2.csv", float(lr))
else:
    nw = Model(20, 24)
    


