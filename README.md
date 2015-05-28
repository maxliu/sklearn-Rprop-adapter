# sklearn-Rprop-adpter

##Motivation
I like to use sklearn for my data mining works especially the functins of Pipeline and parameter searchj. However, to date, I could not find the neuron network I needed such as BP. pybrain is a excelent library for neuron network algorithms, there is no need to invent the wheels again. 

##basic idea
It is easy to make a customerized clssifier fit into sklearn. 
```
class RPClassifier(BaseEstimator, ClassifierMixin):
```
There are three stes pybrain to build and train a network.
1. create dataset.
2. define topology of network.
3. create trainer.

Add those three steps into "fit" method will do the job.
```
    def fit(self, X, y):
        """ build a network from training set (X, y).
```

```
        ds = SupervisedDataSet(self.in_size, self.out_size)

        ds.setField('input', X)
        ds.setField('target', y_train)

        self.net = buildNetwork(self.in_size,
                                self.h_size, self.out_size, bias=True)
        trainer = RPropMinusTrainer(self.net, dataset=ds)
```

##References
http://pybrain.org/docs/index.html

http://scikit-learn.org/stable/developers/

