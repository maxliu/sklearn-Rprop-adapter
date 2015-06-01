# sklearn-Rprop-adpter

##Motivation
I like to use sklearn for my data mining works especially the functions of Pipeline and parameter search. However, to date, I could not find the neuron network I needed such as BP. pybrain is an excellent library for neuron network algorithms, there is no need to invent the wheels again. 

##Basic idea
It is easy to make a customerized clssifier fit into sklearn. 
```
class RPClassifier(BaseEstimator, ClassifierMixin):
```
There are three steps for pybrain to build and train a network.


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
##Examples
###With Pipeline (see ./examples/rprop_pipeline.py)
```
    bpc = RPClassifier(h_size=h_size, epo=epo)

    anova_filter = SelectKBest(f_regression, k=4)

    anova_bp = Pipeline([
        ('anava', anova_filter),
        ('bpc', bpc)
    ])

    anova_bp.fit(x_train, y_train)

    p = anova_bp.predict_proba(x_train)

    p_c = anova_bp.predict(x_train)
```
###With parameter search (see ./examples/rprop_search.py)
```
    bpc = RPClassifier(h_size=h_size, epo=epo)

    print "starting grid search ... "
    param_dist = {"h_size": sp_randint(2, 10)}
    clf = bpc
    n_iter_search = 2
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       verbose=1
                                       )
```
##References
http://pybrain.org/docs/index.html

http://scikit-learn.org/stable/developers/
