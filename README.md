# maxentstudy
Efficiency is not concered in this project.<br/>
This project is to study maximum entropy classifier on my own.<br/>
I refered to following github to surprass overflow error on exponential function<br/>
https://github.com/hprovenza/MaxEnt-classifier<br/>

Note :
1. Unigram and Bigram are used as a feature function<br/>
2. In the learning step, first term of the likelihood is 'empirical count' of feature function and second term is 'expected count' of feature function.<br/>
For example, suppose sentence "bad bad experience restaurant" has unique three feature functions<br/>
We can formulate 2d vector [sentiment x features in corpus] and sum up empirical count<br/>
pos-[0, 0, 0]<br/>
neg-[2, 1, 1]<br/>
where, first column denotes "bad" and rest tokens corresponds to each following columns in order<br/>
For second term, supppose same example sentence is recognized as negative with 70% probabilty<br/>
We can formulate same 2d vector as follow for expected count<br/>
pos-[0, 0, 0]<br/>
neg-[0.7+0.7, 0.7, 0.7]<br/>


(pelican) Kwanghoons-MacBook-Pro:MyPractice kwanghoonan$ python maxent.py <br/>
5    20187<br/>
4    12876<br/>
3     6527<br/>
1     6263<br/>
2     4147<br/>
Name: stars, dtype: int64<br/>
Train # : 45000<br/>
Test # : 5000<br/>
total feature numbers :  1497696<br/>
batch steps :  9<br/>
Iteration : 0  Gradient : 327608.9599161494  Neg likelihood -4491.772501073527  ACC : 0.5496<br/>
Iteration : 1  Gradient : 293796.19116865494  Neg likelihood -4101.960967595788  ACC : 0.576<br/>
Iteration : 2  Gradient : 275133.8030917825  Neg likelihood -3934.512490013268  ACC : 0.586<br/>
Iteration : 3  Gradient : 265233.79090478015  Neg likelihood -3927.748926361367  ACC : 0.5974<br/>
Iteration : 4  Gradient : 255854.5050274092  Neg likelihood -3829.354464276137  ACC : 0.6062<br/>
Iteration : 5  Gradient : 250786.25502524382  Neg likelihood -3869.4517431731847  ACC : 0.6094<br/>
Iteration : 6  Gradient : 244294.44230927844  Neg likelihood -3975.723697428084  ACC : 0.611<br/>
Iteration : 7  Gradient : 229760.20564771473  Neg likelihood -3882.3239995867466  ACC : 0.6192<br/>
Iteration : 8  Gradient : 231860.20637058772  Neg likelihood -3899.164706149007  ACC : 0.6192<br/>
Iteration : 9  Gradient : 233250.09569945032  Neg likelihood -3888.1097378659915  ACC : 0.6228<br/>
Iteration : 10  Gradient : 221399.7404561218  Neg likelihood -3930.0455557952355  ACC : 0.6242<br/>
Iteration : 11  Gradient : 228422.08284046422  Neg likelihood -4223.580680043418  ACC : 0.6258<br/>
Iteration : 12  Gradient : 220909.77532204858  Neg likelihood -3871.7739250846785  ACC : 0.6272<br/>

My hand-written note deriving relationship between maximum entropy clasifier and maximum entropy principle is based on
https://www.quora.com/What-is-the-relationship-between-Log-Linear-model-MaxEnt-model-and-Logistic-Regression
http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week6.pdf
