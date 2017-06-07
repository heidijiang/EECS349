### **Predicting neural spike trains from two-photon calcium imaging**
 
Northwestern University, EECS349 Spring 2017 - Machine Learning <br>
Maite Azcorra-Sedano, Han Jiang, Torben Noto, Vivek Sagar <br>
torben.noto@gmail.com
 
 
#### <u>Abstract<u>
 
> Research in neuroscience usually aims to study neuronal activity, or spiking; but measuring spiking is an arduous process that can only be done one cell at a time. A current challenge is determining the spiking of neurons from recordings of features that are related to their firing, but which are naturally noisy and imprecise. Calcium imaging is a popular technique that optically measures intracellular levels of calcium from thousands of neurons simultaneously, but despite the several advantages of calcium imaging, it suffers from the drawback that the calcium levels only provide a proxy for neuronal firing. Even though there is a strong biophysical framework to explain how neuronal firing relates to calcium currents, it is not clear how we can mathematically calculate the neuronal spiking from calcium signals due to factors like limited sampling rates and dye-buffering. A precise and fast algorithm for doing this would help enormously in the understanding of neuronal functions at a larger scale, eliminating the caveat of equating calcium signaling to activity.
>
> Several computational models (deconvolution filters, Bayesian, biophysical and generalized linear models) have been proposed to predict the spike trains from calcium currents, but their estimation necessitates making several assumptions about the mechanism underlying the relationship between calcium currents and neuronal spiking. Here we implemented several supervised machine learning algorithms which do not require such assumptions, including logistic regression, gradient boosting, feedforward neural networks, and recurrent neural networks, to predict spike trains from calcium signals and their derivatives. We obtained different sensitivities, specificities and precisions for the algorithms, and particularly in some cases an appreciable amount of specificity, but none of the algorithms were able to isolate the spikes with high precision. We speculate that this might be due to the heavily skewed data, and to variability in the mapping from the calcium trace to the spike train across neurons. 
 
> <br>
> <br>
> <center><img src = "figures/abstract.png" alt="Fig. 1" class="inline" width="400"/></center><br>
> Figure 1. Upper: example time series of calcium signal and concurrent spikes.
>Lower: Sensitivity, specificity and precision of 3 different classifiers: 
>LR (logistic regression), XGB (gradient boosting) and RNN (recurrent neural nets).
 
#### <u> Final Report</u>
 
#### Introduction
The central focus of neuroscientific research is to understand how the biology, activity, and connections between neurons gives rise to cognitive processes. One of the first observations that was made about neural activity was that they transmit chemical signals to their downstream targets when they undergo a stereotyped activity pattern called spiking or firing. For this reason, many researchers aim to understand the nature of neural firing by understanding the processes surrounding it. However, measuring spiking in neurons is technically challenging and prohibitively difficult in certain model systems so many prefer to measure features of neurons that may serve as a proxy for spiking. 
	Calcium imaging is a popular technique that optically measures intracellular levels of calcium from thousands of neurons simultaneously. There is a strong framework describing the biological processes underlying the relationship between calcium levels and neural spikes but there are often discrepancies between our biophysical models and data beyond what would be expected by noise. Researchers have built probabalistic models to [predict spiking](http://www.sciencedirect.com/science/article/pii/S0896627316300733?np=y) but to our knowledge, nobody has used modern machine learning approaches to model and predict spiking from calcium imaging.
 
#### Data Acquisition and Feature Selection
We obtained a dataset of concurrent calcium and spiking recordings from the [Collaborative Research in Computational Neuroscience (CRCNS) website](https://crcns.org/data-sets/methods/cai-3/about-ret-2). This dataset contains 5 sessions of the experimentors recording neurons from different parts of the brain using different calcium indicators and under different brain states  (described in detail [here](https://crcns.org/files/data/cai-3/crcns_cai-3_data_description.pdf)). Each session contained between 5-21 neurons and each neuron yielding of the order of 30,000-80,000 time points worth of instances).
 
After examining the data, we observed that calcium traces obtained using different calcium indicators (OGB-1 vs GCaMP6s) were considerably different, as expected given the distinct dynamics of these indicators. We therefore decided that training should not be done on all data simultaneously, and instead algorithms trained on data from one indicator should be tested on data from that same indicator. For simplicity, we decided to first train and test on data from only one indicator, GCaMP6s, as it is the more commonly used of the two; but we expect that, once the optimal algorithm and parameters are set, this approach could be used for any indicator, as long as training is done on the corresponding data. 
 
This subset of the data that we used (Fig 1.) contained 781082 time points from 13 cells in the mouse retina and 9 cells in the mouse visual cortex. Spikes are relatively infrequent in neural recordings, although it varies enormously between neuronal types. Our dataset contains 24301 total spikes (3% of the total time points). This sparsity of positive examples presented a challenge for further analysis. Furthermore, while spikes are discrete events, they tend to occur in bursts called 'spike trains'. The sampling rate of the recordings (100 Hz) was such that multiple spikes were sometimes binned into single time point. However, because spike events containing more than one spike were even more sparse, we chose to binarize the spiking, as otherwise it would add an additional level of complexity to the classification problem.<br>
<img src = "figures/raw_data_example.png" alt="Fig. 2" class="inline" width="600"/><br>
Figure 2. Example calcium (yellow) and spike train (black) traces of 3 cells over time 
 
Several features were then extracted from the preprocessed calcium signals (Fig 3.). For each time point, we calculated the instantaneous calcium signal, all of the calcium signals in a range of one second in the future and past, the derivative of the calcium signal, the second derivative of the calcium signal, and a sliding window average of calcium activity over 11 increasingly broad windows. Additionally, we included labels about the brain region that the neurons were recorded from, and the cognitive state of the mouse.
<img src = "figures/feature_matrix.png" alt="Fig. 3" class="inline" width="600"/><br>
Figure 3. Example of extracted features from calcium signal over time
 
#### Classification Methods<br>
We implemented 4 supervised learning algorithms to generate predictions of spike patterns from concurrent calcium signal. At a first pass, we implemented logistic regression and feedforward neural nets to examine baseline performance in cases when the classifier does not naturally take into account the temporal nature of the signal. Next, we implemented a gradient boosted decision trees algorithm, (XGBoost) an algorithm that constructs a regularized ensemble of decision trees. This framework specializes in constructing robust learners using high numbers of weakly informative features. We chose this because it can potentially integrate dozens of aspects of calcium dynamics that may predict or underlie spiking. Also [this package](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions) has been used as part of over half of winning Kaggle competition entries across many feature sets so we expected that it would work well here as well. Finally, we implemented a recurrent neural network which applies LSTM to potentially capture temporal dependencies that affect spiking. We hoped that the LSTM will be able to capture the spike trains accurately since the probability of observing a spike depends upon the prior history of the calcium trace. We had also implemented a feedforward neural network with attributes similar to the ones used in gradient boosting.
 
##### Logistic Regression: <br>
 
First, we applied a logistic regression model, which estimates the probabilistic relationship between independent input and dependent output using a logistic function in order to classify spike trains. We empirically chose a regularization parameter between L1 and L2 by calculating precision of the training model over all lambdas between 0 and 1, with step size .05, and choosing the lambda with the highest precision. Then we used the output model from the best lambda to test on the independently drawn test dataset. Code for logistic regression can be found [here](https://github.com/heidijiang/EECS349/logreg).
 
##### Gradient Boosting: <br>
We built a gradient boosted decision tree classifier using XGBoost and TensorFlow. This classifier builds many weakly accurate, shallow decision trees and combines their predictions into an ensemble that can be fairly accurate. We used a binary, sigmoidal loss function and optimized several important hyperparameters (L1 regularization, L2 regularization, gamma - an entropy-like metric for splitting decisions, and learning step size) of the model to improve performance. All code for training and optimizing gradient boosted trees can be found [here](https://github.com/torbenator/calcium_spikes).
 
##### Recurrent and Feedforward Neural Network: <br>
The idea is that the history of the calcium activity affects the probability of observing a spike. Keeping this in mind, we implemented an LSTM (using TensorFlow) that takes the previous history of two seconds as the input. The RNN has 64 units, that takes inputs in 10 chunks, each chunk containing 20 time points. The data was fed in batches of size 128 and the network was run for 10 epochs for each of the 5 folds in cross validation. We implemented a dropout of 0.8 for regularization. The value of the dropout was chosen to provide the best possible value of recall in the test data. We used cross entropy with logits as the cost function and penalized the network for using a large number of predicted spikes. 
 
We had also implemented a feedforward neural network. Due to the skewness in the data and the inappropriate cost function, it acted like a ZeroR. The codes for the neural nets can be found [here](https://github.com/viveksgr/RNN-for-calcium-imaging/tree/master/Scripts). 
 
#### Results
 
##### Logistic Regression
Logistic Regression obtained a total accuracy of 55.7%, with a true positive/recall rate of 16.54%, true negative rate of 57.3%, and a precision of 1.52%. A chi square test revealed total accuracy to be highly significant relative to chance of 50% χ² (1,155654) = 1030.6, p < .0001. Though this result is highly significant, the logistic regression suffered from an extremely high false positive rate, as can be seen in Figure X.<br>
<img src = "figures/calcium_logreg.png" alt="Fig. X" class="inline" width="400"/><br>
##### Gradient Boosting
<img src = "figures/xgboost_accuracy_curve.png" alt="Fig. X" class="inline" width="600"/>
##### RNN
<img src = "figures/image.png" alt="Fig. X" class="inline" width="600"/><br>
The total accuracy for RNN was 78.1%. The recall rate (percentage of spikes correctly predicted) was 48.5%. The classification accuracy was significantly different from chance classification (χ² test, p-value < 0.00001). However, RNN had a high false positive rate, evident from the following zoomed in plot.
<img src = "figures/RNN_image3.png" alt="Fig. X" class="inline" width="600"/><br>
 
#### Discussion
 
Investigating the feature importance scores for the gradient boosted trees provides novel insights to which features contributed to accurate predictions. Perhaps not so surprisingly, the feature that provided on average the most information gain and that was split on the most, was the brain region that the neuron was recorded from (f60). This supports the idea that calcium dynamics are not identical from one neuron to the next, but may be somewhat specific to cell type and cellular environment.
More surprisingly, simultaneous calcium level (f0) was not a particularly useful feature for predicting spiking. Concurrent calcium values alone correlated with raw spiking with a Pearson's r value of 0.10, and the boosted trees algorithm had a .16 Pearson's r value with spiking in the test set. The calcium currents preceding and following a given time point were relied upon in this model (f17-57). Even calcium at times 200 ms before spikes (f53) was a particularly strong contributor to prediction accuracy.
 
<center>
<img src ="figures/feature_importance_scores.png" alt="Fig. X" class="inline" width="400"/>
 
We also have an understanding of why the recurrent neural network did not yield satisfactory results. The RNN puts spikes all across the time series, but the actual number of predicted spikes is small compared to the overall length of the time series (not visible in the plot). This ensures that the accuracy is not abysmal. Also, since we accepted a spike predicted one sample (10ms) before or after the actual spike as a correct prediction in this model, the percentage of predicted spikes is also not small. Thus, the model does optimize accuracy and recall, but it does so by putting a number of spikes all over the time series – which is not desirable. 
 
#### Future Work 
Using sliding window derivative.
Explicitly involving the derivatives and their history in the RNN.
More sensible cost function in the neural nets, optimized to predict higher percentage of spike match, and not just overall accuracy.
 
