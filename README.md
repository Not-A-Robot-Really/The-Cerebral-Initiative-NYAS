# The-Cerebral-Initiative-NYAS
Offline simulation using OpenNeuro data. Analyzing a P300 speller, combining an LLM-based Phrase Prediction system (Llama 3.2) along with Adaptive Bayesian Early Stopping and measuring its Information Transfer Rate (ITR) and initial baseline accuracy relative to the baseline P300 speller.

# Hypothesis
*In an offline simulation on OpenNeuro data, a P300 speller combining an LLM-based Phrase Prediction system (Llama 3.2) along with Adaptive Bayesian Early Stopping will increase Information Transfer Rate (ITR) by at least 10% relative to the baseline P300 speller while maintaining at least 90% of the initial baseline accuracy.*

# Research Plan:

We will run four simulations. We will run each 11 times, one for each participant (P_01 to P_11) in the dataset.

Control group: First, we will conduct an offline simulation with a baseline P300 speller as our control case. The speller will be generic, modeled after the speller used in the dataset.
Next, we will conduct an offline simulation with an Adaptive Bayesian Early Stopping system with the P300 speller.
Next, we will conduct an offline simulation with an LLM-based (Llama 3.2 1B with 1.23B parameters***) Phrase prediction system with the P300 speller.
Next, we will combine both the LLM prediction system and the Adaptive Bayesian Early Stopping system into one with the P300 speller.

In addition to testing both Bayesian Early Stopping and phrase prediction as stated in the hypothesis, we also chose to test them individually to figure out how much they affect the accuracy and ITR individually. 

Selections not present in the recordings are simulated using each participant's measured classifier accuracy and score distributions from their test data.

All of the trials above will use an P300 BCI open data set from OpenNeuro by Nand Chandravadia et al. (dataset in bibliography). The data catalogs a collection of EEG signals of 11 patients during P300 testing. We will then calculate the ITR from our trials. We chose this dataset because it fit our goal of using P300 spelling boards to find EEG data.

We will be using the LDA classifier (1 per participant)  to categorize each epoch and help recognize the P300 wave from the other noise. We will take the raw EEG dataset from the OpenNeuro dataset, filter it, cut out the epochs using a python script, and put it into the LDA classifier, where it will output a score for each individual epoch. Disregarding Bayesian stopping, the scores for all epochs in all cycles (repetitions) will then be combined into a single letter. The classifier score will also update our LLM phrase predictor. We chose LDA over other alternatives because it is the standard classifier in most clinical P300 studies & trials. LDA is also compatible with Bayesian early stopping and works well in small datasets.

We will compare the ITR of each trial. A trial is considered to have a sufficient ITR if the ITR increased by at least 10% relative to the baseline P300 speller. The ITR will be calculated using the Wolpaw ITR formula from Shannon's channel capacity theory.

We will also measure the accuracy (correct characters divided by total number of characters) using the same dataset. Any test with an accuracy less than 90% than the accuracy of the control case will be considered failed, regardless of ITR

***We chose to use the model Llama 3.2 1B because it is a practical, small, yet modern model that is both fast and works on a phone.

After the experiment, we will use PsychoPy as the frontend to do a live demonstration of the P300 spellers as a visual interface demo (substituting real EEGs with a keyboard input).

Some of our limitations:

Lose the realism of actual participants
Cannot model frustration with LLM incompetencies
No real-time feedback from EEG
Small sample size (n = 11)
PsychoPy’s interface is using keyboard input, which mimics but does not perfectly replicate EEG from BCIs.

Our final solution will be viable because we will be able to utilize the results of our research & experiments to determine the best combination of features to optimize for speed while retaining accuracy. We use a pre-existing classifier and dataset for reliability, maximizing viability.

# Bibliography:

7135-the-cerebral-initiative
Joana Almeida Ferreira, Nuno Pinto, Tiago Maricoto, Maria Vaz Pato,
Relationship between event-related potentials and cognitive dysfunction in multiple sclerosis: A systematic review,
Clinical Neurophysiology,
Volume 163,
2024,
Pages 174-184,
ISSN 1388-2457,
https://doi.org/10.1016/j.clinph.2024.04.024.
(https://www.sciencedirect.com/science/article/pii/S1388245724001494)
Abstract: Objective
This systematic review aimed to evaluate if event-related potentials (ERPs) can be a relevant tool for cognitive dysfunction diagnosis in Multiple Sclerosis (MS).
Methods
Four databases were consulted (PubMed, Embase, Scielo, and Web of Science). The included studies should include adults with clear MS diagnoses, independently of having cognitive complaints, and all should have been submitted to ERPs (P300, N400 or mismatch negativity (MMN)). The main outcomes evaluated were ERPs’ amplitude and/or latency.
Results
425 studies were obtained initially from all databases, with 26 studies fulfilling the eligibility criteria. P300 was the most used ERP (25 studies), showing a reduced amplitude or an increased latency in 84% of those. N400 was evaluated in one study, showing also abnormal results. MMN was addressed in two studies with inconsistent findings. Some studies further suggest that ERPs may show earlier abnormal results compared with neuropsychological tests.
Conclusions
Most MS patient groups revealed ERP abnormalities, suggesting that these neurophysiological tests may be a relevant and appropriate diagnostic aid method for cognitive impairment in MS.
Significance
The use of ERPs in MS patients seems able to demonstrate cognitive impairment and its use should be considered as part of the regular patient evaluation.
Keywords: Multiple Sclerosis; Event-Related Potentials; Cognitive Impairment; Systematic Review


EEG/ERP data on OpenNeuro: 
Nand Chandravadia, Shrita Pendekanti, Dustin Roberts, Robert Tran, Saarang Panchavati, Corey Arnold, Nader Pouratian, and William Speier (2024). Comparing P300 Flashing paradigms in online typing with language models. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds005028.v1.0.0



Kar, Poulami. Introduction to EEG and ERP. drive.google.com/file/d/1Ocimbl6Q7yuGkSnGP6jCLGaCsuwG1-QR/view?usp=sharing. Accessed 24 Oct. 2026.

Ryan, D B et al. “Predictive spelling with a P300-based brain-computer interface: Increasing the rate of communication.” International journal of human-computer interaction vol. 27,1 (2011): 69-84. doi:10.1080/10447318.2011.535754

Pan J, Chen X, Ban N, He J, Chen J and Huang H (2022) Advances in P300 brain–computer interface spellers: toward paradigm design and performance evaluation. Front. Hum. Neurosci. 16:1077717. doi: 10.3389/fnhum.2022.1077717


Kar, Poulami. EEG Analysis Pipeline. drive.google.com/file/d/1mCxGfsGiDCqjDTkPrWc3y7pXGXYnqs-i/view?usp=sharing. Accessed 25 Oct. 2026.


Copyright (C) 2026 The Cerebral Initiative. All rights reserved.
