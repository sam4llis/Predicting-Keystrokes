# Predicting Keystrokes using an Audio Side-Channel Attack and Machine Learning

My MSc Computer Science research project, titled 'Predicting Keystrokes using an Audio Side-Channel Attack and Machine Learning'. The findings and thesis conducted from this research can be found [here](https://www.samwilliamellis.com/src/files/masters_thesis.pdf).

Audio side-channel attacks are increasingly becoming a security concern regarding ‘keystroke snooping’, in which an attack can utilise the emanation of a keystroke to predict a specific key (or contextual passage of keys) being pressed. This can potentially be used to gather a users’ private data if keystroke audio is able to be discretely captured.

In this project, Python code has been created to analyse the acoustic emanation and geometric features of a keystroke signal, and this information is used to provide enough information to accurately classify keystroke emanations. A combination of MFCC and TDoA features are shown to provide superior classification results when compared to other input features.

A novel attack is presented which utilises cross-prediction techniques on a stereo array of microphones to increase keystroke recognition accuracy. Cross-predictions increase singular character recovery of keystrokes by 7% when using a supervised Random Forest machine learning model. A Random Forest classifier is able to achieve up to 89% inter-dataset single-character recovery from a 40-key classification problem.

User experiments are also conducted to show the model in real-world scenarios. In the experiments, up to 85% keystroke recovery from contextual arguments were achieved from a 26-key classification problem using a Random Forest classifier. Keystroke recovery can increase by as much as 15% when utilsiing cross-prediction methods on contextual sentences. Contextual arguments were best predicted when using a user-created database of keystroke emanations.

It is shown in this research that different users emit distinct sonic fingerprints when typing on the same keyboard. Provided that a database of labelled keystrokes can be collected from a user, a supervised attack remains feasible in real-world scenarios.
