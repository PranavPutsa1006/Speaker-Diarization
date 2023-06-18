# Speaker-Diarization

### Abstract 

Speaker Diarization is the process of identifying individual speakers in an audio stream based on the unique characteristics found in individual voices.
The goal is not to identify known speakers, but to co-index segments that are associated with the same speaker; in other words, diarization intends to find speaker boundaries and grouping segments that belong to the same speaker, and, as a by-product, ascertaining the number of distinct speakers. In combination with speech recognition, diarization enables speaker-attributed speech-to-text transcription.
Speaker Diarization is an integral part in human interaction and communication, which makes it important in the evolution of Autonomous machines and Interaction systems.

### Implementation of Speech Detection 
1) HTML, JavaScript to capture audio input from Microphone
2) Plotting audio waveform using Matplotlib
3) Export Audio to .wav file for further processing
4) Reduce stationary noise using low pass filter
5) Split audio and reattach after removing silence
 
### Implementation of Segmentation 
1) Segment audio based on ‘speech’ (or ‘male’/ ‘female’), ‘noise’, ‘silence’, ‘music’ using inaSpeechSegmenter Module
2) Transcribe the speech segments

### Implementation of Embedding extraction
The MFCC (Mel Frequency Cepstral Coefficient) of the audio segments is extracted. These are basically feature coefficients which capture the variations in the speech like pitch, quality, intone etc of the voice in a much better way. They are obtained by doing a specialized Fourier Transform of the speech signal. The Python Library Librosa helps to visualize the audio signals and also do the feature extractions in it using different signal processing techniques.
Implementation of Spectral Clustering
1) Standard scaler for original data columns
2) Normalize
3) PCA (2d)
4) Pairwise_distances and vectorize (Adjacency matrix(0/1))
5) Degree matrix and Laplacian matrix
6) Eigenvectors and Eigenvalue from laplacian matrix 
7) Find k-means clustering using eigenvectors

### Tools 
-	Colab Notebook
-	MS teams (for real time recording)
-	Seaborn
-	DeepSpeech Data set 
-	MS teams meeting recordings

### Python Libraries used: 
-	Numpy
-	Matplotlib 
-	Pandas 
-	Seaborn
-	Sklearn 
-	Webrtcvad 
-	IPython
-	Librosa
-	Scipy
-	ffmpeg-python
-	Pydub
-	Tensorflow-gpu
-	inaSpeechSegmenter
-	Pyaudio
-	SpeechRecognition
-	Scikit
-	DeepSpeech
-	Wave

### Description of modules used

- loadClusterData()
  > Takes the transposed mfccs and converts it into DataFrame.Then does the forward filling.

- ScaleAndNorm()
  > To Scale and normalize the data.

- getPCA()
  > Reduce the dimension/features to 2 using principal component analysis.

- getVectorize()
  > Calculating euclidean distance between 2 columns taking as x and y coordinates and then vectorizing it to convert to adjacency matrix.

- getLaplacian()
  > To Find the degree matrix and then convert it to laplacian matrix.

- eigenvectorEigenvalue()
  > To Find the eigenvalue and the corresponding eigenvectors.

- toNumpy()
  > Converts PCA data to numpy array and decide iteration for k-means clustering.

- decidingRandomCentriods()
  > Random centroids decided

- euclidDist()
  > Calculates euclid distance

- iteration1()
  > Assigns points to centroid and updates them.

- iterations()
  > Repeats the process of assigning points to centroid and updating them. 

- calc_distance()
  > Finds the distance to final centroid to form the cluster

- findClosestCentroids()
  > Derives the clustered array from final centroid

- get_audio()
  > Display HTML for button, evaluate the corresponding JavaScript to capture audio from system Microphone

- transcribe()
  > Uses Google’s SpeechRecognition library to transcribe the speech segments in the input audio using the segmentation list which contains start and end times of segments

- inaSpeechSegmenter
  > Returns a list of tuples with type of audio (‘speech’, ‘noise’ or ‘noEnergy’), start and end times of all the segments

- PyDub - AudioSegment, split_on_silence
  > Used to read and export audio files, also to remove silence in the audio to decrease file size and increase efficiency

- createStream()
  > Create a new streaming inference state.

- getframerate()
  > Returns the framerate of the buffer

- getnframes()
  > Returns the total number of frames in the buffer


### Problems encountered and Solutions:
- Audio input in colab notebook

  > Using JavaScript to stream system Microphone audio through Chrome browser and HTML to create a button for interaction with the console.
  

- Automatic Voice Activity Detection

  > Using the inaSpeechSegmenter module in Python, the recorded audio is stripped of silences more than 500 msec long and reconstructing the audio with segments with decibels higher than -36 (the value identified to differentiate speech and silence while recording audio directly from colab notebook or -26 incase of pre-recorded audio files).

- Finding number of clusters before clustering
  > Using elbow graph we drew a straight line and found the distance from it to the curve and took the max distance index as the number of clusters. 


- Validation measure for unsupervised model
  > Silhouette score with other models accuracy

### Result 
A speaker diarization module is hence built using the most suitable functions and libraries involved in the concepts of speech detection and segmentation, embeddings extraction, spectral clustering, and transcription. Speech recordings are given as input and through the related wave signals, heat maps and graphs, the speakers are identified along with the corresponding time frames, which makes the diarization module a successful one.


 


### Scope of future work 
Spectral clustering is mostly used unsupervised data analysis technique in the field of clustering. Our future work direction is to connect our clustering results with text transcripts and give the user more organized data such that it becomes easier to integrate it with real applications like MS teams to differentiate speakers contents effortlessly. Also, we would explore more measures to make our algorithm more accurate like using DNN autoencoders.
