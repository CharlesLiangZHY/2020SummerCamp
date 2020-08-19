### Usage:

# Show the fourier series of common wave

python3 Wave.py -t WaveType -n NumOfTerms
E.g. 
python3 Wave.py -t square - 3
python3 Wave.py -t sharp  - 0
python3 Wave.py -t oblique - 10

# Fourier Drawing

python3 FourierDraw.py -f filename -n NumOfTerms 

E.g. python3 FourierDraw.py -f img/Bulldog.png -n 20 
If you use WIN, the path should be .\img\Bulldog.png.

-s flag means saving the gif. If you want to save the gif, please install the imagemagick first.

# Fourier Noise Filtering

python3 Filter.py

# Chord Recognition
python3 ChordRecognition.py -f filename -n NumOfNotes

E.g. python3 ChordRecognition.py -f audio/CEG3.wav -n 3
If you use WIN, the path should be .\audio\CEG3.wav.

-p flag means ploting the signal.
-3d flag means ploting spectrogram.