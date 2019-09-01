# Interview-bot

The problem Interview Bot solves


Many industries now-a-days look for credible employees.So the need of the hour is an Interview Bot which can generate a report of the employee based on his performance.It will analyse - -Voice -Facial Features -Speech(words used) The bot detects the emotions like confidence level and nervousness of the candidate after monetoring the above parameters. The model was very accurate in distinguishing between male/female voice and had 96.25% accuracy overall for speech emotion detection. The bot can have various other uses as detecting emotions can be one of the most useful marketing tool or even ensuring safety of people in smart cars by detecting if one is not fit to drive. The model can have various other untapped uses as people often want to know how others feel.

Challenges we ran into


It was difficult to find good quality datasets as speech datasets are very subjective and can be cause inaccuracy in the model. There was no indian dataset available, so it will be less accurate to indian culture and accents. Due to computational constraints it was not possible to take larger datasets which prohibited us from increasing our models accuracy. We faced a lot of hurdles to implemet our model using Flask . We also faced a lot of python environments conflicts which slowed us down alot. Its very well known that a person's body language(93%) tells more about a person than the words(7%) used while speaking. But we still focused on speech for emotion recognition because we achieved a lot higher accuracy than prediciting through facial features.

Technologies we used


pythonTensorFlowkerasSpeech Synthesisdeep learningimage processingCNNMFCC




Have made a Speech Emotion detection model using CNN
96.25% accuracy
Used the The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset :https://zenodo.org/record/1188976#.XV_btnvhVPZ

Model in file named Module2.

