# Briefcase

Neural network designed originally to produce uppercase versions of lowercase glpyhs, taking inspiration from [Tom7's lowercasing work](http://tom7.org/lowercase/). This iteration uses a U-Net architecture operating on pre-rendered bitmap glyphs of the fonts. 
Initial iterations have used ~3,000 fonts, with strong accuracy.

# Getting Started

Running this project is only a matter of cloning it, creating a python environment, and installing the packages in requirements.txt. I recommend fiddling around with test.py for fun first. 

## Training a model

To train a model, you first need to acquire fonts to train it on. I am in no position to manage copyright for literal thousands of fonts, so you will need to find your own fonts. I recommend just using the fonts available on your system as a starting place. 
These should be placed in data/fonts. Then, create data/bitmaps and data/sdf. These will be the directories that store the rendered glyphs. From here, you can run train.ipynb by first running vis.py (to visualize training), then train.ipynb. 
Early stopping can be done by stopping the execution of the cell in the notebook, this will automatically create a checkpoint.
Hopefully the config files are self-explanatory enough if you want to adjust training. Running test.py after training a model will automatically use the most recently trained model.

# Architecture

<img width="781" height="574" alt="Briefcase Net drawio" src="https://github.com/user-attachments/assets/729da119-c465-490d-aec1-e815983bb128" />

Note: Glyph classifier was added to force differentiation between characters.

# Preliminary Results

<img width="883" height="369" alt="Screenshot 2025-09-26 132213" src="https://github.com/user-attachments/assets/41ad392c-1978-4d32-8840-bb07e252d5d5" />
<img width="872" height="379" alt="Screenshot 2025-09-26 132333" src="https://github.com/user-attachments/assets/204f9eba-3d1b-4828-8f59-7c1addfae54e" />
<img width="871" height="366" alt="Screenshot 2025-09-26 134328" src="https://github.com/user-attachments/assets/9d9abb34-ac40-4fe8-a929-2233f4494362" />
<img width="877" height="369" alt="Screenshot 2025-09-26 132421" src="https://github.com/user-attachments/assets/a28901c1-bc3b-41bd-bc85-deebe8df575f" />
<img width="875" height="373" alt="Screenshot 2025-09-26 132253" src="https://github.com/user-attachments/assets/f5cb64a8-51b2-44fb-902e-c43da8b34dc5" />


# Future Work

I plan on both improving this model with a larger dataset and using the model's weights as a pre-training step for a font search tool. 
The idea is to map from style queries like "gothic pixelated font" to a font like [Lady Radical](https://www.dafont.com/search.php?q=lady+radical). 
I think this would be very useful to designers but there aren't really existing datasets for this, so pre-training is likely required at least on the font end. Text embeddings can use BERT probably.
