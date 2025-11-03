# Briefcase

Suite of neural networks used as pretraining for font and query contrastive learning. 
Suite of neural networks designed originally to produce uppercase versions of lowercase glpyhs, taking inspiration from [Tom7's lowercasing work](http://tom7.org/lowercase/). This iteration uses a U-Net architecture operating on pre-rendered bitmap glyphs of the fonts. 
Initial iterations have used ~3,000 fonts, with strong accuracy.

# Getting Started

Running this project is only a matter of cloning it, creating a python environment, and installing the packages in requirements.txt. I recommend fiddling around with draw.py for fun first. 

## Training a model

To train a model, you first need to acquire fonts to train it on. I am in no position to manage copyright for literal thousands of fonts, so you will need to find your own fonts. I recommend just using the fonts available on your system as a starting place. 
These should be placed in data/fonts. From here, you can run pretrain a model by first running vis.py (to visualize training), then pretrain.ipynb. 
Early stopping can be done by stopping the execution of the cell in the notebook, this will automatically create a checkpoint.
Hopefully the config files are self-explanatory enough if you want to adjust training. Running draw.py currently uses a pretrained uppercasing model, but can be changed quickly in the testPath at the top of the file.

## Transferability Estimation

Part of this project consisted of evaluating the pretrained models' ability to transfer learning to a contrastive learning task (which is treated as regression) using measures like LogME, TransRate and H-Alpha.
I also compare to two poor models, including one which outputs random gaussian noise and another which outputs only 0s.
You can perform this yourself in estimation.py, but here are the results from the models I pretrained. 

### [LogME](https://arxiv.org/abs/2102.11005)
|             | Lowercase    | Uppercase | Masked Autoencoder | CLIP (Image) | Random | Zeroes  |
| :------:    | :------:     | :------:  | :------:           | :--:         | :----: | :----:  |
| CLIP (Text) | -0.621       | -0.619    | -0.641             | -0.647       | -1.118 | null    |
| BERT        | 1.601        | 1.605     | 1.602              | 1.570        | -0.628 | null    |

### [TransRate](https://arxiv.org/pdf/2106.09362)
|             | Lowercase    | Uppercase | Masked Autoencoder | CLIP (Image) | Random | Zeroes  |
| :------:    | :------:     | :------:  | :------:           | :--:         | :----: | :----:  |
| CLIP (Text) | 14.02        | 14.20     | 11.96              | 15.94        | 9.477  | 0.0     |
| BERT        | 21.49        | 23.58     | 20.64              | 26.45        | 23.20  | 0.0     |

### [H-Alpha](https://arxiv.org/pdf/2110.06893)
|             | Lowercase    | Uppercase | Masked Autoencoder | CLIP (Image) | Random | Zeroes  |
| :------:    | :------:     | :------:  | :------:           | :--:         | :----: | :----:  |
| CLIP (Text) | 1.671        | 1.750     | 1.582              | 1.463        | 0.0    | 0.506   |
| BERT        | 1.048        | 1.007     | 0.887              | 0.689        | 0.0    | 0.506   |

### Notes: 
* Scores shown select the best layer from applicable models, CLIP only features the pooled output
* Contrastive tasks are not regression tasks, these are not definitive
* H-Alpha and TransRate require classification-style labels. Several methods were attempted for labeling, including clustering. All returned similar results

## Finetuning

After creating a pretrained model, the scripts in finetune.ipynb can be run to train a set of contrastive models for querying fonts using a style-based text query. 
The tags for the fonts are taken from the Google Fonts repository, and the QueryData dataset is designed explicitly to work with the data in that repository. Any others will require rewrites.
After training a text and image model, search.py offers a basic GUI for searching through fonts in the directory of your choosing. 

# UNet Model

<img width="781" height="574" alt="Briefcase Net drawio" src="https://github.com/user-attachments/assets/729da119-c465-490d-aec1-e815983bb128" />

Note: Glyph classifier was added to force differentiation between characters.

# Pretraining Results

<img width="883" height="369" alt="Screenshot 2025-09-26 132213" src="https://github.com/user-attachments/assets/41ad392c-1978-4d32-8840-bb07e252d5d5" />
<img width="872" height="379" alt="Screenshot 2025-09-26 132333" src="https://github.com/user-attachments/assets/204f9eba-3d1b-4828-8f59-7c1addfae54e" />
<img width="871" height="366" alt="Screenshot 2025-09-26 134328" src="https://github.com/user-attachments/assets/9d9abb34-ac40-4fe8-a929-2233f4494362" />
<img width="877" height="369" alt="Screenshot 2025-09-26 132421" src="https://github.com/user-attachments/assets/a28901c1-bc3b-41bd-bc85-deebe8df575f" />
<img width="875" height="373" alt="Screenshot 2025-09-26 132253" src="https://github.com/user-attachments/assets/f5cb64a8-51b2-44fb-902e-c43da8b34dc5" />


# Future Work
More font labels. The Google Fonts repo is big, but it only describes a total of ~3600 useable fonts. 
