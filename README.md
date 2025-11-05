# Briefcase

Suite of neural networks used as pretraining for font and query contrastive learning. 
Started as an experiment to produce uppercase versions of lowercase glpyhs, taking inspiration from [Tom7's lowercasing work](http://tom7.org/lowercase/). This iteration uses a U-Net architecture operating on pre-rendered bitmap glyphs of the fonts. 
Initial iterations have used ~3,000 fonts, with strong accuracy on the pretraining tasks, and showing reasonable transferability.

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
|             | Lowercase    | Uppercase                    | Masked Autoencoder | CLIP (Image) | Random | Zeroes  |
| :------:    | :------:     | :------:                     | :------:           | :--:         | :----: | :----:  |
| CLIP (Text) | -0.621       | <mark>***-0.619***</mark>    | -0.641             | -0.647       | -1.118 | null    |
| BERT        | 1.601        | <mark>***1.605***</mark>     | 1.602              | 1.570        | -0.628 | null    |

### [TransRate](https://arxiv.org/pdf/2106.09362)
|             | Lowercase    | Uppercase | Masked Autoencoder | CLIP (Image)                    | Random                    | Zeroes  |
| :------:    | :------:     | :------:  | :------:           | :--:                            | :----:                    | :----:  |
| CLIP (Text) | 4.679        | 5.308     | 5.832              | <mark>***9.220***</mark>        | 5.933                     | 0.0     |
| BERT        | 6.327        | 9.193     | 7.798              | 15.86                           | <mark>***17.89***</mark>  | 0.0     |

### [H-Alpha](https://arxiv.org/pdf/2110.06893)
|             | Lowercase                       | Uppercase                    | Masked Autoencoder | CLIP (Image) | Random | Zeroes  |
| :------:    | :------:                        | :------:                     | :------:           | :--:         | :----: | :----:  |
| CLIP (Text) | 1.671                           | <mark>***1.750***</mark>     | 1.582              | 1.463        | 0.0    | 0.506   |
| BERT        | <mark>***1.048***</mark>        | 1.007                        | 0.887              | 0.689        | 0.0    | 0.506   |

### Notes: 
* Scores shown select the best layer from applicable models, CLIP only features the pooled output
* Contrastive tasks are not regression tasks, these are exploratory
* H-Alpha and TransRate require classification-style labels. Several methods were attempted for labeling, including clustering. All returned similar results

## Finetuning

After creating a pretrained model, the scripts in finetune.ipynb can be run to train a set of contrastive models for querying fonts using a style-based text query. 
The tags for the fonts are taken from the Google Fonts repository, and the QueryData dataset is designed explicitly to work with the data in that repository. Any others will require rewrites.
After training a text and image model, search.py offers a basic GUI for searching through fonts in the directory of your choosing. 

# UNet Model

<img width="781" height="574" alt="Briefcase Net drawio" src="https://github.com/user-attachments/assets/729da119-c465-490d-aec1-e815983bb128" />

Note: Glyph classifier was added to force differentiation between characters.

# Pretraining Results

<img width="430" height="175" alt="Screenshot 2025-11-04 223007" src="https://github.com/user-attachments/assets/8e7cac0f-e237-4b59-a4bc-4e49dcf5dd7f" />
<img width="430" height="175" alt="Screenshot 2025-11-04 223028" src="https://github.com/user-attachments/assets/3d3acb3f-80af-4b14-84f5-3636c725a331" />
<img width="430" height="175" alt="Screenshot 2025-11-04 223107" src="https://github.com/user-attachments/assets/80f15e49-0a8d-451e-9e57-3113257c07fb" />
<img width="430" height="175" alt="Screenshot 2025-11-04 223120" src="https://github.com/user-attachments/assets/8087ce41-31bd-42ac-a462-eb00763e7a29" />
<img width="430" height="175" alt="Screenshot 2025-11-04 223135" src="https://github.com/user-attachments/assets/c454bd3e-ad1e-452e-bd4a-281853fc3fe2" />


# Future Work
More font labels. The Google Fonts repo is big, but it only describes a total of ~3600 useable fonts. 
