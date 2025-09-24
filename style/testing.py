# Script to prove viability of model's analysis of style
# Need to train model with several standardized fonts excluded
# Standardized meaning there exists bold, italicized, serifed, and other variants of the font
# MSE((Normal - Variant), (Predicted Normal - Predicted Variant)) should give some kind of indication of whether the model recognizes difference in style
# Find correlation between the stylistic differences in the ground truth and the predicted images, prove statistical significance