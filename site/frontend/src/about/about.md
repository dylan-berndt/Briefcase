# About

## Introduction

Hello there! This is Font Search, a project of mine I've wanted to use for years: a plain-text font searching tool with free fonts.
It's not exactly as I envisioned it (still relies on a set of tags for matching), but it functions about as good as I could achieve within a year's worth of work.
Either way, I've produced something good enough to put my name on. Enjoy!

I got inspired to work on this project by watching a great video by Tom7 on experimenting with uppercasing and lowercasing letters with neural networks, and all the nonsense you can do with a trained model. I recommend you watch every Tom7 video except the ones you don't like probably.

____ (Tom7 video)

All fonts are hosted on DaFont and Google Fonts, this site just acts to score them according to your query. 

## Usage

The search mostly operates like any Google search. You figure out what you want to look for, and the underlying models try to figure out what content best matches your query. In the case of this project, the search relies on a fixed set of visual tags that describe each font. For this reason, if you can't seem to find a font you're looking for, or the search doesn't return tags that are useful to you, it can help to use synonyms of the visual concept you're looking for. Or just scroll, there's a hell of a lot of fonts. This system also isn't perfect. All the tags that are assigned to the fonts were assigned by a model, and can have errors. If you encounter an error, make sure to hit the disapprove button on the search result, that helps me build better models for next time.

## Architectural Overview

This project was a huge head-bash into neural networks -- particularly pre-training, fine-tuning, ____, and ____. The final architecture I landed on works like this: A pre-trained vision transformer embeds an image that contains every glyph in a font into a vector that represents the visual information communicated by the image. That embedding is passed through a much smaller network that's designed to figure out what traits that font has (wide, tall, spooky, etc.). 

These tags are acquired from a public dataset of fonts put together in this paper: ____. Note: It's necessary to predict tags for fonts because the MyFonts dataset contains fonts that aren't publicly available, and the DaFont/Google Fonts fonts don't have robust tagging. I could have spent money or talked to people about getting proper human-curated tags, but I'm a computer scientist. 

The predicted ratings for each font are normalized across the dataset and can then be searched by the tag search engine. The tag search engine uses basic natural language processing (snowballstemmer, spacy synonym-matching) to determine which tags are closest to the words in your query (the vocab only contains ~1800 tags, and there's a lot of adjectives in the world), and uses the matched tags and close-enough tags to search the corpus. 

# Creating the Tool

This project has been over a year in the making before I finally produced a model that I could be proud of for searching fonts, and it started with a school project to demonstrate that I could read like 8 research papers in 4 months. For this reason, I made the project way too complex and made a huge mess. But I learned a hell of a lot about computer vision and other related stuff. 

I should be clear, I prefer to learn things by bashing my head into them until I figure out that I should have been kicking them or something. There are a few times I start down a path that is clearly doomed, but that's how I do it.

Let's start there. This project was originally an exercise into investigating pretraining for text -> font searching methods. There exist a few good approaches, but I wanted to evaluate them in a standard framework and to create a lowercasing neural network of my own.

## Casing the Scene

The first dataset and model I implemented were the default fonts on my Windows computer and a UNet. UNets are models that I am very familiar with that got their start in medical imaging, but have a lot of useful properties and are very easy to implement. So I did. I trained the model to take in an image of a lowercase letter rendered as a 32x32 image and output the uppercased version of that image. This worked okayish. 

____ (UNet)

There were several problems. First, the model didn't know where exactly to place the uppercased letter thanks to some rendering bugs and ascent and descent and all that. Second, because the model didn't have any idea which letter it was looking at (and didn't care to figure it out apparently), it usually output a blob of some kind that looked a little bit like the letter if you weren't looking at the monitor. So I added auxiliary tasks to make the model better at its job. The first was to reduce similarity between the input image and the output image. That way, the model was striving to produce something new even early on in training. This didn't really change much in terms of the model's score, but I liked that it produced images that were more interesting. The second auxiliary task was much more effective, but a little disappointing. I trained the model to produce an internal representation that could be transformed to predict the specific letter the model was looking at. To be clear, I just took the most downsampled layer activations in the UNet and fed it into a feed-forward network trained with softmax. This drastically improved the model, but it did make sure that the model didn't produce bonkers results for an uppercase 4. Oh well. 

As a result of this second task, though, I could generate my own fonts. Not by "generating" a font with "generative neural networks". Gross. We steal again from Tom7, and ask our fully-trained model to rank a random image of noise, ask it if it looks like an F, and if it doesn't, we edit the image until it does. Optimize until you get its favorite F ever. I produced two full "fonts" with this, one of them utilizing softmax (this is an F and only an F) and another using plain maximization (>100% F). These fonts I have titled: serif-fim and Oblivion, respectively.

Somewhere along here, I also pulled in the Google Fonts repository of fonts to get a lot more training data. Anyways, this model worked well enough for my purposes. I could've extended it with diffusion to reduce the blurriness of the final images, but I prefer believing that uppercasing is a deterministic task.

____ (Casing results)


### Style Learning

Having an uppercasing model is cool and all, but the goal here was to prove that the pretraining could encode style information and transfer that learning for use elsewhere. There are a few ways I could have properly tested this, but again, Head Basher 9000. This led to a few fun experiments in figuring out whether the models could encode style. 

The first of these was pretty simple. ____ (Bolding). This showed what I needed it to, but I wasn't satisfied.

____ (Bold results)

Next up, we could measure the model's actual internal understanding of the fonts. The idea was this: if the model produces representations that look similar when they're from the same font (even when they're a different letter), but look different when they're the same letter (but a different font), then the model has in some way figured out that it should encode the style of the font it's looking at. These results were okay and the logic is annoying to explain, so I moved on.

____ (Cosine results)

The next method is a lot more standard. Take PCA on the representations, then visualize the vectors on a chart. If similar looking fonts clump together, then the model is naturally learning to encode style. This one could be taken on each individual layer and produced some of the coolest results, with different font styles clumping together on different layers.

____ (PCA results)

Last up, plain clustering. This way, we can visualize the full structure of the learned representations without having to squish everything down to only 2 dimensions with PCA. Considering there's a lot of fonts and the dimension of the model is 256, we're leaving a lot of style information on the table if we squish that much. Here's the results for clustering at layer 4, which is somewhere in the middle of the model.

____ (Clustering results)

All this work was definitely fun and cool and such, but it didn't really contribute anything (especially academically) to my school project. So we had to move on to actually testing ablations and all that.

### Some Science

Here's what we needed to prove: can we find a pretraining method that produces a model with more stylistic information representation than the existing approach (FontCLIP) to pretraining for font searching? We needed a real methodology to prove this with, and it turns out there are a lot of variations of one in particular: Transferability Estimation. With transferability estimation, it is possible to read the outputs of a pre-trained neural network and determin how likely its representations can transfer to a new task, like font searching. So, I would test my new pre-training methods' ability to transfer, and compare the transferability estimation score to the existing approach. Then, we could perform the actual transfer with a linear probe on top of the models and a finetuning run.

Great. We have a process. To test, I set up three pre-training methods of my own. First, uppercasing and lowercasing. Trained each with fixed run lengths and model sizes. Next, I needed an industry standard pretraining technique. I chose to evaluate masked autoencoding, a way of training computer vision networks that involves masking out part of the image and asking the model to reproduce the information that we deleted. ____.

### The Website

### Hiatus

## Oh No

I came back to the project in March of 2026. I had no good reason to, but I was now determined to use contrastive learning with vision transformers for the pre-training. ____.

### Creating Something Good

The best thing to come out of this investigation and everything after was this: a flower. The pre-trained model produced embeddings of its own that represented the visual information in each font image. I could take these embeddings, use PCA or tSNE similar to before, but observe the outputs that were trained specifically on style. This time, however, because of some recent work with medical imaging, I wanted to do a 3D render. So I embedded the Google Fonts repository using the model, and compressed it down with tSNE to 6 dimensions. I then assigned the dimensions to X, Y, Z, and R, G, B as dots in a render. This, for one randomly initialized model, trained on randomly ordered fonts, produced (to me) a very convincing flower. It has purple petals, a yellow ____, and a layer of green grass underneath. If I wanted to print the shape, there's even a little leg on the underside that could hold up the grass at a good angle. Of course, any 3D rendering looks like anything to someone that really wants meaning from their work. But I really like this: it placed all the cursive fonts in that yellow ____. It's just neat. 

____ The flower, and link to explore

### Another Good Idea

Coming fresh off the heels of that good idea, I figured that maybe it would be best to ditch the whole contrastive learning post-training situation that was producing such garbage 24/7 and replace it with tagging. It made sense: if the model couldn't figure out which caption matched to which font because so many words were shared between fonts, then it made more sense to predict what words each font belonged to and then let the user pick how specific they were. This model was good, and I produced a result nearly on par with the SOTA (____) by messing around for a bit. However, when implementing a search that used this method, I got frustrated with non-matching tags and got extremely overzealous with implementing synonym matching on every word. This ruined the search, and it was the only version of the tag search that I ever actually tried. And for whatever reason, I decided to ditch this good idea in favor of a worse one.

### Detour

The idea was pretty simple and relied on my terrible understanding of information theory and Akinator. If we ask the user to pick between two fonts, and we can divide the space of fonts in half on every question, then we only need to ask the user a few questions to get to the *exact* font they want. It works out on a piece of paper designed explicitly for fools. It looks like this: we have 40,000 fonts, so we need ceiling(log_2(40,000)) = 16 total bits of information or 16 total questions. We could even ask three-way, four-way questions, eight-way (just 6 total questions!), whatever! This is what I ended up calling the meander search (fitting name). I would take a few different shots at this, but my first implementations relied on updating a randomly initialized embedding vector to point towards which ever font the user selected, and away from the ones they didn't. That way, the vector moves towards ____.

This part of the project is where I began relying more on the use of generative LLMs like Claude. :(

## The Devil Puts His Hand in My Brain

### Trust, but Verify

### Verifying

### Back to Reason

# Conclusion

I don't know why I've been so much more determined to make this work than any other project. Maybe it's the wide range of tasks I could work on at any time to keep pushing it forward. Maybe it's because I thought it should've been really easy, and if I couldn't do it then I must be a doofus. For reference on that determination part, I'm writing this About page at my job for money when I could be lazy a lot more other really useful ways (plan my wedding, look for new cars, plan brother's bachelor party, schedule even one doctor appointment ever, etc.) Either way, I'm glad to have created something good enough to release. In the course of this project, I have: moved houses, decided to not complete my thesis, finished my master's degree, moved into a full-time position (research analyst, risk and emerging technologies /[I don't know why, honestly]), got married (eloped lol), and got a new cat. I don't know where that leaves me, but this thing was a part of most of my nights. Idk. Thanks for reading.