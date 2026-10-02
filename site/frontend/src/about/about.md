# About

## Introduction

Hello there! This is Font Search, a project of mine I've wanted to use for years: a plain-text font searching tool with free fonts.
It's not exactly as I envisioned it (still relies on a set of tags for matching), but it functions about as good as I could achieve within a year's worth of work.
Either way, I've produced something good enough to put my name on. Enjoy!

I got inspired to work on this project by watching a great video by Tom7 on experimenting with uppercasing and lowercasing letters with neural networks, and all the nonsense you can do with a trained model. I recommend you watch every Tom7 video except the ones you don't like probably.

____ (Tom7 video)

All fonts are hosted on DaFont and Google Fonts, this site just acts to score them according to your query. 

## Usage

The search mostly operates like any Google search. You figure out what you want to look for, and the underlying models try to figure out what content best matches your query. In the case of this project, the search relies on a fixed set of visual tags that describe each font. For this reason, if you can't seem to find a font you're looking for, or the search doesn't return tags that are useful to you, it can help to use synonyms of the visual concept you're looking for. Or just scroll, there's a hell of a lot of fonts. This system also isn't perfect. All the tags that are assigned to the fonts were assigned by a model, and can have errors. If you encounter an error, make sure to hit the disapprove button on the search result, that helps me build better models for next time. For issues with things other than poor search results, please email me directly <a href="mailto:dylanberndt123@gmail.com">here</a>.

## Architectural Overview

This project was a huge head-bash into neural networks -- particularly pre-training, fine-tuning, ____, and ____. The final architecture I landed on works like this: A pre-trained vision transformer embeds an image that contains every glyph in a font into a vector that represents the visual information communicated by the image. That embedding is passed through a much smaller network that's designed to figure out what traits that font has (wide, tall, spooky, etc.). 

These tags are acquired from a public dataset of fonts put together in this paper: ____. Note: It's necessary to predict tags for fonts because the MyFonts dataset contains fonts that aren't publicly available, and the DaFont/Google Fonts fonts don't have robust tagging. I could have spent money or talked to people about getting proper human-curated tags, but I'm a computer scientist. 

The predicted ratings for each font are normalized across the dataset and can then be searched by the tag search engine. The tag search engine uses basic natural language processing (snowballstemmer, spacy synonym-matching) to determine which tags are closest to the words in your query (the vocab only contains ~1800 tags, and there's a lot of adjectives in the world), and uses the matched tags and close-enough tags to search the corpus. 

## Me

____. I created this tool mostly by myself [citation needed]. I'm a Research Analyst, Risk and Emerging Technologies at Associated Electric Cooperative. 

# Creating the Tool

This project has been over a year in the making before I finally produced a model that I could be proud of for searching fonts, and it started with a school project to demonstrate that I could read like 8 research papers in 4 months. For this reason, I made the project way too complex and made a huge mess. But I learned a hell of a lot about computer vision and other related stuff. 

I should be clear, I prefer to learn things by bashing my head into them until I figure out that I should have been kicking them or something. There are a few times I start down a path that is clearly doomed, but that's how I do it.

Let's start here. This project was originally an exercise into investigating pretraining for text -> font searching methods. There exist a few good approaches, but I wanted to evaluate them in a standard framework and to create a lowercasing neural network of my own.

## Casing the Scene

The first dataset and model I implemented were the default fonts on my Windows computer and a UNet. UNets are models that I am very familiar with that got their start in medical imaging, but have a lot of useful properties and are very easy to implement. So I did. I trained the model to take in an image of a lowercase letter rendered as a 32x32 image and output the uppercased version of that image. This worked okayish. 

![An image of a standard UNet architecture.](./images/unet.png)
Figure 1: An image of a standard UNet architecture. Obtained from here: [brain segmentation](https://github.com/mateuszbuda/brain-segmentation-pytorch)

There were several problems. First, the model didn't know where exactly to place the uppercased letter thanks to some rendering bugs and ascent and descent and all that. Second, because the model didn't have any idea which letter it was looking at (and didn't care to figure it out apparently), it usually output a blob of some kind that looked a little bit like the letter if you weren't looking at the monitor. So I added auxiliary tasks to make the model better at its job. The first was to reduce similarity between the input image and the output image. That way, the model was striving to produce something new even early on in training. This didn't really change much in terms of the model's score, but I liked that it produced images that were more interesting. The second auxiliary task was much more effective, but a little disappointing. I trained the model to produce an internal representation that could be transformed to predict the specific letter the model was looking at. To be clear, I just took the most downsampled layer activations in the UNet and fed it into a feed-forward network trained with softmax. This drastically improved the model, but it did make sure that the model didn't produce bonkers results for an uppercase 4. Oh well. 

![Image of the results of the uppercasing model. Shows a crudely drawn pixel lowercase e on the left, and a blurry but legible uppercase e on the right](./images/upper.png)
Figure 2: The results of one of the best uppercasing models, shown on a crudely drawn lowercase e.

As a result of this second task, though, I could generate my own fonts. Not by "generating" a font with "generative neural networks". Gross. We steal again from Tom7, and ask our fully-trained model to rank a random image of noise, ask it if it looks like an F, and if it doesn't, we edit the image until it does. Optimize until you get its favorite F ever. I produced two full "fonts" with this, one of them utilizing softmax (this is an F and only an F) and another using plain maximization (>100% F). These fonts I have titled: serif-phim and Oblivion, respectively.

<img src="./images/font_samples.png" width="100%" alt="Renderings of the serif-fim and oblivion fonts">

Figure 3: Renderings of the serif-phim and Oblivion fonts. These are showing a, b, c, d, and e obviously.

Somewhere along here, I also pulled in the Google Fonts repository of fonts to get a lot more training data. Anyways, this model worked well enough for my purposes. I could've extended it with diffusion to reduce the blurriness of the final images, but I prefer believing that uppercasing is a deterministic task.


### Style Learning

Having an uppercasing model is cool and all, but the goal here was to prove that the pretraining could encode style information and transfer that learning for use elsewhere. There are a few ways I could have properly tested this, but again, Head Basher 9000. This led to a few fun experiments in figuring out whether the models could encode style. 

The first of these was pretty simple because it relied on the simplest kind of style most fonts have. I would test whether the model had effectively learned the difference between the standard rendering of a font and its bold and italic versions. I essentially just had to compare the outputs of a few standard fonts with their bold and italic counterparts and meaure the difference. This technically showed that the model understood one part of style, but it didn't show differentiation between fonts. So I moved on.

____ (Bold results)

Next up, we could measure the model's actual internal understanding of the fonts. The idea was this: if the model produces representations that look similar when they're from the same font (even when they're a different letter), but look different when they're the same letter (but a different font), then the model has in some way figured out that it should encode the style of the font it's looking at. These results were okay, and better than the last. It showed that the model encoded fonts in a way that was somewhat consistent, but it was clear that a lot of the representations were dedicated to solving the letter being shown. 

![alt text](./images/sample.png)

Figure 4: The fonts that were chosen to evaluate styling. These fonts were ordered to loosely represent a transition from one font style to another. The idea was for the cosine activations to look mostly like a gradient coming out from the diagonals if the model's understanding of visual similarity was similar to human's understanding

![Plot showing several matrices with the cosine similarity between activations for a range of fonts](./images/different.png)

Figure 5: Plot of the activation cosine similarity across different layers. In this case, we compare only situations with differing letters.

![Plot showing several matrices with the cosine similarity between activations for a range of fonts](./images/same.png)

Figure 6: Plot of the activation cosine similarity across different layers. In this case, we compare situations where the letter is identical, showing the similarity between styles.

The next method is a lot more standard. Take PCA on the representations, then visualize the vectors on a chart. If similar looking fonts clump together, then the model is naturally learning to encode style. This one could be taken on each individual layer and produced some of the coolest results, with different font styles clumping together on different layers.

![Image showing the PCA embedded vector placings for each font in the original corpus](./images/pca.png)

Figure 7: This is the PCA embeddings for each font in the original corpus, by layer, color coded for the use of bold and italic. 

Last up, plain clustering. This way, we can visualize the full structure of the learned representations without having to squish everything down to only 2 dimensions with PCA. Considering there's a lot of fonts and the dimension of the model is 256, we're leaving a lot of style information on the table if we squish that much. Here's the results for clustering at layer 4, which is somewhere in the middle of the model.

____ (Clustering results)

All this work was definitely fun and cool and such, but it didn't really contribute anything (especially academically) to my school project. So we had to move on to actually testing ablations and all that.

### Some Science

Here's what we needed to prove: can we find a pretraining method that produces a model with more stylistic information representation than the existing approach (FontCLIP) to pretraining for font searching? We needed a real methodology to prove this with, and it turns out there are a lot of variations of one in particular: Transferability Estimation. With transferability estimation, it is possible to read the outputs of a pre-trained neural network and determin how likely its representations can transfer to a new task, like font searching. So, I would test my new pre-training methods' ability to transfer, and compare the transferability estimation score to the existing approach. Then, we could perform the actual transfer with a linear probe on top of the models and a finetuning run.

Great. We have a process. To test, I set up three pre-training methods of my own. First, uppercasing and lowercasing. Trained each with fixed run lengths and model sizes. Next, I needed an industry standard pretraining technique. I chose to evaluate masked autoencoding, a way of training computer vision networks that involves masking out part of the image and asking the model to reproduce the information that we deleted. ____.

### The Website

By this point, I decided I wanted to make this a real tool and not just a learning project. ____. ____. Creating this website did provide one motivation for the project. I was the one paying for hosting, and not hosting something there was wasteful.

### Hiatus/Thesis

Oddly enough, right after deciding that I wanted to create an entire website around this already convoluted mess, I got tired of the project. That's not entirely true. I also started my thesis around this time (Octoberish 2025), and the work was enough to take me away from this project. So I worked on the thesis for months. My lofty goal was to produce a global river forecasting model that could outperform Google's Flood Hub. This was entirely my choosing, and my entire computer science department was unfamiliar with the methods I was using to achieve my goal. I toiled away at designing these spatiotemporal neural networks for ____. I ____, spending a significant portion of my winter break that year driving to campus so I could sit outside the student union in the 4-40 degree (Fahrenheit) weather for the wifi that connected to the (super?)computer that I used to run my experiments (40GB VRAM lol). I worked on this day and night ____.

## Trying Again

I came back to the project in March of 2026. I had no good reason to, but I was now determined to use contrastive learning with vision transformers for the pre-training. I wanted to produce the best possible version of a font searching tool and I knew it had to be possible. This obsession with improving the tool would produce a lot of things, some good and some bad. The setup worked like this: take an image and run it through a vision transformer alongside an extra token, [CLS], that would be extracted afterwards and fed to the loss. The loss was InfoNCE paired with SIGReg (I had recently read up on JEPA and its siblings). InfoNCE acted to take the [CLS] token and pull the values in the token closer to the other tokens produced by the model running other glyphs from the same font, and push the values away from the other fonts. This would make the model invariant to the specific letter passed in, but require that it produced unique embeddings for each font that were identifying enough to match that font and no other font. This did work. 

### Creating Something Good

The best thing to come out of this investigation and everything after was this: a flower. The pre-trained vision transformer produced embeddings of its own that represented the visual information in each font image. I could take these embeddings, use PCA or tSNE similar to before, but observe the outputs that were trained specifically on style. This time, however, because of some recent work with medical imaging, I wanted to do a 3D render. So I embedded the Google Fonts repository using the model, and compressed it down with tSNE to 6 dimensions. I then assigned the dimensions to X, Y, Z, and R, G, B as dots in a render. This, for one randomly initialized model, trained on randomly ordered fonts, produced (to me) a very convincing flower. It has purple petals, a yellow stamen, and a layer of green grass underneath. If I wanted to print the shape, there's even a little leg on the underside that could hold up the grass at a good angle. Of course, any 3D rendering looks like anything to someone that really wants meaning from their work. But I really like this: it placed all the cursive fonts in that yellow stamen. It's just neat. 

____ The flower, and link to explore

### Another Good Idea

Coming fresh off the heels of that good idea, I figured that maybe it would be best to ditch the whole contrastive learning post-training situation that was producing such garbage 24/7 and replace it with tagging. It made sense: if the model couldn't figure out which caption matched to which font because so many words were shared between fonts, then it made more sense to predict what words each font belonged to and then let the user pick how specific they were. This model was good, and I produced a result nearly on par with the SOTA (____) by messing around for a bit. However, when implementing a search that used this method, I got frustrated with non-matching tags and got extremely overzealous with implementing synonym matching on every word. This ruined the search, and it was the only version of the tag search that I ever actually tried. And for whatever reason, I decided to ditch this good idea in favor of a worse one.

### Detour

The idea was pretty simple and relied on my terrible understanding of information theory and Akinator. If we ask the user to pick between two fonts, and we can divide the space of fonts in half on every question, then we only need to ask the user a few questions to get to the *exact* font they want. It works out on a piece of paper designed explicitly for fools. It looks like this: we have 40,000 fonts, so we need $$ceiling(log_2(40000)) = 16$$ total bits of information or 16 total questions. We could even ask three-way, four-way, eight-way question, whatever! The more options, the less choices the user had to make. This is what I ended up calling the meander search (fitting name). I would take a few different shots at this, but my first implementations relied on updating a randomly initialized embedding vector to point towards which ever font the user selected, and away from the ones they didn't. That way, the vector moves towards ____.

____ (animation of the thingy)

This part of the project is where I began relying more on the use of generative LLMs like Claude. What a fool.

I tried taking the scientific approach: gather a list of proper articles on the subject of user choice, iterative refinement, optimization, bayesian statistics; then replicate the methods and observe the expected results. But I goofed. I pressed forward with all these techniques (____), iteratively improving the number of choices, total questions asked, exploration and exploitation of the methods. I made the best possible version of this Akinator idea. For a machine. A machine that understands these embeddings at a (near) perfect level. I did try simulating noise. The oracle that made decisions on the choices would randomly make the wrong choice, and the system could recover. But I didn't measure how often people would make the wrong choice. Or if they could ever make a right one. When I finally implemented a search (a month later) that used the methods I had refined so well, it was impossible to navigate. Even when I showed the user 50+ fonts per choice they had to make, there was no way to determine what kind of fonts I was being asked to decide between. 100% failure rate, even when looking for a very, very specific font.

From here, I came to the conclusion that I had chosen the wrong direction for the search. Not the wrong direction for development. I began using Claude Code for $5/month.

## The Devil Puts His Hand in My Brain

I had decided I wasn't happy with these weird ways of searching the font corpus. And I wasn't happy with the poor performance of the contrastive learning setup. I figured there had to be some way to make this work whether it wanted to work or not. ____

### Trust, but Verify

____

### Verifying

____

### Back to Reason

____

## The Final Product

After testing the new variant of the LeVJEPA trained model, I was able to come up with nearly the exact same performance metrics of the best model I had trained months ago. Turns out I had done all I needed to do in terms of pre-training. From here, I took the fine-tuned version of my older model and ran each font through it. I put together a sample of some queries and their results, and it looked good. I started measuring the number of results that matched the query by my own eye, rather than using the one, single, "correct" answer as the be-all-end-all of performance. It worked. Simple tag matching, with a co-occurence letting words from the generated captions map to others in the vocabulary, and then synonym matching after. The previous experiments ate up a full month, most of which was spent in hotels traveling for my new job. My impatience in airports and planes and conference seats made me rush to build something that was designed for anything but people. This final attempt using the tags took less than a week to implement and fully deploy to the website.

## The Future

I would love for this to be the best font searching experience there is. I really do care about it. I have mechanisms in place to capture usage for the sake of improving the model. The site just grabs the tags you used and the query you typed if you approve/disapprove. You also need to login to do this. Either way. I will probably still be obsessed with this. I will have a ton of silly new ideas that I have no idea how to begin verifying, and I will smash my head into a wall trying to prove myself right. Whether or not it works is up to the strength of the wall.


# Conclusion

I don't know why I've been so much more determined to make this work than any other project. Maybe it's the wide range of tasks I could work on at any time to keep pushing it forward. Maybe it's because I thought it should've been really easy, and if I couldn't do it then I must be a doofus. For reference on that determination part, I'm writing this About page at my job for money when I could be lazy a lot more other really useful ways (plan my wedding, look for new cars, plan brother's bachelor party, schedule even one doctor appointment ever, etc.) Either way, I'm glad to have created something good enough to release. In the course of this project, I have: moved houses, decided to not complete my thesis, finished my master's degree, moved into a full-time position (research analyst, risk and emerging technologies [I don't know why, honestly]), got married (eloped lol), and got a new cat. I don't know where that leaves me, but this thing was a part of most of my nights. Idk. Thanks for reading.