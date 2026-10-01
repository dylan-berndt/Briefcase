# About

Font Search finds fonts from a description of how they look. Type a few words, such as *elegant script* or *chunky
rounded display*, and you get specimen images of the closest matches. You can keep paging for as long as you like;
there is no cut-off.

## How a search works

A model looks at the letters of every font and predicts which style tags describe it: things like `serif`, `script`,
`rounded`, `grunge` or `art-deco`. Your description is turned into the same kind of tags, and the fonts are ranked by how
well their predicted tags match yours.

### The tag line

Under the search box you see the tags your words were understood as. Each has a box:

- A ticked box means the tag is part of the search.
- An empty box means it is left out. Click to flip it, and the results change straight away.
- Words the site did not recognise but could guess at get ticked tags of their own, with a lower weight.
- Close synonyms of words it did not know are offered as empty boxes. Tick one to add it.

Negation is not supported: "not thin" does not exclude thin fonts, it just adds nothing.

### Writing a good description

- Short and visual works best: a few style words rather than a sentence.
- Words for the *look* (`slab`, `condensed`, `handwritten`) match better than words for the *use* (`wedding`, `startup`).
- If the first results are off, untick the tag that is pulling them in the wrong direction.

## Feedback

You can log in to tell the site how it did.

- **Thumbs up or down** says whether a font answered your search. It is saved with your description and the tags that
  were ticked at the time.
- **Rating** is separate: it is about the font itself, whether it is a good font at all.

It is stored so the matching can be checked against what people actually wanted.

## Where the fonts come from

The fonts are the free ones from DaFont and Google Fonts. The site only shows specimen images and links to the font's
own page, where its licence and download are. It does not host or serve the font files.

## The model

The tags come from a vision model pretrained to reconstruct letters from other letters, which pushes it to learn
a font's style rather than the shape of any single glyph, and then trained to predict style tags from the
[Large-scale Tag-based Font Retrieval](https://arxiv.org/abs/1909.02072) dataset. Search ranks fonts on the
tag probabilities it produces; the model itself does not run on the server.

## Limits

Tags describe the general look of a font, not its exact identity. Expect a good neighbourhood of similar fonts, not
one precise answer, and expect rare styles to be matched less reliably than common ones.
