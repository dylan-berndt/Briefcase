import random

class Description:
    tokenizer = None
    maxDescriptors = 7

    def __init__(self, name, adjectives, tags=None, plainText=None, fixed=True):
        tags = tags if tags is not None else {}
        
        self.name = name
        self.adjectives = adjectives
        self.tags = tags

        self.plainText = plainText if plainText is not None else ""

        self.fixedSample = self._sample()
        self.fixed = fixed

    def _sample(self):
        if self.tags is not None:
            tags = [tag for tag, value in self.tags.items() if random.uniform(0, 1) < value]

            descriptors = self.adjectives + tags
        else:
            descriptors = self.adjectives
        numDescriptors = int(random.uniform(0.6, 1.0) * Description.maxDescriptors)
        chosenDescriptors = random.sample(descriptors, min(len(descriptors), numDescriptors))

        joined = ", ".join(chosenDescriptors) + " font"

        return "a " + joined
    
    def sample(self):
        if self.fixed:
            return self.fixedSample
        return self._sample()

    def __len__(self):
        descriptors = self.adjectives + list(self.tags.keys())
        description = ", ".join(descriptors) + " font named " + self.name
        tokens = Description.tokenizer([description], padding=False, return_tensors="pt")
        return len(tokens["input_ids"][0])