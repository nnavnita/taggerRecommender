# taggerRecommender
A tagger and recommender for web pages based on content.

Original usecase: to recommend related articles and products on a site.

## Dependencies

To install the dependencies:

```
pip install selenium flair==0.4.3 nltk pandas sklearn scipy tqdm torch
```

## Usage

To run:

```
python main.py [files with URLs].txt [result file].json
```

## Next Steps
- improve tagging
- incorporate date into recommendations
