from methods import text_scraper, lsa, sim
import json
import sys

# scraping the text from the pages of the URLs provided
text_scraper(open(sys.argv[1]).readlines(), sys.argv[2])

# performing latent semantic analysis on the title and content to extract top 15 keywords and add to file
lsa(json.load(open(sys.argv[1])), sys.argv[2])

# based on their keywords, determine related/similar products and articles for each product and article
sim(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])

