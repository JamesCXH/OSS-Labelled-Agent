# Web Agent w/ Action Labelling

## Development Notes

- Requires Python version 3.10.4
- Install dependencies using: `pip install -r requirements.txt`
- The code which produces the scraped action effects can be found at https://github.com/JamesCXH/Scaling-Web-Agents
- Due to severe time constraints, the last-mile cleaning/writeup for both this and the scrape code may be slow/indefinite.

## How to use
Get the dominos zip here, unzip it so that the path from root is just `dominos`. This zip contains all the scraped information of dominos needed for labelling the actions for the LLM.


https://drive.google.com/drive/folders/1lzrBo-2YVYiPb8_fBpeLwVxC3RXxd8Mw?usp=sharing


First export your API keys

`export ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY`

`export OPENAI_API_KEY=YOUR_OAI_KEY`

Then run the main file

`python main.py`


## Issues & contact
If there are any issues, email me at

jamesc3 [at] andrew [dot] cmu [dot] edu

or DM me (though due to spam I may miss it)

https://x.com/jchencxh

## Authors
James Chen

Cem Adatepe (https://cemadatepe.com/)
