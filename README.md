# AI Scripts

This repo contains various utility scripts to use AI to refresh an users index.

These scripts can be run by a one-click script provided in the repo. The [get-ai-script.py](./get-ai-script.py) will take care of pulling the correct script, verifying the dependencies and running it as well.

```sh
curl -s https://raw.githubusercontent.com/appbaseio/ai-scripts/master/get-ai-script.py --output get-ai.py && python3 get-ai.py knn
```