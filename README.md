# AI Scripts

This repo contains various utility scripts to use AI to refresh an users index.

These scripts can be run by a one-click script provided in the repo. The [get-ai-script.py](./get-ai-script.py) will take care of pulling the correct script, verifying the dependencies and running it as well.

## Supported Scripts

As of now, the following scripts are supported

| Name                       | Description                              | Key        | Details |
| -------------------------- | ---------------------------------------- | ---------- | ------- | ----------------------------------------- |
| kNN indexing Script        | Re-Index the data with vector injection  | `knn`      |         | [Read more here](./knn_reindex/README.md) |
| Metadata Enrichment Script | Enrich the dataset by injecting metadata | `metadata` |         | [Read more here](./metadata/README.md)    |

**Get started with the script by doing the following**

> Based on the script needed, pass the Key from the above table in place of `<KEY>`

```sh
curl -s https://raw.githubusercontent.com/appbaseio/ai-scripts/master/get-ai-script.py --output get-ai.py && python3 get-ai.py "<KEY>"
```
