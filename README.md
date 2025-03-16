# NetExplAIner
Using LLMs to analyze network traces by decomposing queries in manageable steps.

## Docker container for unit testing
To build and run the container locally, you can use the following command:
```bash
docker build -t netexplainer . && docker run -u 1001 -t -v "$(pwd):/app/test" netexplainer
```

In case of not want to build the image locally, you can pull and run the image directly from [DockerHub](https://hub.docker.com/repository/docker/hossam1522/netexplainer/general) by using the following command:
```bash
docker run -u 1001 -t -v "$(pwd):/app/test" hossam1522/netexplainer
```

### Additional notes

- License: [LICENSE](LICENSE)
