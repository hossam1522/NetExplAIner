# NetExplAIner
Using LLMs to analyze network traces by decomposing queries into manageable steps and using mathematical external tools. The idea can be seen more clearly in the following schema:

![image](https://github.com/user-attachments/assets/acbefac2-6951-40e7-b7ad-e127092fa0b7)

The idea consists of investigating the improvement (or not) of the approach of dividing complex questions into manageable steps and providing the LLM model with an external mathematical tool to supply the areas where the LLMs are weaker, like doing math.

## How to use

1. Clone the repository.
2. Install `uv`:
   ```
   make install-uv
   ```
3. Modify the `netexplainer/data/questions.yaml` file by indicating what models you want to evaluate (the available models are located in `netexplainer/llm.py` at the end of the file, in the `models` structure) and if you want questions to be divided or not. Feel free to modify the subquestions that will be compared to. For example:
   ```yaml
    models:
      - "llama2-7b"
      - "mistral-7b"
    
    questions:
      - question: "What is the total number of packets in the trace?"
        divide_in_subquestions: True
        subquestions:
          - "What is the first packet number?"
          - "What is the last packet number?"
    
      - question: "How many unique communicators are present in the trace?"
        divide_in_subquestions: False
        subquestions:
          - "How many unique source IPs are there?"
          - "How many unique destination IPs are there?"
          - "Are there any IPs that are both source and destination?"
     ...
   ```
4. If you want to test another model not contemplated in the existing ones, feel free to modify and add the models you want, following the same format as the other models. You can use models by their API (like Gemini ones) or local models deployed by Ollama. You must indicate the LLM function to use and whether the context window is small or big.
5. If you want to modify the questions of the `netexplainer/data/questions.yaml` file, you should add the new questions and a way to answer them in the `netexplainer/llm.py` file. More specifically, you should add that information to the `__answer_question` function.
6. If you have some network traces to evaluate, put them in a folder called `raw/` located in `netexplainer/data/`. If not, you can download network traces from the [Wireshark Wiki](https://wiki.wireshark.org/samplecaptures) using the following command:
   ```
   make download-data
   ```
7. If you want to filter the network traces to a maximum number of packets per trace, you can indicate it with:
   ```
   make clean-data <N>
   ```
   Where `<N>` is the maximum number of traces. The filtered traces will be saved in the `netexplainer/data/cleaned` folder. If you don't want to filter, simply change the `raw/` folder name to `cleaned/`.
8. Finally, once you have all configured properly, execute the program by doing:
   ```
   make run
   ```

## Unit testing
To check the correct functioning of the project without the need to install all dependencies, a Docker container has been created to perform all the processes and check the unit tests located in the `tests/` folder.

To build and run the container locally, you can use the following command:
```bash
docker build -t netexplainer . && docker run -u 1001 -t -v "$(pwd):/app/test" netexplainer
```

In case of not want to build the image locally, you can pull and run the image directly from [DockerHub](https://hub.docker.com/repository/docker/hossam1522/netexplainer/general) by using the following command:
```bash
docker run -u 1001 -t -v "$(pwd):/app/test" hossam1522/netexplainer
```

If you want to test the code without using Docker, you should execute the following commands to install `uv` and then test:
```
make install-uv && make test
```

### Additional notes

- [LICENSE](LICENSE)
