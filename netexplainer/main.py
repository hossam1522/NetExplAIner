from dataset import Dataset
from llm import LLM

def main():
    dataset = Dataset('downloads')
    processed_files = dataset.process_files()
    llm = LLM(processed_files[0])
    #answer = llm.model.invoke({"question": "What is the total number of packets in the trace?"})
    messages = [
        (f"{llm.file[0].page_content[0:1000000]}"
        )
    ]
    answer = llm.model.invoke(messages)
    print(answer)
    #print(len(llm.file[0].page_content))

if __name__ == '__main__':
    main()