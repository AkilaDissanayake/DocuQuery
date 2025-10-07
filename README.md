##### DocuQuery (Document based question Answering App)



This app allows you to upload PDF or TXT documents, process them into embeddings, and ask questions about their content using OpenAI LLMs.

##### 

##### Installation and Setup (Once)



1\. Install Python



* &nbsp;   Windows: Download and install Python 3.10+ from [python.org](python.org)
* &nbsp;   Mac: Download and install Python 3.10+ from [python.org](python.org)
* &nbsp;   Linux: Install via package manager



2.Check installation by running python--version.

2.1 Install git if not installed.
&nbsp;       Go to: https://git-scm.com/downloads


3.Clone the repository use below commands in bash.

&nbsp;        git clone https://github.com/AkilaDissanayake/DocuQuery.git

&nbsp;        cd DocuQuery\DocuQuery_App


4.Open .env file and give your API key. (Optional)

5.Install dependencies run below command in bash.

&nbsp;    pip install -r requirements.txt



#### To run the app open terminal in DocuQuery\DocuQuery_App and run (On each run)

&nbsp;    streamlit run app.py



##### How to use



* Open the sidebar.
* Enter your OpenAI API key.
* Select the LLM model (default: gpt-4-turbo).
* Upload PDF/TXT files.
* Click Start Processing to extract text and create embeddings.
* Enter a question in the text area and click Get Answer.
* View the generated answer and previous Q\&A in the Chat History section.





##### Notes



* Embeddings are cumulative: Adding new files will not delete embeddings from previously uploaded files.
* Processing is required after uploading all files for embeddings to be created.
* Cache: Embeddings are temporarily stored in embeddings\_cache.npz and cleared when the app exits.



##### Troubleshooting



* No answer after clicking Get Answer: Ensure Start Processing was clicked after uploading files.
* OpenAI API errors: Verify your API key.
* PDF text not extracted: Some PDFs are scanned images; this app supports only text-based PDFs.













