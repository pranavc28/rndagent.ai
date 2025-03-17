from openai import OpenAI
import os

# export OPENAI_API_KEY=<your-api-key>
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class RNDAgent:
    def __init__(self):
        self.client = OpenAI()
        self.assistant = None
        self.vector_store = None
        
    def create_assistant(self, name, instructions):
        # Create assistant with file search capability
        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model="gpt-3.5-turbo",
            tools=[{"type": "file_search"}]
        )
        return assistant

    def create_vector_store(self, name, file_paths):
        # Create a vector store
        vector_store = self.client.vector_stores.create(name=name)
        
        # Ready the files for upload
        file_streams = [open(path, "rb") for path in file_paths]
        
        # Upload files and add them to vector store
        file_batch = self.client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )
        print(file_batch.status)
        print(file_batch.file_counts)
        return vector_store

    def update_assistant_with_vector_store(self, assistant_id, vector_store_id):
        # Update assistant to use the vector store
        assistant = self.client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        return assistant

    def create_thread_with_question(self, question, file_path=None):
        # If a file is provided, upload it and attach to the message
        message_params = {
            "role": "user",
            "content": question
        }
        
        if file_path:
            message_file = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="assistants"
            )
            message_params["attachments"] = [
                {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
            ]
        
        # Create thread with the message
        thread = self.client.beta.threads.create(
            messages=[message_params]
        )
        return thread

    def get_response(self, thread_id, assistant_id, stream=True):
        if stream:
            # Stream the response
            with self.client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id
            ) as stream:
                stream.until_done()
                
            # Get messages after run is complete
            messages = list(self.client.beta.threads.messages.list(thread_id=thread_id))
            return self._format_response(messages[0])
        else:
            # Create run and wait for completion
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            
            # Get messages after run is complete
            messages = list(self.client.beta.threads.messages.list(
                thread_id=thread_id,
                run_id=run.id
            ))
            return self._format_response(messages[0])

    def _format_response(self, message):
        # Format the response with citations
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text,
                f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = self.client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")
        
        return {
            "response": message_content.value,
            "citations": citations
        }

    def read_template(self, template_path: str, **kwargs) -> str:
        """
        Read and render a template file with the given variables
        
        Args:
            template_path (str): Path to the template file
            **kwargs: Variables to render in the template
        
        Returns:
            str: Rendered template content
        """
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            return template_content
        except Exception as e:
            print(f"Error reading template {template_path}: {str(e)}")
            return ""

    def run_with_templates(self, 
                         instructions_path: str = "instructions.tmpl",
                         question_and_policy_path: str = "question_with_policy.txt",
                         file_path: str = 'documents',
                         assistant_name: str = "Sustainability Research Assistant"):
        """
        Run the agent using template files for instructions and questions
        
        Args:
            instructions_path (str): Path to instructions template
            question_path (str): Path to question template
            file_paths (list[str]): List of paths to files to analyze
            assistant_name (str): Name of the assistant
        """
        # Get all PDF and text files from the documents directory
        documents_dir = os.path.join(os.path.dirname(__file__), file_path)
        file_paths = [
            os.path.join(documents_dir, f) for f in os.listdir(documents_dir)
            if f.endswith(('.pdf', '.txt', '.md', '.docx'))
        ]
        
        if not file_paths:
            print("Warning: No document files found in documents directory")
            file_paths = []
        
        # Read instructions template
        prompt_path = os.path.join(os.path.dirname(__file__), instructions_path)

        instructions = self.read_template(prompt_path)
        
        # Read question template
        question_path = os.path.join(os.path.dirname(__file__), question_and_policy_path)
        question = self.read_template(question_path)
        
        return self.run(
            question=question,
            file_paths=file_paths,
            assistant_name=assistant_name,
            instructions=instructions
        )

    def run(self, question: str, file_paths: list[str], assistant_name: str = "Research Assistant", instructions: str = ""):
        """
        Run the agent by:
        1. Creating an assistant if not exists
        2. Creating a vector store with the provided files
        3. Updating the assistant with the vector store
        4. Creating a thread with the question
        5. Getting the response
        
        Args:
            question (str): The question to ask
            file_paths (list[str]): List of paths to files to analyze
            assistant_name (str): Name of the assistant
            instructions (str): Instructions for the assistant
        
        Returns:
            dict: Response and citations
        """
        try:
            # Create assistant if not exists
            if not self.assistant:
                self.assistant = self.create_assistant(assistant_name, instructions)
            
            # Create vector store with documents
            self.vector_store = self.create_vector_store(
                name=f"{assistant_name}_store",
                file_paths=file_paths
            )
            
            # Update assistant with vector store
            self.assistant = self.update_assistant_with_vector_store(
                self.assistant.id,
                self.vector_store.id
            )
            
            # Create thread with question
            thread = self.create_thread_with_question(question)
            
            # Get response
            return self.get_response(thread.id, self.assistant.id)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "citations": []
            }



if __name__ == "__main__":
    agent = RNDAgent()
    response = agent.run_with_templates()
    
    print("\nResponse:", response["response"])
    print("\nCitations:")
    for citation in response["citations"]:
        print(citation)
