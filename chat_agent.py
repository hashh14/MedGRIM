from langchain_ollama import OllamaLLM
from typing import Dict, List
import logging
import json

def chat(evaluation_results: Dict[str, str], disease_data: Dict[str, Dict]) -> None:
    """
    Main function that runs the diagnostic analyzer with the provided evaluation results
    and disease data. Enables interactive analysis of medical diagnostic results.
    
    Args:
        evaluation_results (Dict[str, str]): Dictionary containing evaluation scores and analysis for each disease
        disease_data (Dict[str, Dict]): Dictionary containing reference information about each disease
    """
    def parse_disease_name(disease_key: str) -> str:
        """
        Parse the disease name from the string representation of a list.
        
        Args:
            disease_key (str): The string key containing [score, disease_name]
            
        Returns:
            str: A cleaned and formatted disease name
        """
        try:
            # Convert string representation of list to actual list using json
            disease_list = json.loads(disease_key.replace("'", '"'))
            # Get disease name from second element and format it
            disease_name = disease_list[1].replace('-', ' ').title()
            return disease_name
            
        except Exception as e:
            #logging.warning(f"Error parsing disease name from key '{disease_key}': {str(e)}")
            return disease_key  # Return the original key if parsing fails

    def print_evaluation_scores(evaluation_results: Dict[str, str]) -> str:
        """Print the evaluation scores and return them as a formatted string."""
        output = "\n=== Initial Diagnostic Results ===\n"
        llm = OllamaLLM(model="mistral")
        
        for disease_key, evaluation_text in evaluation_results.items():
            try:
                disease_name = parse_disease_name(disease_key)
                
                # Create a prompt for summarizing this disease's evaluation
                summary_prompt = f"""Provide a very brief (2-3 sentences) summary of the following medical evaluation for {disease_name}. 
                Include only the most crucial findings and implications. Also, mention the percentage score that
                was mentioned in the evaluation.

                Evaluation text:
                {evaluation_text}
                """
                
                try:
                    summary = llm.invoke(summary_prompt).strip()
                except Exception as e:
                    #logging.error(f"Error getting summary for {disease_name}: {str(e)}")
                    summary = "Error generating summary"
                
                output += f"\nDisease: {disease_name}\n"
                output += f"Summary: {summary}\n"
                output += "-" * 80 + "\n"
                
            except Exception as e:
                #logging.error(f"Error processing disease entry {disease_key}: {str(e)}")
                continue
        
        print(output)
        return output

    def handle_user_query(query: str, conversation_history: List[str]) -> str:
        """Handle follow-up questions from the user."""
        llm = OllamaLLM(model="mistral")
        
        context = """You are a medical diagnostic expert assistant helping a patient understand 
        their diagnostic results. 
        You must not form your own diagnosis or make your own inferences based on the data given.
        The percentage score/liklihood score mentioned in the evaluation results is the final
        diagnosis, which has been arrived at by careful evaluation and analysis.
        You must treat the given diagnosis as the only reference, and answer user questions accordingly.
        Use the following evaluation results and disease information 
        to answer their question accurately and clearly.
        
        Previous conversation:
        """
        
        # Add conversation history
        for message in conversation_history[-5:]:  # Keep last 5 messages for context
            context += f"\n{message}"
            
        # Add evaluation results and disease information
        context += "\n\nDetailed Evaluation Results:"
        for disease_key, eval_text in evaluation_results.items():
            try:
                disease_info = disease_data.get(disease_key, {})
                disease_name = parse_disease_name(disease_key)
                
                context += f"\n\nDisease: {disease_name}"
                context += f"\nComplete Evaluation:\n{eval_text}"
                
                if disease_info:
                    context += "\n\nReference Disease Information:"
                    for section, content in disease_info.items():
                        context += f"\n{section}: {content}"
                    
            except Exception as e:
                #logging.error(f"Error adding disease information for {disease_key}: {str(e)}")
                continue
        
        prompt = context + f"""
        
        Patient Question: {query}
        
        Provide a clear, accurate answer that:
        1. Directly addresses the patient's question
        2. References relevant information from the diagnostic results if necessary
        3. Provides context about severity and implications when appropriate
        4. Recommends seeking professional medical advice when appropriate

        Answer:
        """
        
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            #logging.error(f"Error handling user query: {str(e)}")
            return "I apologize, but I encountered an error processing your question. Could you please rephrase it?"

    try:
        # Validate inputs
        if not evaluation_results or not disease_data:
            raise ValueError("Evaluation results or disease data is empty")
            
        # Set up logging
        #logging.basicConfig(level=logging.INFO)
        #logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Print and store evaluation scores
        initial_analysis = print_evaluation_scores(evaluation_results)
        
        # Start interactive conversation
        conversation_history = ["Initial Analysis: " + initial_analysis]
        
        print("\nYou can now ask questions about your diagnostic results. Type 'exit' to end the conversation.")
        
        while True:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the diagnostic analyzer. Please remember to consult with a healthcare professional for proper medical advice.")
                break
                
            response = handle_user_query(user_input, conversation_history)
            print("\n" + response)
            
            # Update conversation history
            conversation_history.extend([
                f"Patient: {user_input}",
                f"Assistant: {response}"
            ])
            
    except Exception as e:
        #logging.error(f"Error in diagnostic analyzer: {str(e)}")
        print("Error: An unexpected error occurred while running the diagnostic analyzer.")
        raise



