from langchain_ollama import OllamaLLM
import json
from typing import Dict, List, Tuple
import logging
import re

def evaluate(qa_responses: Dict[str, List[Dict[str, str]]], 
                             disease_data: Dict[str, Dict[str, str]]) -> Dict[str, Dict]:
    """
    Process Q&A responses and disease data to evaluate likelihood of each disease.
    
    Args:
        qa_responses (Dict): Dictionary of Q&A responses from the diagnostic questions
        disease_data (Dict): Dictionary of disease reference data
        
    Returns:
        Dict[str, Dict]: Dictionary mapping disease keys to evaluation results containing:
            - score: float (0-100)
            - justification: str
            - matching_symptoms: List[str]
            - discrepancies: List[str]
            
    Side effects:
        - Creates 'evaluation_scores.json' with complete evaluation results
    """
    def llm_answer_evaluator(qa_responses: Dict[str, List[Dict[str, str]]], 
                           disease_data: Dict[str, Dict[str, str]]) -> Dict[str, Dict]:
        llm = OllamaLLM(model="mistral")
        evaluation_scores = {}
        
        for disease_key, qa_pairs in qa_responses.items():
            similarity_score, disease_name = eval(disease_key)
            disease_info = disease_data[disease_key]
            
            context = f"""You are an expert medical diagnostic evaluator. Your task is to evaluate a set of 
            patient responses to diagnostic questions for {disease_name} and determine the likelihood 
            (as a percentage) that the patient has this condition based on the reference disease information given.
            You must not provide any extra information, and only use knowledge from the reference data for evaluation.

            Important Context:
            1. This disease has a {similarity_score*100:.1f}% match with images of the patient's symptoms
            2. You have access to comprehensive reference information about the disease
            3. Each question and answer should be evaluated against this reference material
            4. Consider both positive and negative indicators in the responses
            
            Reference Disease Information:
            """
            
            for section, content in disease_info.items():
                context += f"\n{section.upper()}:\n{content}\n"
                
            context += "\nPatient's Q&A Responses:\n"
            for qa in qa_pairs:
                context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
                
            prompt = context + """
            Based on the above information, generate responses only about the below given points one by one in a brief and concise manner:
            1. A percentage score (0-100) indicating how well the patient's symptoms match this disease
            2. A brief justification for your scoring
            3. Key matching symptoms found in responses
            4. Key discrepancies or missing symptoms
            
            """
            
            try:
                response = llm.invoke(prompt)
                evaluation_scores[disease_key] = response
                #logging.info(f"\nEvaluation for {disease_name} completed")
                    
            except Exception as e:
                #logging.error(f"Error evaluating {disease_name}: {str(e)}")
                evaluation_scores[disease_key] = f"Error: {str(e)}"
        
        return evaluation_scores

    def save_to_json(data: Dict, filename: str):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            #logging.info(f"Successfully saved data to {filename}")
        except Exception as e:
            #logging.error(f"Error saving to {filename}: {str(e)}")
            logging.error("")

    # Set up logging
    #logging.basicConfig(level=logging.INFO)
    #logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Run evaluation
    evaluation_scores = llm_answer_evaluator(qa_responses, disease_data)
    
    # Save results
    save_to_json(evaluation_scores, 'evaluation_scores.json')
    
    return evaluation_scores