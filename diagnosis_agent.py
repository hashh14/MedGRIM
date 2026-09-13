from graph_retrieval import query_disease_graph
from langchain_ollama import OllamaLLM
import json
from typing import Dict, List, Tuple
import logging
import re

def diagnose(query: str) -> Tuple[Dict, Dict]:
    """
    Process a disease-related query to generate diagnostic questions and collect answers.
    
    Args:
        query (str): User's initial query about symptoms
        
    Returns:
        Tuple[Dict, Dict]: Tuple containing two dictionaries:
            1. disease_questions: Mapping of diseases to diagnostic questions
            2. qa_responses: Mapping of diseases to question-answer pairs
            
    Side effects:
        - Creates 'disease_questions.json' with generated questions
        - Creates 'qa.json' with question-answer pairs
    """
    def llm_question_generator(disease_data: Dict, disease_key: str) -> List[str]:
        llm = OllamaLLM(model="mistral")
        
        similarity_score, disease_name = eval(disease_key)
        
        prompt = f"""You are a medical diagnostic expert. Your task is to analyze the provided disease information 
        and generate precise diagnostic questions. These questions must collectively allow for a conclusive diagnosis 
        of {disease_name} when answered by a patient.

        Analysis Guidelines:
        1. Consider all provided information about symptoms, causes, and risk factors
        2. Focus on distinctive characteristics that differentiate this condition from similar ones
        3. Include questions about severity, duration, and pattern of symptoms
        4. Consider risk factors and predisposing conditions
        5. Include both primary symptoms and secondary indicators

        Disease Information:
        Name: {disease_name}
        """

        for key, value in disease_data.items():
            prompt += f"\n{key.title()}:\n{value}\n"

        prompt += """
        Based on the above information, generate a list of diagnostic questions. Generate a maximum
        of 5 questions to reach a conclusive diagnosis. Each question should:
        - Be specific and unambiguous
        - Require clear yes/no answers or quantifiable responses
        - Focus on one aspect at a time
        - Help differentiate this condition from similar conditions
        - Cover both primary symptoms and risk factors

        """

        try:
            response = llm.invoke(prompt)
            
            # Split the response into questions using regex
            questions = []
            # Match numbered items (1. 2. 3. etc.) followed by text until the next number or end
            matches = re.finditer(r'(?m)^\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|$)', response, re.DOTALL)
            
            for match in matches:
                # Clean up the question text
                question = match.group(1).strip()
                if question:  # Only add non-empty questions
                    questions.append(question)
            
            if not questions:
                raise ValueError("No valid questions found in response")
                
            return questions
        
        except Exception as e:
            #logging.error(f"Error generating questions for {disease_name}: {str(e)}")
            #logging.error(f"Raw response: {response}")
            return []

    def process_disease_questions(results: Dict) -> Dict[str, List[str]]:
        questions_dict = {}
        
        #logging.basicConfig(level=logging.INFO)
        #logging.getLogger("httpx").setLevel(logging.WARNING)
        
        for key, details in results.items():
            try:
                questions = llm_question_generator(disease_data=details, disease_key=key)
                questions_dict[key] = questions
                #logging.info(f"Generated {len(questions)} questions for {key}")
            except Exception as e:
                #logging.error(f"Error processing disease {key}: {str(e)}")
                questions_dict[key] = []
        
        return questions_dict

    def collect_user_answers(disease_questions: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        qa_dict = {}
        
        for disease_key, questions in disease_questions.items():
            disease_name = disease_key.split(",")[1].strip().strip("[]'\" ")
            print(f"\nQuestions for {disease_name}:")
            qa_list = []
            
            for question in questions:
                print(f"\nQ: {question}")
                answer = input("Your answer: ").strip()
                qa_list.append({
                    "question": question,
                    "answer": answer
                })
            
            qa_dict[disease_key] = qa_list
        
        return qa_dict

    def save_to_json(data: Dict, filename: str):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            #logging.info(f"Successfully saved data to {filename}")
        except Exception as e:
            #logging.error(f"Error saving to {filename}: {str(e)}")
            logging.error("")

    # Main process
    disease_data = query_disease_graph(query)
    
    # Get questions for all diseases
    disease_questions = process_disease_questions(disease_data)
    
    # Save questions to JSON file
    save_to_json(disease_questions, 'disease_questions.json')
    
    # Collect user answers
    qa_responses = collect_user_answers(disease_questions)
    
    # Save Q&A responses to JSON file
    save_to_json(qa_responses, 'qa.json')
    
    # Print summary
    # print("\nProcessing complete!")
    # print(f"Generated questions for {len(disease_questions)} diseases")
    # print("Questions saved to: disease_questions.json")
    # print("Q&A responses saved to: qa.json")
    
    return disease_data, disease_questions, qa_responses
