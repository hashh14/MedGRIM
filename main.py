from diagnosis_agent import diagnose
from evaluation_agent import evaluate
from chat_agent import chat

def main():
    query = input("Enter Your Query: ")
    img = input("Enter Image Path: ")
    
    # Process the query
    # Returns 3 dictionaries: disease_data, disease_questions, qa_responses
    # Function iteratively asks questions to the user.
    disease_data, disease_questions, qa = diagnose(query)
    
    #print("\nNumber of diseases analyzed:", len(disease_questions))

    # Returns a dictionary where the values are diagnosis strings
    evaluations = evaluate(qa, disease_data)

    chat(evaluations, disease_data)

if __name__ == "__main__":
    main()


