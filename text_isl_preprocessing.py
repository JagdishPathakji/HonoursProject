import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from helper.railway_dictionary import RAILWAY_IDS

class RailwaysAnnouncementPreprocessor:
   def __init__(self, dictionary=RAILWAY_IDS):
      self.multi_word_list = [key for key, value in dictionary.items() if len(key.split()) > 1]
      self.stop_words = [
         "a", "an", "the","by", 
         "is", "am", "are", "was", "were", "be", "been", "being", 
         "do", "does", "did", "has", "have", "had", "will", "shall", 
         "would", "should", "can", "could", "may", "might", "must",
         "and", "or", "nor", "so", "for", "yet", 
         "oh", "uh", "um", "ah", "wow", 
      ]
      self.list = None
      self.number_words = {
      "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
      "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
      "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
      "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
      "eighteen": 18, "nineteen": 19, "twenty": 20
      }
      # self.llm = ChatOpenAI(model='gpt-4o-mini')
      # self.llm = ChatGroq(model='llama-3.1-70b-versatile')
      self.llm = ChatGroq(model='openai/gpt-oss-120b')

      
   def multi_word_matcher(self,sentence):
      prompt_template = PromptTemplate.from_template("""
      ### Debug Prompt:
      You are a debugging assistant. Your task is to **identify exact multi-word phrase matches** from the input list and output those matches.  

      ### Inputs:
      - Sentence: `{sentence}`  
      - List: `{list}`  

      ### Steps:  
      1. **Match Multi-Word Phrases**:
         - Check for exact matches from the list in the sentence.
         - Extract all multi-word phrases from the list that are present in the sentence as-is.  

      2. **Return Matches**:
         - Output the matched multi-word phrases in a **list format** ONLY and not in code format if they are present in the list.
         - If no matches are found, return an empty list and NOTHING ELSE.
 
      ### Example:  

      #### Input:  
      Sentence: "Attention all, train no. 1675, Rajdhani from platform nine B is leaving from Andhra Pradesh at 12:45."   

      #### Output:  
      ['andhra pradesh'] (because it's present in the list)
      
      ### Input : "Listen please, train number 173 destination to mumbai started from chennai arrive at platform number 3"
      
      ### Output: []
      
      NO PRELUDE OR EXPLANATION. Just return the matched multi-word phrases in a list format.
      """)
      prompt = prompt_template.invoke({
         'sentence': sentence,
         'list': str(self.multi_word_list),
      })
      
      response = self.llm.invoke(prompt)
      self.list = eval(response.content)
      return eval(response.content)
    
   def preprocess(self, sentence):
      sentence = " ".join([word.lower() for word in sentence.split() if word.lower() not in self.stop_words])
      self.multi_word_matcher(sentence)
      
      prompt_template = PromptTemplate.from_template("""
      You are a highly advanced text preprocessing assistant for Indian Sign Language (ISL) translation. Your task is to process railway announcements into context-aware tokens suitable for ISL video generation. Follow these steps precisely and **STRICTLY ADHERE** to the instructions.

      ### NOTE:
      1. Stop words have already been removed from the sentence. **Do NOT remove stop words again.**
      2. **Do NOT ignore or skip any words under any circumstances.**

      ### Inputs:
      - Sentence: {sentence}
      - List: {list} (Preserve the words in list as they are in the output, they must be treated as a single token, refer the example to understand better).
      
      ### Steps:

      1. **Lemmatize Tokens**:
         - Perform **lemmatization using part-of-speech tagging (POS)**.
         - Ensure all tokens are lemmatized to their root forms based on their context and part of speech. 
         - The word 'I' (first person pronoun) should be lemmatized to 'me' always.
         - Examples:
         - "leaving/left" → "leave"
         - "platforms" → "platform"
         - "assisting" → "assist"
         - "requested" → "request"
         - "departing/departed" → "depart"
         - "arriving" → "arrive"
         - "flies" → "fly" (if referring to the verb).
         - **DO NOT SKIP lemmatization for any words. Each word MUST be processed.**

      2. **Named Entity Recognition (NER)**:
         - Extract:
         - Train Numbers (e.g., "train no. 1675" → `train`, `number`, `1`, `6`, `7`, `5`).
         - Platform Names (e.g., "platform 9B" → `platform`, `9`, `B`).
         - Times (e.g., "12:45" → `1`, `2`, `4`, `5`).
         - These entities must be tokenized exactly as specified and not altered or capitalized.

      3. **Normalize Time and Numbers**:
         - Break time into individual digits.
         - Replace abbreviations with their expanded forms:
         - Example: "no." → "number", "P/F" → "platform".
         - Alphabets which are part of the token should be retained as they are.
         - Example: "9 B" → `9`, `B`.

      ### Example Inputs and Outputs:
      Input:  
         Sentence: "attention all train.no 1675 from platform 9B is leaving from andhra pradesh."
      Wrong Output:  
         ['attention', 'all', 'train','number', '1', '6', '7', '5', 'from', 'platform', '9', 'B','leave', 'from', 'andhra', 'pradesh']
      Correct Output:  
         ['attention', 'all', 'train','number','1', '6', '7', '5', 'from', 'platform', '9', 'B','leave', 'from', 'andhra pradesh']

      ### Important:
      - **NO PREAMBLE OR EXPLANATIONS** in your response.
      - Only return the processed tokens in a **list format**.
      - You are not to return code.
      - Any deviation will result in severe penalties.

      Now process the sentence according to these instructions.
      """)
      
      prompt = prompt_template.invoke({
         'sentence': sentence,
         'list': str(self.list),
      })
      
      response = self.llm.invoke(prompt)
      keys = eval(response.content)
      final_keys = []
      for key in keys:
        if key in RAILWAY_IDS.keys():
            final_keys.append(key)
        elif key in self.number_words.keys():
            final_keys.append(str(self.number_words[key]))
        elif key == ' ':
            continue
        else:
            for letter in key:
               final_keys.append(letter.upper())
      return final_keys

if __name__ == "__main__":
   load_dotenv()
   preprocessor = RailwaysAnnouncementPreprocessor()
   sentence = "sorry to inform train number 1 2 3 is arrived for 2 hour"
   print(preprocessor.preprocess(sentence))