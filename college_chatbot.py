import sqlite3
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
import re
import openai
from dotenv import load_dotenv
import os

load_dotenv()

class UserPreferences(BaseModel):
    """User preferences extracted from conversation using LangChain"""
    location: Optional[str] = Field(None, description="Preferred city or state for college")
    state: Optional[str] = Field(None, description="Preferred state for college")
    course_type: Optional[str] = Field(None, description="Type of course like Engineering, Medicine, Arts, Commerce, etc.")
    college_type: Optional[str] = Field(None, description="Government, Private, or Deemed university")
    level: Optional[str] = Field(None, description="UG (Undergraduate) or PG (Postgraduate)")
    budget_range: Optional[str] = Field(None, description="Budget preference like low, medium, high")
    specific_course: Optional[str] = Field(None, description="Specific course like BTech, MBA, MBBS, etc.")

@dataclass
class College:
    college_id: str
    name: str
    type: str
    affiliation: str
    location: str
    website: str
    contact: str
    email: str
    courses: str
    scholarship: str
    admission_process: str

class SimpleDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with simplified tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create chat_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                chat_id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'College Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                message_type TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chat_sessions (chat_id)
            )
        ''')
        
        # Create preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_preferences (
                chat_id TEXT PRIMARY KEY,
                preferences TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chat_sessions (chat_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_chat_session(self, chat_id: str):
        """Create a new chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO chat_sessions (chat_id) VALUES (?)',
            (chat_id,)
        )
        conn.commit()
        conn.close()
    
    def save_message(self, chat_id: str, message_type: str, content: str):
        """Save a message to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure chat session exists
        cursor.execute('INSERT OR IGNORE INTO chat_sessions (chat_id) VALUES (?)', (chat_id,))
        
        # Save message
        cursor.execute(
            'INSERT INTO messages (chat_id, message_type, content) VALUES (?, ?, ?)',
            (chat_id, message_type, content)
        )
        
        # Update session timestamp
        cursor.execute(
            'UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?',
            (chat_id,)
        )
        
        conn.commit()
        conn.close()
    
    def get_chat_messages(self, chat_id: str) -> List[Dict]:
        """Retrieve all messages for a chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message_type, content, timestamp 
            FROM messages 
            WHERE chat_id = ? 
            ORDER BY timestamp
        ''', (chat_id,))
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'type': msg[0],
                'content': msg[1],
                'timestamp': msg[2]
            }
            for msg in messages
        ]
    
    def save_preferences(self, chat_id: str, preferences: dict):
        """Save chat preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO chat_preferences (chat_id, preferences) VALUES (?, ?)',
            (chat_id, json.dumps(preferences))
        )
        conn.commit()
        conn.close()
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get chat preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT preferences FROM chat_preferences WHERE chat_id = ?',
            (chat_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}

class CollegeDataManager:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.colleges = self.load_college_data()
    
    def load_college_data(self) -> List[College]:
        """Load college data from Excel file"""
        try:
            df = pd.read_excel(self.excel_path)
            colleges = []
            
            for _, row in df.iterrows():
                college = College(
                    college_id=str(row.get('College ID', '')),
                    name=str(row.get('College', '')),
                    type=str(row.get('Type', '')),
                    affiliation=str(row.get('Affiliation', '')),
                    location=str(row.get('Location', '')),
                    website=str(row.get('Website', '')),
                    contact=str(row.get('Contact', '')),
                    email=str(row.get('E-mail', '')),
                    courses=str(row.get('Courses (ID, Category, Duration, Eligibility, Language, Accreditation, Fees)', '')),
                    scholarship=str(row.get('Scholarship', '')),
                    admission_process=str(row.get('Admission Process', ''))
                )
                colleges.append(college)
            
            print(f"Loaded {len(colleges)} colleges from Excel file")
            return colleges
        except Exception as e:
            print(f"Error loading Excel data: {e}")
            return []
    
    def filter_colleges_by_preferences(self, preferences: UserPreferences) -> List[Dict]:
        """Filter colleges based on user preferences"""
        matching_colleges = []
        
        for college in self.colleges:
            match_score = 0
            match_reasons = []
            missing_criteria = []
            
            # Location filtering
            location_match = False
            if preferences.location:
                location_terms = [preferences.location.lower()]
                if preferences.state:
                    location_terms.append(preferences.state.lower())
                
                college_location = college.location.lower()
                for term in location_terms:
                    if term in college_location:
                        location_match = True
                        match_score += 30
                        match_reasons.append(f"Located in {preferences.location}")
                        break
                
                if not location_match:
                    missing_criteria.append(f"Not in preferred location: {preferences.location}")
                    continue
            
            # College type filtering
            if preferences.college_type:
                if preferences.college_type.lower() in college.type.lower():
                    match_score += 25
                    match_reasons.append(f"Matches college type: {preferences.college_type}")
                else:
                    missing_criteria.append(f"Not a {preferences.college_type} college")
                    continue
            
            # Course type filtering
            course_match = False
            if preferences.course_type or preferences.specific_course:
                college_courses = college.courses.lower()
                
                if preferences.specific_course:
                    course_terms = [preferences.specific_course.lower()]
                    if preferences.course_type:
                        course_terms.append(preferences.course_type.lower())
                else:
                    course_terms = [preferences.course_type.lower()]
                
                for term in course_terms:
                    if term in college_courses:
                        course_match = True
                        match_score += 25
                        match_reasons.append(f"Offers {term} courses")
                        break
                
                if not course_match:
                    missing_criteria.append(f"Doesn't offer preferred course type")
                    continue
            
            # Level filtering
            if preferences.level:
                if preferences.level.lower() in college.courses.lower():
                    match_score += 10
                    match_reasons.append(f"Offers {preferences.level} programs")
            
            # Only add colleges that meet the strict criteria
            if match_score > 0:
                matching_colleges.append({
                    'college': college,
                    'score': match_score,
                    'reasons': match_reasons,
                    'missing': missing_criteria
                })
        
        # Sort by match score
        matching_colleges.sort(key=lambda x: x['score'], reverse=True)
        return matching_colleges[:5]  # Return top 5 matches

class CollegeRecommendationChain:
    def __init__(self, api_key: str, excel_path: str, db_path: str):
        self.api_key = api_key
        openai.api_key = api_key
        
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        
        self.db_manager = SimpleDatabaseManager(db_path)
        self.data_manager = CollegeDataManager(excel_path)
        self.conversation_chains = {}  # Store conversation chains per chat
        
        # Initialize preference extraction chain
        self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        self.preference_chain = self._create_preference_extraction_chain()
        
        # Initialize recommendation detection chain
        self.recommendation_detector = self._create_recommendation_detector()
        
        # System prompt for the conversation
        self.system_prompt = """
You are a helpful college recommendation assistant for Indian colleges and universities. Your role is to:

1. Have natural, friendly conversations with users about their educational interests and preferences
2. Ask clarifying questions to understand their needs better
3. Extract information about their preferred location, course type, college type, etc.
4. ONLY provide college recommendations when the user explicitly asks for suggestions or recommendations
5. When recommending, use the provided college database and explain why each college matches their preferences
6. Be encouraging and supportive in your responses

Key Guidelines:
- Do NOT recommend colleges until the user specifically asks for recommendations
- Ask follow-up questions to better understand their preferences
- Be conversational and helpful
- If you don't have colleges in their preferred location in the database, mention this clearly
- Always explain your reasoning for recommendations

Remember: Wait for the user to ask for recommendations before providing them!
"""
    
    def _create_preference_extraction_chain(self):
        """Create chain for extracting user preferences"""
        preference_prompt = PromptTemplate(
            template="""
            Extract user preferences for college search from the following conversation history.
            Look for mentions of:
            - Location/City/State (like "Indore", "MP", "Delhi", "Bangalore", etc.)
            - Course types (like "Engineering", "Medical", "Commerce", "Arts", "Management")
            - Specific courses (like "BTech", "MBA", "MBBS", "BCom")
            - College types (like "Government", "Private", "Deemed")
            - Level (like "UG", "PG", "Undergraduate", "Postgraduate")
            - Budget preferences

            Conversation History:
            {conversation_history}

            Current Message:
            {current_message}

            {format_instructions}

            Extract preferences as JSON. If no clear preference is mentioned, use null for that field.
            """,
            input_variables=["conversation_history", "current_message"],
            partial_variables={"format_instructions": self.preference_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=preference_prompt)
    
    def _create_recommendation_detector(self):
        """Create chain for detecting recommendation requests"""
        detector_prompt = PromptTemplate(
            template="""
            Determine if the user is asking for college recommendations or suggestions.
            
            User message: "{message}"
            
            Return "YES" if the user is asking for college recommendations, suggestions, or wants to know about colleges.
            Return "NO" if they are just having a conversation or asking general questions.
            
            Keywords that indicate recommendation request:
            - "recommend", "suggest", "which college", "best college", "good college"
            - "colleges for", "help me find", "looking for", "options"
            - "where should I", "what colleges", "any suggestions"
            
            Answer with only YES or NO.
            """,
            input_variables=["message"]
        )
        
        return LLMChain(llm=self.llm, prompt=detector_prompt)
    
    def _create_conversation_chain(self, chat_id: str):
        """Create a conversation chain with memory for a chat session"""
        memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True
        )
        
        # Load previous messages
        previous_messages = self.db_manager.get_chat_messages(chat_id)
        for msg in previous_messages[-10:]:  # Load last 10 messages
            if msg['type'] == 'human':
                memory.chat_memory.add_user_message(msg['content'])
            elif msg['type'] == 'ai':
                memory.chat_memory.add_ai_message(msg['content'])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        conversation = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=memory,
            verbose=False
        )
        
        return conversation
    
    def _extract_preferences_with_llm(self, chat_id: str, current_message: str) -> UserPreferences:
        """Extract user preferences using LangChain"""
        try:
            # Get conversation history
            messages = self.db_manager.get_chat_messages(chat_id)
            conversation_history = "\n".join([
                f"{msg['type'].title()}: {msg['content']}" for msg in messages[-10:]
            ])
            
            # Extract preferences using LLM
            result = self.preference_chain.run(
                conversation_history=conversation_history,
                current_message=current_message
            )
            
            # Parse the result
            try:
                preferences = self.preference_parser.parse(result)
                
                # Save preferences to database
                pref_dict = preferences.dict()
                self.db_manager.save_preferences(chat_id, pref_dict)
                
                return preferences
            except OutputParserException as e:
                print(f"Parser error: {e}")
                # Try to fix the output
                fixing_parser = OutputFixingParser.from_llm(parser=self.preference_parser, llm=self.llm)
                preferences = fixing_parser.parse(result)
                return preferences
                
        except Exception as e:
            print(f"Error extracting preferences: {e}")
            # Return previous preferences if available
            prev_prefs = self.db_manager.get_preferences(chat_id)
            if prev_prefs:
                return UserPreferences(**prev_prefs)
            return UserPreferences()
    
    def _is_asking_for_recommendations(self, user_input: str) -> bool:
        """Use LangChain to detect recommendation requests"""
        try:
            result = self.recommendation_detector.run(message=user_input)
            return "YES" in result.upper()
        except Exception as e:
            print(f"Error in recommendation detection: {e}")
            # Fallback to keyword matching
            recommendation_keywords = [
                'recommend', 'suggest', 'colleges', 'universities', 'which college',
                'best college', 'good college', 'college for', 'options for',
                'where should i', 'help me find', 'looking for college', 'any college',
                'colleges in', 'suggest college', 'show me college'
            ]
            
            user_input_lower = user_input.lower()
            return any(keyword in user_input_lower for keyword in recommendation_keywords)
    
    def _get_openai_college_recommendations(self, preferences: UserPreferences, location: str = None) -> List[Dict]:
        """Get college recommendations from OpenAI for specific locations"""
        try:
            # Build preference description
            pref_parts = []
            if location:
                pref_parts.append(f"Location: {location}")
            if preferences.course_type:
                pref_parts.append(f"Course type: {preferences.course_type}")
            if preferences.specific_course:
                pref_parts.append(f"Specific course: {preferences.specific_course}")
            if preferences.college_type:
                pref_parts.append(f"College type: {preferences.college_type}")
            if preferences.level:
                pref_parts.append(f"Level: {preferences.level}")
            
            preference_text = ", ".join(pref_parts) if pref_parts else "General preferences"
            
            prompt = f"""
            Recommend 5 good colleges/universities in India based on these preferences: {preference_text}
            
            Focus on well-known, reputable institutions. If location is specified, prioritize colleges in that area.
            
            Provide response as a JSON array with this exact structure:
            [
                {{
                    "name": "College Name",
                    "location": "City, State",
                    "type": "Government/Private/Deemed",
                    "courses_offered": "Relevant courses offered",
                    "website": "Official website if known or N/A",
                    "admission_process": "Brief admission process",
                    "approximate_fees": "Fee range if known",
                    "notable_features": "Any notable features or rankings"
                }}
            ]
            
            Return only the JSON array, no additional text.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # Extract JSON if there's additional text
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return []
                
        except Exception as e:
            print(f"Error getting OpenAI recommendations: {e}")
            return []
    
    def _format_college_recommendations(self, filtered_colleges: List[Dict], openai_colleges: List[Dict]) -> Dict:
        """Format college recommendations with detailed explanations"""
        recommendations = []
        
        # Add filtered colleges from database
        for item in filtered_colleges:
            college = item['college']
            rec = {
                "college_id": college.college_id,
                "name": college.name,
                "type": college.type,
                "affiliation": college.affiliation,
                "location": college.location,
                "website": college.website,
                "contact": college.contact,
                "email": college.email,
                "courses": college.courses,
                "scholarship": college.scholarship,
                "admission_process": college.admission_process,
                "match_score": item['score'],
                "match_reasons": item['reasons'],
                "source": "database"
            }
            recommendations.append(rec)
        
        # Add OpenAI colleges if needed
        if len(recommendations) < 3 and openai_colleges:
            needed = min(5 - len(recommendations), len(openai_colleges))
            for college in openai_colleges[:needed]:
                college["source"] = "openai_knowledge"
                college["match_score"] = 75  # Default score for OpenAI recommendations
                recommendations.append(college)
        
        return {"college_recommendations": recommendations}
    
    def process_message(self, chat_id: str, message: str) -> Dict:
        """
        Main method to process a message and return response
        This is the single entry point for the chain
        
        Args:
            chat_id: Unique identifier for the chat session
            message: User's input message
            
        Returns:
            Dict containing success status and response/recommendations
        """
        try:
            # Save user message
            self.db_manager.save_message(chat_id, 'human', message)
            
            # Extract preferences from current conversation
            preferences = self._extract_preferences_with_llm(chat_id, message)
            
            # Check if user is asking for recommendations
            if self._is_asking_for_recommendations(message):
                print(f"Recommendation request detected. Preferences: {preferences}")
                
                # Filter colleges from database
                filtered_colleges = self.data_manager.filter_colleges_by_preferences(preferences)
                
                if not filtered_colleges and preferences.location:
                    # No colleges found in database, get from OpenAI
                    openai_colleges = self._get_openai_college_recommendations(preferences, preferences.location)
                    
                    if openai_colleges:
                        final_response = f"I don't have specific colleges for {preferences.location} in my database. Here are some well-known institutions in that area:"
                        recommendations = self._format_college_recommendations([], openai_colleges)
                    else:
                        final_response = f"I apologize, but I couldn't find specific college recommendations for {preferences.location} with your preferences. Could you please provide more details about your requirements or consider nearby locations?"
                        recommendations = {"college_recommendations": []}
                
                elif not filtered_colleges:
                    # No specific location, try to get general recommendations
                    openai_colleges = self._get_openai_college_recommendations(preferences)
                    if openai_colleges:
                        final_response = "Based on your preferences, here are some colleges I recommend:"
                        recommendations = self._format_college_recommendations([], openai_colleges)
                    else:
                        final_response = "I need more specific information about your preferences. Could you please tell me your preferred location, course type, or other requirements?"
                        recommendations = {"college_recommendations": []}
                
                else:
                    # Found colleges in database
                    final_response = "Based on your preferences, here are the best matching colleges:"
                    
                    # Get additional colleges from OpenAI if needed
                    openai_colleges = []
                    if len(filtered_colleges) < 3:
                        openai_colleges = self._get_openai_college_recommendations(preferences, preferences.location)
                    
                    recommendations = self._format_college_recommendations(filtered_colleges, openai_colleges)
                
                # Save AI response
                response_with_recs = final_response + "\n\n" + json.dumps(recommendations, indent=2, ensure_ascii=False)
                self.db_manager.save_message(chat_id, 'ai', response_with_recs)
                
                return {
                    'success': True,
                    'response_type': 'recommendations',
                    'message': final_response,
                    'recommendations': recommendations['college_recommendations'],
                    'preferences': preferences.dict()
                }
            
            else:
                # Regular conversation
                if chat_id not in self.conversation_chains:
                    self.conversation_chains[chat_id] = self._create_conversation_chain(chat_id)
                
                conversation = self.conversation_chains[chat_id]
                
                # Add context about extracted preferences if any
                context_info = ""
                if preferences.location or preferences.course_type or preferences.college_type:
                    pref_list = []
                    if preferences.location:
                        pref_list.append(f"location: {preferences.location}")
                    if preferences.course_type:
                        pref_list.append(f"course: {preferences.course_type}")
                    if preferences.college_type:
                        pref_list.append(f"type: {preferences.college_type}")
                    
                    context_info = f"(I understand you're interested in {', '.join(pref_list)}. Feel free to ask for college recommendations when ready!)\n\n"
                
                final_response = context_info + conversation.predict(input=message)
                
                # Save AI response
                self.db_manager.save_message(chat_id, 'ai', final_response)
                
                return {
                    'success': True,
                    'response_type': 'conversation',
                    'message': final_response,
                    'preferences': preferences.dict()
                }
        
        except Exception as e:
            error_message = f"Sorry, there was an error processing your request: {str(e)}"
            return {
                'success': False,
                'response_type': 'error',
                'message': error_message,
                'error': str(e)
            }
    
    def get_chat_history(self, chat_id: str) -> List[Dict]:
        """Get complete chat history for a session"""
        return self.db_manager.get_chat_messages(chat_id)

# Factory function to create the chain (for easy integration)
def create_college_recommendation_chain(api_key: str, excel_path: str, db_path: str) -> CollegeRecommendationChain:
    """Factory function to create and return a CollegeRecommendationChain instance"""
    return CollegeRecommendationChain(api_key, excel_path, db_path)

# Example usage
if __name__ == "__main__":
    # Load environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_PATH = os.getenv("DB_PATH", "college_chat.db")
    EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")
    
    # Create the chain
    college_chain = create_college_recommendation_chain(
        api_key=OPENAI_API_KEY,
        excel_path=EXCEL_PATH,
        db_path=DB_PATH
    )
    
    # Example usage
    chat_id = "example_chat_123"
    
    # Process a message
    result = college_chain.process_message(
        chat_id=chat_id,
        message="suggest me colleges in delhi"
    )
    
    print(f"Success: {result['success']}")
    print(f"Response Type: {result['response_type']}")
    print(f"Message: {result['message']}")
    
    if result['response_type'] == 'recommendations':
        print(f"Number of recommendations: {len(result['recommendations'])}")
        for rec in result['recommendations'][:2]:  # Show first 2
            print(f"- {rec.get('name', 'N/A')}: {rec.get('location', 'N/A')}")