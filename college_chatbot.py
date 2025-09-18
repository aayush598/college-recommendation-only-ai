import sqlite3
import json
import pandas as pd
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, validator
import re
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# --- Data Models ---
class UserPreferences(BaseModel):
    """User preferences extracted from conversation"""
    location: Optional[str] = Field(None, description="Preferred city or state")
    course_type: Optional[str] = Field(None, description="Engineering, Medicine, Arts, etc.")
    college_type: Optional[str] = Field(None, description="Government, Private, Deemed")
    level: Optional[str] = Field(None, description="UG or PG")
    budget: Optional[str] = Field(None, description="Budget preference")
    specific_course: Optional[str] = Field(None, description="BTech, MBA, MBBS, etc.")

class ConversationState(TypedDict):
    """State for the conversation graph"""
    messages: Annotated[List[Dict], add_messages]
    chat_id: str
    user_input: str
    preferences: Dict
    response: str
    recommendations: List[Dict]
    needs_tool_call: bool

@dataclass
class College:
    college_id: str
    name: str
    type: str
    location: str
    courses: str
    website: str
    fees: str
    admission_process: str

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chat sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                chat_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Messages with memory
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chat_sessions (chat_id)
            )
        ''')
        
        # Preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                chat_id TEXT PRIMARY KEY,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_message(self, chat_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('INSERT OR IGNORE INTO chat_sessions (chat_id) VALUES (?)', (chat_id,))
        cursor.execute(
            'INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)',
            (chat_id, role, content)
        )
        cursor.execute(
            'UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?',
            (chat_id,)
        )
        
        conn.commit()
        conn.close()
    
    def get_messages(self, chat_id: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content, timestamp FROM messages 
            WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (chat_id, limit))
        messages = cursor.fetchall()
        conn.close()
        
        return [{'role': m[0], 'content': m[1], 'timestamp': m[2]} for m in reversed(messages)]
    
    def save_preferences(self, chat_id: str, preferences: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO preferences (chat_id, data) VALUES (?, ?)',
            (chat_id, json.dumps(preferences))
        )
        conn.commit()
        conn.close()
    
    def get_preferences(self, chat_id: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM preferences WHERE chat_id = ?', (chat_id,))
        result = cursor.fetchone()
        conn.close()
        
        return json.loads(result[0]) if result else {}

# --- College Data Manager ---
class CollegeDataManager:
    def __init__(self, excel_path: str):
        self.colleges = self.load_colleges(excel_path)
    
    def load_colleges(self, excel_path: str) -> List[College]:
        try:
            df = pd.read_excel(excel_path)
            colleges = []
            for _, row in df.iterrows():
                college = College(
                    college_id=str(row.get('College ID', '')),
                    name=str(row.get('College', '')),
                    type=str(row.get('Type', '')),
                    location=str(row.get('Location', '')),
                    courses=str(row.get('Courses (ID, Category, Duration, Eligibility, Language, Accreditation, Fees)', '')),
                    website=str(row.get('Website', '')),
                    fees=str(row.get('Fees', '')),
                    admission_process=str(row.get('Admission Process', ''))
                )
                colleges.append(college)
            return colleges
        except Exception as e:
            print(f"Error loading college data: {e}")
            return []

# --- College Recommendation Tool ---
class CollegeRecommendationTool(BaseTool):
    name: str = "college_recommendation"
    description: str = """Use this tool when user asks for college recommendations or suggestions. 
    Input should be a JSON string with user preferences like location, course_type, college_type, etc.
    This tool will return detailed college recommendations based on the preferences."""
    
    data_manager: CollegeDataManager
    openai_api_key: str
    
    def __init__(self, data_manager: CollegeDataManager, openai_api_key: str):
        super().__init__(data_manager=data_manager, openai_api_key=openai_api_key)
        openai.api_key = openai_api_key
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            # Parse preferences from query
            preferences = self._extract_preferences(query)
            
            # Filter colleges from database
            filtered_colleges = self._filter_colleges(preferences)
            
            # Get OpenAI recommendations if needed
            openai_colleges = []
            if len(filtered_colleges) < 3:
                openai_colleges = self._get_openai_recommendations(preferences)
            
            # Format response
            recommendations = self._format_recommendations(filtered_colleges, openai_colleges)
            
            return json.dumps({
                "success": True,
                "recommendations": recommendations,
                "total_found": len(recommendations),
                "message": f"Found {len(recommendations)} colleges matching your preferences"
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "recommendations": [],
                "message": "Error getting college recommendations"
            })
    
    def _extract_preferences(self, query: str) -> UserPreferences:
        """Extract preferences from query string"""
        try:
            if query.startswith('{'):
                # JSON input
                data = json.loads(query)
                return UserPreferences(**data)
            else:
                # Text input - extract using simple parsing
                prefs = {}
                query_lower = query.lower()
                
                # Location extraction
                locations = ['delhi', 'mumbai', 'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata', 'indore', 'bhopal']
                for loc in locations:
                    if loc in query_lower:
                        prefs['location'] = loc.title()
                        break
                
                # Course type extraction
                if any(word in query_lower for word in ['engineering', 'engineer', 'btech', 'b.tech']):
                    prefs['course_type'] = 'Engineering'
                elif any(word in query_lower for word in ['medicine', 'medical', 'mbbs', 'doctor']):
                    prefs['course_type'] = 'Medicine'
                elif any(word in query_lower for word in ['mba', 'management', 'business']):
                    prefs['course_type'] = 'Management'
                
                return UserPreferences(**prefs)
        except:
            return UserPreferences()
    
    def _filter_colleges(self, preferences: UserPreferences) -> List[Dict]:
        """Filter colleges based on preferences"""
        matching = []
        
        for college in self.data_manager.colleges:
            score = 0
            reasons = []
            
            # Location match
            if preferences.location and preferences.location.lower() in college.location.lower():
                score += 30
                reasons.append(f"Located in {preferences.location}")
            
            # Course type match
            if preferences.course_type:
                if preferences.course_type.lower() in college.courses.lower():
                    score += 25
                    reasons.append(f"Offers {preferences.course_type} courses")
            
            # College type match
            if preferences.college_type and preferences.college_type.lower() in college.type.lower():
                score += 20
                reasons.append(f"Matches {preferences.college_type} type")
            
            if score > 0:
                matching.append({
                    'college': college,
                    'score': score,
                    'reasons': reasons
                })
        
        return sorted(matching, key=lambda x: x['score'], reverse=True)[:5]
    
    def _get_openai_recommendations(self, preferences: UserPreferences) -> List[Dict]:
        """Get recommendations from OpenAI"""
        try:
            pref_text = f"Location: {preferences.location or 'Any'}, Course: {preferences.course_type or 'Any'}"
            
            prompt = f"""
            Recommend 3 good colleges in India for: {pref_text}
            Return as JSON array with structure:
            [{{"name": "College Name", "location": "City, State", "type": "Government/Private", "courses": "Relevant courses", "fees": "Approximate fees", "website": "URL or N/A"}}]
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except:
            return []
    
    def _format_recommendations(self, filtered_colleges: List[Dict], openai_colleges: List[Dict]) -> List[Dict]:
        """Format recommendations for response"""
        recommendations = []
        
        # Add database colleges
        for item in filtered_colleges:
            college = item['college']
            recommendations.append({
                "name": college.name,
                "location": college.location,
                "type": college.type,
                "courses": college.courses,
                "website": college.website,
                "fees": college.fees,
                "admission_process": college.admission_process,
                "match_reasons": item['reasons'],
                "source": "database"
            })
        
        # Add OpenAI colleges
        for college in openai_colleges:
            college["source"] = "openai"
            college["match_reasons"] = ["AI recommendation based on preferences"]
            recommendations.append(college)
        
        return recommendations

# --- Career Counselor Agent ---
class CareerCounselorAgent:
    def __init__(self, llm: ChatOpenAI, db_manager: DatabaseManager, college_tool: CollegeRecommendationTool):
        self.llm = llm
        self.db_manager = db_manager
        self.college_tool = college_tool
        
        # Create agent with tool - Updated for new LangChain version
        self.tools = [college_tool]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=False  # Set to False to avoid multiple outputs
        )
        
        # System prompt
        self.system_prompt = """You are an expert career counselor specializing in Indian higher education.

Your responsibilities:
1. Have warm, supportive conversations about career goals and education
2. Ask thoughtful questions to understand student interests and preferences
3. Provide career guidance and educational pathway advice
4. When students ask for college recommendations, use the college_recommendation tool

Guidelines:
- Be empathetic and encouraging
- Ask one question at a time to avoid overwhelming students
- Extract preferences like location, course type, college type during conversation
- When they ask for specific college recommendations, use the college_recommendation tool
- Always provide context and explanations for recommendations

Tool Usage:
- Use college_recommendation tool when user asks for college suggestions or recommendations
- Pass user preferences as JSON to the tool
- Explain the recommendations you receive from the tool

Remember: You are primarily a counselor and guide. Help students make informed decisions."""

    def _extract_preferences_from_conversation(self, chat_id: str, current_message: str) -> str:
        """Extract preferences and format as JSON for tool"""
        # Get recent conversation
        messages = self.db_manager.get_messages(chat_id, limit=5)
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        conversation_text += f"\nhuman: {current_message}"
        
        # Simple extraction logic
        prefs = {}
        text_lower = conversation_text.lower()
        
        # Extract location
        locations = ['delhi', 'mumbai', 'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata', 'indore', 'bhopal']
        for loc in locations:
            if loc in text_lower:
                prefs['location'] = loc.title()
                break
        
        # Extract course type
        if any(word in text_lower for word in ['engineering', 'engineer', 'btech', 'b.tech']):
            prefs['course_type'] = 'Engineering'
        elif any(word in text_lower for word in ['medicine', 'medical', 'mbbs', 'doctor']):
            prefs['course_type'] = 'Medicine'
        elif any(word in text_lower for word in ['mba', 'management', 'business']):
            prefs['course_type'] = 'Management'
        elif any(word in text_lower for word in ['arts', 'literature', 'ba']):
            prefs['course_type'] = 'Arts'
        
        # Extract college type
        if any(word in text_lower for word in ['government', 'govt', 'public']):
            prefs['college_type'] = 'Government'
        elif any(word in text_lower for word in ['private']):
            prefs['college_type'] = 'Private'
        
        return json.dumps(prefs)

    def _is_asking_for_recommendations(self, message: str) -> bool:
        """Check if user is asking for college recommendations"""
        message_lower = message.lower()
        recommendation_keywords = [
            'recommend colleges', 'suggest colleges', 'which colleges', 'best colleges',
            'good colleges', 'colleges for', 'help me find colleges', 'looking for colleges',
            'college options', 'where should i apply', 'what colleges', 'any colleges',
            'colleges in', 'suggest college', 'show me colleges', 'list colleges',
            'recommend some', 'suggest some'
        ]
        return any(keyword in message_lower for keyword in recommendation_keywords)

    def process_message(self, chat_id: str, message: str) -> Dict:
        """Process message through the agent"""
        try:
            # Load conversation memory
            previous_messages = self.db_manager.get_messages(chat_id, limit=5)
            
            # Check if this is a recommendation request
            asking_for_recs = self._is_asking_for_recommendations(message)
            
            if asking_for_recs:
                # Direct tool usage approach
                preferences_json = self._extract_preferences_from_conversation(chat_id, message)
                tool_result = self.college_tool._run(preferences_json)
                
                try:
                    tool_data = json.loads(tool_result)
                    if tool_data.get('success'):
                        recommendations = tool_data.get('recommendations', [])
                        
                        # Create response with recommendations
                        if recommendations:
                            response = f"Based on your preferences, I found {len(recommendations)} colleges that match your requirements. Here are my recommendations:\n\n"
                            for i, rec in enumerate(recommendations[:3], 1):  # Top 3
                                response += f"{i}. **{rec.get('name', 'N/A')}**\n"
                                response += f"   - Location: {rec.get('location', 'N/A')}\n"
                                response += f"   - Type: {rec.get('type', 'N/A')}\n"
                                response += f"   - Match reasons: {', '.join(rec.get('match_reasons', []))}\n\n"
                            
                            response += "Would you like more details about any of these colleges or have other questions about your educational path?"
                        else:
                            response = "I understand you're looking for college recommendations. Could you please provide more specific details about your preferences like location, course type, or college type so I can help you better?"
                        
                        return {
                            'success': True,
                            'response': response,
                            'recommendations': recommendations,
                            'used_tool': True
                        }
                    else:
                        # Tool failed, continue with conversation
                        response = "I'd be happy to help you find colleges. Could you tell me more about your preferences like preferred location, course type, and what you're looking for in a college?"
                        return {
                            'success': True,
                            'response': response,
                            'recommendations': [],
                            'used_tool': False
                        }
                except json.JSONDecodeError:
                    # Continue with normal conversation
                    pass
            
            # Normal conversation flow
            # Build conversation context
            context = f"{self.system_prompt}\n\nConversation History:\n"
            for msg in previous_messages[-3:]:  # Last 3 messages for context
                context += f"{msg['role'].title()}: {msg['content']}\n"
            
            context += f"\nCurrent message: {message}\n\nPlease respond as a helpful career counselor. Focus on understanding the student's needs and providing guidance."
            
            # Use LLM directly for conversation
            from langchain.schema import HumanMessage
            response = self.llm([HumanMessage(content=context)])
            
            return {
                'success': True,
                'response': response.content,
                'recommendations': [],
                'used_tool': False
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'recommendations': [],
                'used_tool': False,
                'error': str(e)
            }

# --- Main College Counselor System ---
class CollegeCounselorSystem:
    def __init__(self, api_key: str, excel_path: str, db_path: str):
        # Initialize components
        self.db_manager = DatabaseManager(db_path)
        self.data_manager = CollegeDataManager(excel_path)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        
        # Create college recommendation tool
        self.college_tool = CollegeRecommendationTool(
            data_manager=self.data_manager,
            openai_api_key=api_key
        )
        
        # Create agent
        self.agent = CareerCounselorAgent(self.llm, self.db_manager, self.college_tool)
    
    def chat(self, chat_id: str, message: str) -> Dict:
        """Main chat method - single entry point"""
        try:
            # Save user message
            self.db_manager.save_message(chat_id, 'human', message)
            
            # Process through agent
            result = self.agent.process_message(chat_id, message)
            
            if result['success']:
                # Save assistant response
                self.db_manager.save_message(chat_id, 'assistant', result['response'])
                
                # Update preferences if recommendations were made
                if result['used_tool'] and result['recommendations']:
                    # Extract and save preferences (simplified)
                    preferences = self._extract_preferences_from_context(chat_id, message)
                    self.db_manager.save_preferences(chat_id, preferences)
                
                return {
                    'success': True,
                    'response_type': 'recommendations' if result['used_tool'] else 'conversation',
                    'message': result['response'],
                    'recommendations': result['recommendations'],
                    'chat_id': chat_id
                }
            else:
                return {
                    'success': False,
                    'response_type': 'error',
                    'message': result['response'],
                    'recommendations': [],
                    'chat_id': chat_id,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            return {
                'success': False,
                'response_type': 'error',
                'message': f"System error: {str(e)}",
                'recommendations': [],
                'chat_id': chat_id,
                'error': str(e)
            }
    
    def _extract_preferences_from_context(self, chat_id: str, message: str) -> dict:
        """Extract preferences from conversation context"""
        # Simple preference extraction
        preferences = {}
        message_lower = message.lower()
        
        # Location
        locations = ['delhi', 'mumbai', 'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata', 'indore']
        for loc in locations:
            if loc in message_lower:
                preferences['location'] = loc.title()
                break
        
        # Course type
        if any(word in message_lower for word in ['engineering', 'btech']):
            preferences['course_type'] = 'Engineering'
        elif any(word in message_lower for word in ['medicine', 'mbbs']):
            preferences['course_type'] = 'Medicine'
        elif any(word in message_lower for word in ['mba', 'management']):
            preferences['course_type'] = 'Management'
        
        return preferences
    
    def get_chat_history(self, chat_id: str) -> List[Dict]:
        """Get chat history"""
        return self.db_manager.get_messages(chat_id, limit=50)
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get saved preferences"""
        return self.db_manager.get_preferences(chat_id)

# --- Factory Function ---
def create_college_counselor_system(api_key: str, excel_path: str, db_path: str) -> CollegeCounselorSystem:
    """Factory function to create the college counselor system"""
    return CollegeCounselorSystem(api_key, excel_path, db_path)

# --- Example Usage ---
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_PATH = os.getenv("DB_PATH", "college_counselor.db")
    EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set!")
        exit(1)
    
    # Create system
    counselor = create_college_counselor_system(OPENAI_API_KEY, EXCEL_PATH, DB_PATH)
    
    chat_id = "test_user_123"
    
    print("=== College Counselor with Tool-Based Architecture ===\n")
    
    # Test 1: General conversation
    print("Test 1: General conversation")
    result1 = counselor.chat(chat_id, "Hi, I'm confused about what to study after 12th grade.")
    print(f"Response: {result1['message']}")
    print(f"Type: {result1['response_type']}")
    print("-" * 50)
    
    # Test 2: More specific
    print("\nTest 2: Expressing preferences")
    result2 = counselor.chat(chat_id, "I'm interested in engineering, specifically computer science, and I prefer Delhi.")
    print(f"Response: {result2['message']}")
    print(f"Type: {result2['response_type']}")
    print("-" * 50)
    
    # Test 3: Asking for recommendations
    print("\nTest 3: Requesting recommendations")
    result3 = counselor.chat(chat_id, "Can you recommend some good engineering colleges in Delhi?")
    print(f"Response: {result3['message']}")
    print(f"Type: {result3['response_type']}")
    print(f"Recommendations: {len(result3['recommendations'])}")
    
    if result3['recommendations']:
        for i, rec in enumerate(result3['recommendations'][:2], 1):
            print(f"{i}. {rec.get('name', 'N/A')} - {rec.get('location', 'N/A')}")
    
    print("-" * 50)