import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# ✅ Load environment variables
dotenv.load_dotenv()

# ✅ Fetch API Key and Gemini Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Check API Key
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY is missing! Set it in a .env file or manually.")

# ✅ Initialize AI Model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)

# ✅ Define General AI Chatbot Prompt Template
TEMPLATE = """
You are an advanced AI assistant designed to provide **intelligent, structured, and well-explained answers** to users. Your goal is to help users with different types of queries.

📝 **Past Conversations**:
{history}

💬 **User's Current Query**:
{input}

---

## **🛠 How to Answer Based on Use Case**

### **1️⃣ Chatbots (General Conversations)**
- Engage in natural, friendly, and human-like conversations.
- Keep the discussion interactive by asking relevant follow-up questions.

### **2️⃣ Customer Support**
- Provide **step-by-step guidance** for user queries.
- If needed, direct users to external resources for further help.

### **3️⃣ Education & Tutoring**
- Break down **complex concepts into simple explanations**.
- Offer **real-world examples** to improve understanding.

### **4️⃣ Healthcare Assistance**
- Give **general wellness advice** but **avoid medical diagnoses**.
- Always recommend consulting a healthcare professional for serious concerns.

📌 **General Guidelines**:
- Be **accurate, structured, and engaging** in responses.
- If a question is beyond your knowledge, say: **"I don’t have enough data on this, but here’s what I do know..."**
- Adapt the tone based on the use case (friendly, professional, or educational).
"""

# ✅ Create a Prompt Template
prompt = PromptTemplate.from_template(TEMPLATE)

# ✅ Select Memory Based on Use Case
USE_CASE = "education"  # Change this to "chatbot", "customer_support", or "healthcare"

if USE_CASE == "chatbot":
    memory = ConversationBufferMemory(memory_key="history")  # ✅ Remembers everything
elif USE_CASE == "customer_support":
    memory = ConversationBufferMemory(memory_key="history")  # ✅ Keeps full chat history
elif USE_CASE == "education":
    memory = ConversationSummaryMemory(memory_key="history", llm=llm)  # ✅ Summarizes past chats
elif USE_CASE == "healthcare":
    memory = ConversationBufferWindowMemory(memory_key="history", k=3)  # ✅ Remembers last 3 messages only
else:
    memory = ConversationBufferMemory(memory_key="history")  # Default memory

# ✅ Attach Memory to Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

# ✅ Start Chat Loop
print("🤖 AI Chatbot is ready! Type 'exit' to stop.\n")

while True:
    user_input = input("👤 You: ")

    if user_input.lower() == 'exit':
        print("👋 Exiting AI Chatbot. Have a great day! ✨")
        break

    response = conversation.invoke({"input": user_input})

    print("🤖 AI Bot:", response["response"])  

