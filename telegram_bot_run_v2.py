
from dotenv import load_dotenv

import logging
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackContext, filters
import os
from rich.console import Console
from rich.markdown import Markdown
import uuid
import time

user_sessions = {}
SESSION_TIMEOUT = 30 * 60 

def generate_session():
    """Generate a unique session ID"""
    return str(uuid.uuid4())


from tim_review_agent_v2 import get_graph

load_dotenv(override=True)

graph = get_graph()


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

greeting = """
Hi there! I'm here to answer questions based on articles from *Tim Review Journal* articles.
Please ask me anything related to _technology innovation management_. Oterwise, I will steer you back on track!
A session will expire after 30 minutes of inactivity. 🕒
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Command /start received.")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=greeting, parse_mode="Markdown")
    
    chat_id = update.effective_chat.id
    user_sessions[chat_id] = {
        'session_id': generate_session(),
        'last_time': time.time()
    }
        
    
async def message_handler(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id
    current_time = time.time()
    
    if chat_id in user_sessions:
        last_interaction = user_sessions[chat_id]['last_time']
        # If inactive for more than 30 minutes, create a new session
        if (current_time - last_interaction) > SESSION_TIMEOUT:
            user_sessions[chat_id]['session_id'] = generate_session()
            await context.bot.send_message(chat_id=chat_id, text="Your session expired. A new session has been assigned. 🔄", parse_mode="Markdown")

        user_sessions[chat_id]['last_time'] = current_time  # Update last interaction time
    else:
        # First-time interaction: create a new session
        user_sessions[chat_id] = {
            'session_id': generate_session(),
            'last_time': current_time
        }
        await context.bot.send_message(chat_id=chat_id, text="New session assigned! 🎯", parse_mode="Markdown")
    
    await update.message.reply_text(f"Please give me a second to process ⏳...")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    config = {"configurable": {"thread_id": user_sessions[chat_id]['session_id']}}
    response = graph.invoke({"messages": user_message}, config)
    await update.message.reply_text(text=response['messages'][-1].content, parse_mode="Markdown")

if __name__ == '__main__':
    application = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    application.run_polling()
    
    
    