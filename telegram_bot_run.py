
from dotenv import load_dotenv

import logging
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackContext, filters
import os
from rich.console import Console
from rich.markdown import Markdown

from tim_review_agent_multitopics import get_graph

load_dotenv(override=True)

graph = get_graph()


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

greeting = """
Hi there! I'm here to answer questions based on articles from *Tim Review Journal* articles.
Please ask me anything related to _technology innovation management_. Oterwise, I will steer you back on track!
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Command /start received.")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=greeting, parse_mode="Markdown")
    
async def message_handler(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Please give me a second to process ⏳...")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    response = graph.invoke({"query": user_message})
    
    await update.message.reply_text(text=response['messages'][-1].content, parse_mode="Markdown")

if __name__ == '__main__':
    application = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    application.run_polling()
    
    
    