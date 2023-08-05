# Discord-GPT2-Bot // v.1.11 // (c) 2023
# For local TensorFlow-based GPT-2 models
# By FlyingFathead (w/ ChaosWhisperer)
# https://github.com/FlyingFathead/

import sys
import os
import asyncio
import re
import requests
import json
import numpy as np
import tensorflow.compat.v1 as tf
import model, sample, encoder
import discord
import random
import time
from random import randint, choice
from discord.ext import commands
from collections import deque

tf.disable_eager_execution()

# Bot's version number
BOT_VERSION = "1.11"

# Greet on join (True/False)
BOT_GREETING = True

# Prefixes; replace with yours, according to your model.
USER_PREFIX = "User: "
BOT_PREFIX = "Bot: "

# model name & (sub-)directory
# default is the model size (i.e. 112M, 124M, ...)
model_name = '124M'
# default is `models`
models_dir = 'models'

# context buffer for the bot
context_buffer = deque(maxlen=10)  # Set the maximum number of messages to remember

# Don't forget to initialize the reply_to_probability variable at the beginning of your code
reply_to_probability = 0.65  # Set the probability of using reply-to (0.5 = 50% chance) # set in beginning
# response_probability = 0.5  # Set the probability of the bot responding to a message (0.5 = 50% chance)
overall_reply_probability = 1.0  # You can adjust this initial value as needed

# Reading authorized users from this file
authorized_users_file_path = "authorized_users.txt"

# List of usernames authorized to use the admin-style commands
# be sure to add yours into: authorized_users.txt
# Add one username per line (i.e. UserName#0000)
# NOTE: in 2023 with Discord's new "unique" handles, username can be UserName#0 if their name has no #xxxx-suffix

def read_authorized_users_from_file(file_path):
    global authorized_users_file_path
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"[WARN] `{authorized_users_file_path}` not found! No users have been authorized.")
        return []
    
    with open(file_path, 'r') as file:
        authorized_users = [user.strip() for user in file.readlines()]

    # Check if the list is empty
    if not authorized_users:
        print(f"[WARN] `{authorized_users_file_path}` is empty! No users have been authorized to the bot.")
        return []

    # Print the [INFO] statement with the list of authorized users
    print(f"[INFO] Authorized users read from {authorized_users_file_path}: {', '.join(authorized_users)}")
    
    return authorized_users

authorized_users = read_authorized_users_from_file(authorized_users_file_path)

# startup temp
temperature = 0.7  # Initialize the global temperature variable

# read the bot API token [token.txt]
def read_token_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"[ERROR] Bot's Discord API token could not be read from {file_path}, cannot start.")
        sys.exit(1)
    
    with open(file_path, 'r') as file:
        token = file.read().strip()

    # Check if the token is empty
    if not token:
        print(f"[ERROR] Bot's Discord API token is empty in {file_path}, cannot start.")
        sys.exit(1)

    return token

token_file_path = "token.txt"
BOT_TOKEN = read_token_from_file(token_file_path)

# Fallback URL's [fallback_urls.txt]
# Banned URL's [banned_urls.txt]
# Read URLs from a file and return a list of URLs
def read_urls_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} not found!")
        return []
    
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file]

    # Check if the list is empty
    if not urls:
        print(f"[WARN] {file_path} is empty!")
    
    return urls

# Reading fallback URLs
fallback_urls_file = 'fallback_urls.txt'
fallback_urls = read_urls_from_file(fallback_urls_file)

# Reading banned URLs
banned_urls_file = 'banned_urls.txt'
banned_urls = read_urls_from_file(banned_urls_file)

# engagement
# Global variables
last_message_time = time.time()
engagement_threshold = 30  # Set the engagement threshold in seconds
mention_cooldown = 30  # Set the mention cooldown in seconds
last_mention_time = time.time()
max_user_mentions = 1
last_message_time = 0
last_mention_time = 0  # Initialize the last_mention_time variable
engagement_threshold = 30  # In seconds
max_responses = 2  # Set the maximum number of responses

### [ LINE BELOW COMMENTED OUT = GPU MODE. UNCOMMENTED: CPU MODE ]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# read our helper lists
def read_list_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        return []
    with open(file_path, 'r') as file:
        item_list = [item.strip() for item in file.readlines()]
    return item_list

# parsing out unwanted and unneeded portions of text based on different variables
# set one rule-matching term, phrase, etc. per line.
parse_out_line_file_path = "parse_out_line.txt" # lines containing these strings will be parsed out completely (redo the whole line)
parse_to_random_file_path = "parse_to_random.txt" # lines containing these strings will be parsed into random
parse_out_file_path = "parse_out.txt" # lines containing these words will have these strings parsed out of them
parse_to_username_file_path = "parse_to_username.txt" # these strings will be parsed into usernames

parse_out_line = read_list_from_file(parse_out_line_file_path)
parse_to_random = read_list_from_file(parse_to_random_file_path)
parse_out = read_list_from_file(parse_out_file_path)
parse_to_username = read_list_from_file(parse_to_username_file_path)

# URL PRE-PROCESSING & STRING REPLACEMENT
# parsing out and replacing unwanted and unneeded strings and replacing them with desired ones
def read_replacements_from_file(file_path):
    with open(file_path, 'r') as file:
        replacements = json.load(file)
    return replacements

word_replacements_file_path = "word_replacements.json"
url_replacements_file_path = "url_replacements.json"
word_replacements = read_replacements_from_file(word_replacements_file_path)
url_replacements = read_replacements_from_file(url_replacements_file_path)
replacements = {**word_replacements, **url_replacements}

# Alert picture, .gif, message, can be anything...
gif_url = "https://<gif-url>"

# Generate a response from GPT-2
alert_prompt = "Bot: I received an alert "
alert_response_text = "I received an alert "

# set the bot's greeting / main output channel
greeting_channel_name = "general"  # Replace this with the name of the desired channel, without the `#`

# Add this helper function to check if a user is authorized
def is_user_authorized(user):
    return str(user) in authorized_users

# get the model's checkpoint number (if available)
def get_checkpoint_number():
    global model_name, models_dir
    # Construct the path using the set variables
    counter_path = f"{models_dir}/{model_name}/counter"
    
    # Check if the file exists
    if not os.path.exists(counter_path):
        print(f"[WARN] Couldn't find checkpoint number from: {counter_path}")
        return None

    with open(counter_path, "r") as f:
        checkpoint_number = f.read()
    return checkpoint_number

# check for invalid URL tokens
def contains_invalid_url_token(text, url_map):
    url_tokens = re.findall(r'<URL_\d{4}>', text)
    for url_token in url_tokens:
        if url_token not in url_map:
            return True
    return False

# discard cyrillics in output text (sorry, model sometimes hallucinates!)
def contains_cyrillic(text):
    return any(u'\u0400' <= char <= u'\u04FF' for char in text)

# v1.05 // Aug 05 2023
def generate_response(user_input, context_buffer, sess, context, output, enc, num_rolls=1, batch_size=1, max_attempts=10, temperature=0.8):
    global USER_PREFIX, BOT_PREFIX

    # Include the context buffer in the prompt
    context_string = '\n'.join(context_buffer)
    prompt = f"{context_string}\n{USER_PREFIX} {user_input}\n{BOT_PREFIX}"
    context_tokens = enc.encode(prompt)

    response_texts = []
    max_attempts = 5 if max_attempts is None else max_attempts

    for _ in range(num_rolls):
        attempts = 0
        while attempts < max_attempts:
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            text = enc.decode(out[0])

            match = re.search(f'{BOT_PREFIX}', text)
            if match:
                start = match.start()
                end = text.find('\n', start)
                if end == -1:
                    end = len(text)
                text = text[match.end():end]

            # Stop generating text if a user line is encountered
            user_match = re.search(f'{USER_PREFIX}', text)
            if user_match:
                text = text[:user_match.start()]

            text = text.replace('', '')
            text = text.lstrip()

            print(f'Attempt: {attempts + 1}, Response: {text}')  # Debugging print statement

            if not contains_cyrillic(text) and not text.startswith("unanswerable"):
                response_texts.append(text)
                break
            attempts += 1

        # If all attempts have Cyrillic characters or unanswerable responses, just add the last generated response
        if attempts == max_attempts:
            response_texts.append(text)

    return '\n'.join(response_texts)

def initialize_gpt2():
    global model_name, models_dir

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # Adjust hyperparameters here
    # NOTE ON THE HPARAMS

        # Temperature: A temperature of around 0.5-0.7 is usually a good starting point. This will produce responses that are relatively creative and varied, without being too random or nonsensical.
        # Top-k: A top-k value of around 20-50 is a good starting point. This will allow the model to consider a wide range of possible responses, while still filtering out low-probability options.
        # Top-p: A top-p value of around 0.8-0.9 is a good starting point. This will allow the model to consider a relatively large number of options, while still maintaining some level of coherence and relevance in the generated responses

    hparams.temperature = 0.77
    hparams.top_k = 40
    hparams.top_p = 0.9

    context = tf.placeholder(tf.int32, [1, None])
    output = sample.sample_sequence(
        hparams=hparams, length=128,
        context=context,
        batch_size=1,
        temperature=hparams.temperature, top_k=hparams.top_k, top_p=hparams.top_p
    )

    saver = tf.train.Saver()
    sess = tf.Session(graph=tf.get_default_graph())
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)

    return sess, context, output, enc

def apply_replacements(text, user_mention, url_map):
    for pattern, replacement in replacements.items():
        if pattern.startswith("<URL_"):
            # Check if replacement is in url_map before replacing URL tokens with their corresponding URL
            if replacement in url_map:
                text = text.replace(pattern, url_map[replacement])
        elif replacement == "":
            # If the replacement is an empty string, replace with the user mention
            text = re.sub(pattern, user_mention, text)
        else:
            # Otherwise, replace with the specified replacement string
            text = re.sub(pattern, replacement, text)
    return text

# TEXT FILTER v4.0 // 09/05/2023
def filter_output(text, parse_out, parse_out_line, parse_to_random, members, user, parse_to_username, banned_urls_list):
    user_mention = f'<@{user.id}>'
    url_map = {}

    # Define the URL pattern
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Replace URLs with unique identifiers and store the mapping in url_map
    i = [0]
    def replace_url(match, counter=i):
        counter[0] += 1
        url_map[f"<URL_{counter[0]:04}>"] = match.group(0)
        return f"<URL_{counter[0]:04}>"
    text = url_pattern.sub(replace_url, text)

    text = apply_replacements(text, user_mention, url_map)

    # Read random image URLs from url_image.txt
    try:
        with open('url_image.txt', 'r') as f:
            image_urls = f.read().splitlines()
    except FileNotFoundError:
        image_urls = []

    try:
        with open(fallback_urls_file, 'r') as f:
            fallback_urls = f.read().splitlines()
    except FileNotFoundError:
        fallback_urls = ['https://google.com', 'https://news.google.com']
        with open(fallback_urls_file, 'w') as f:
            f.write('\n'.join(fallback_urls))

    try:
        with open(banned_urls_file, 'r') as f:
            banned_urls = f.read().splitlines()
    except FileNotFoundError:
        banned_urls = []
        with open(banned_urls_file, 'w') as f:
            f.write('# Add banned URLs, one per line')

    for item in parse_out:
        text = text.replace(item, "")

    lines = text.split("\n")
    filtered_lines = [line for line in lines if not any(item in line for item in parse_out_line)]    

    for old_name in parse_to_random + parse_to_username:
        for line_index, line in enumerate(filtered_lines):
            if old_name in line:
                if old_name in parse_to_username:
                    new_name = f'<@{user.id}>'
                else:
                    new_name = random.choice(members).name
                filtered_lines[line_index] = line.replace(old_name, new_name)

    final_lines = []
    for line in filtered_lines:
        urls = re.findall(r'<URL_\d{4}>', line)
        for url in urls:
            if url in url_map:
                line = line.replace(url, url_map[url])
            else:
                line = line.replace(url, random.choice(fallback_urls))

        # Replace <URL_IMAGE> with a random image URL from url_image.txt
        line = line.replace('<URL_IMAGE>', random.choice(image_urls)) if image_urls else line

        final_lines.append(line)

    return "\n".join(final_lines)

# URL validator v.1.3
# Updated 25/04/2023 /// with advanced YouTube validity check // more validity
def is_url_valid(url):
    try:
        response = requests.get(url, timeout=5)
        return 200 <= response.status_code < 300
    except requests.exceptions.RequestException:
        return False

def is_youtube_video_valid(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return False

        oembed_url = f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_video_id(url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?youtu(?:be\.com\/watch\?v=|\.be\/)([\w\-\_]*)(&(amp;)?‌​[\w\?‌​=]*)?'
    result = re.match(pattern, url)

    if result:
        return result.group(1)
    return None

# RESPONDER v1.08
# Updated 05/05/2023 // fixed user mention cooldown
# Discord bot setup
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
bot = commands.Bot(command_prefix="!", intents=intents)

# !alert v1.2
@bot.command()
async def alert(ctx):

    global alert_response_text
    global alert_prompt

    prompt = alert_prompt
    # response = alert_response_text + generate_response(prompt, sess, context, output, enc, temperature=temperature)
    response = alert_response_text + generate_response(prompt, [], sess, context, output, enc, temperature=temperature)
    
    # Filter the response
    server_members = ctx.guild.members
    filtered_response = filter_output(response, parse_out, parse_out_line, parse_to_random, server_members, ctx.message.author, parse_to_username, banned_urls)

    # Send the GIF URL
    await ctx.send(gif_url)

    # Send the generated response
    await ctx.send(filtered_response)

@bot.event
async def on_message(message):
    global last_message_time, temperature, last_mention_time
    last_mention_time = time.time()

    if message.author == bot.user:
        return

    await bot.process_commands(message)

    current_time = time.time()
    time_elapsed = current_time - last_message_time
    quiet_duration = 5  # Set the quiet time duration in seconds

    server_members = message.guild.members

    # num_responses = random.randint(1, 5)
    num_responses = random.randint(1, max_responses)
    user_mentions = 0
    max_user_mentions = 1

    # Check if the bot is mentioned or replied to directly
    if bot.user in message.mentions or (message.reference and message.reference.resolved.author == bot.user):
        response_probability = 1.0  # Set the response_probability to 100%

    # if time_elapsed >= quiet_duration:
    if time_elapsed >= quiet_duration and current_time >= last_message_time:
        # add context
        context_line = f'{message.author.display_name}: {message.content}'
        context_buffer.append(context_line)
        
        last_message_time = current_time
        last_mention_time = current_time
        user_mentions = 0  # reset the user mention count for every message
        # if random.random() < overall_reply_probability:  # Check if the bot should respond to the message
        if True:  # Check if the bot should respond to the message
            for i in range(num_responses):
                filtered_response = ""
                while not filtered_response.strip():
                    response = generate_response(message.content, context_buffer, sess, context, output, enc, temperature=temperature)
                    filtered_response = filter_output(response, parse_out, parse_out_line, parse_to_random, server_members, message.author, parse_to_username, banned_urls)

                if i == 0 and time_elapsed > engagement_threshold and user_mentions < max_user_mentions:
                    if filtered_response.strip() and user_mentions < max_user_mentions and current_time - last_mention_time >= mention_cooldown:
                        filtered_response = f"{message.author.mention} {filtered_response}"
                        user_mentions += 1
                        last_mention_time = current_time
                        print(f"user_mentions: {user_mentions}")

                filtered_response = re.sub(r'�', '', filtered_response)
                print(f"filtered_response: {filtered_response}")

                if i == 0:
                    if message.reference and message.reference.resolved.author == bot.user:
                        print(f"Sending response (reply to mention): {filtered_response}")
                        await message.reply(filtered_response, mention_author=False)
                    elif message.content.startswith(f'<@!{bot.user.id}>') or message.content.startswith(f'<@{bot.user.id}>'):
                        print(f"Sending response (direct mention): {filtered_response}")
                        await message.channel.send(filtered_response)
                    elif random.random() < reply_to_probability:
                        print(f"Sending response (reply): {filtered_response}")
                        await message.reply(filtered_response, mention_author=False)
                    else:
                        print(f"Sending response (channel): {filtered_response}")
                        await message.channel.send(filtered_response)
                else:
                    print(f"Sending response (channel, no quote): {filtered_response}")
                    await message.channel.send(filtered_response)

            last_message_time = current_time        

# sets maximum number of throws
@bot.command(name='maxrolls')
async def set_max_responses(ctx, num: int):
    global max_responses
    max_responses = num
    await ctx.send(f"🤖🎲 Maximum number of responses set to: {num}")

@bot.command()
async def odds(ctx, new_odds: float):
    if not is_user_authorized(ctx.author):
        await ctx.send(f"Sorry! You're not authorized to adjust my odds.")
        return

    if 0 <= new_odds <= 1:  # You can adjust this range as needed
        global overall_reply_probability  # Add this global variable at the beginning of your code
        overall_reply_probability = new_odds
        await ctx.send(f"My odds of responding are now {overall_reply_probability * 100}%! 🤖🎲")
    else:
        await ctx.send(f"Sorry! That odds value is out of range. Please enter a value between 0 and 1.")

@bot.command()
async def reply_to_odds(ctx, new_odds: float):
    if not is_user_authorized(ctx.author):
        await ctx.send(f"Sorry! You're not authorized to adjust my odds.")
        return

    if 0 <= new_odds <= 1:  # You can adjust this range as needed
        global reply_to_probability  # Add this global variable at the beginning of your code
        reply_to_probability = new_odds
        await ctx.send(f"My odds of responding as a reply are now {reply_to_probability * 100}%! 🐴🎲 Time to make things interesting!")
    else:
        await ctx.send(f"Sorry! That odds value is out of range. Please enter a value between 0 and 1.")

@bot.command(aliases=["shutdown"])
async def goodnight(ctx):
    global authorized_users

    # console / debug
    print(f"[INFO] Goodnight/logoff request from: {ctx.author}")  # This will print out the author who sent the command
    print(f"[INFO] Authorized Users: {authorized_users}")  # This will print out the list of authorized users
    if not is_user_authorized(ctx.author):
        await ctx.send(f"Sorry! You're not authorized to send me off into the night.")
        return

    await ctx.send(f"Goodnight, people! 💤 See y'all on the other side!")
    await bot.close()

@bot.command()
async def temp(ctx, new_temperature: float):
    if 0 <= new_temperature <= 2:  # You can adjust this range as needed
        global temperature  # Add this global variable at the beginning of your code
        temperature = new_temperature
        await ctx.send(f"My temperature is now: {temperature} 🌡️")
    else:
        await ctx.send(f"Sorry! That temperature is out of range. Please enter a value between 0 and 2.")

""" # old
@bot.command(name="gpt2")
async def gpt2(ctx, *, user_input: str):
    replace_user_placeholder = ""

    max_num_responses = 3
    num_responses = random.randint(1, max_num_responses)
    for _ in range(num_responses):
        # response = generate_response(user_input, sess, context, output, enc)
        response = generate_response(message.content, sess, context, output, enc, temperature)
        response = await replace_user_placeholder(response, ctx.message)
        await ctx.send(response)

sess, context, output, enc = initialize_gpt2() """

async def check_bot_connection():
    global BOT_VERSION
    global BOT_GREETING

    while not bot.is_ready():
        await asyncio.sleep(1)  # Check every second

    checkpoint_number = get_checkpoint_number()  # Get the checkpoint number from the counter file    

    # Get the filename of the script being executed
    script_filename = os.path.basename(__file__)
    
    # Get the version number
    version_number = BOT_VERSION

    if BOT_GREETING:
        for guild in bot.guilds:
            greeting_channel = discord.utils.get(guild.text_channels, name=greeting_channel_name)
            if greeting_channel and greeting_channel.permissions_for(guild.me).send_messages:
                await greeting_channel.send(f"**Greetings!** GPT-2 Discord Bot v{version_number} is online! [using checkpoint #{checkpoint_number}]")
                print("[INFO] Greeting message has been sent.")

# Initialize the GPT-2 model
sess, context, output, enc = initialize_gpt2()

# run the bot loop
bot.loop.create_task(check_bot_connection())
bot.run(BOT_TOKEN)
