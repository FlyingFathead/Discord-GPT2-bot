# Discord-GPT2-bot
A simple Discord chatbot powered by OpenAI's GPT-2 model, designed to run on local TensorFlow setups.

# Install
1. **Clone the repository**

```
git clone https://github.com/FlyingFathead/Discord-GPT2-bot
cd Discord-GPT2-bot/
```

2. **Install dependencies**
```
pip install -r requirements.txt
```

3. **Get the OpenAI GPT-2 model** and tune it according to your requirements. Place the model in the corresponding subdirectory (e.g., `models/124M/`). Ensure that the directory contains all necessary TensorFlow model files, including `vocab.bpe`, `encoder.json`, and `hparams.json`.

4. **Set up the bot** (see the code; fix the variables to suit your use scenario):
    - Save your Discord API bot token in `token.txt` (one line).
    - Define the bot's authorized users in `authorized_users.txt` (one username per line; syntax i.e. `UserName#0` or `UserName#xxxx`).

6. **Optional & additional configuration**
   - Feel free to configure various replacement lists for elements like unwanted URLs, words, strings, allowed channels etc. Refer to the source code for a complete list of replaceable items.

8. **Run** with `python Discord-GPT2-bot.py` and enjoy! (?)

# Usage
Talk to the bot and it should respond.

  Commands:
- `!goodnight` - the bot shuts down (user must be in the authorized users list, `authorized_users.txt` by default)
- `!maxrolls x` - max number of replies to a single line (x = number of replies)
- `!odds x` - odds (probability) of the bot answering; 0 for 0%, 1 for 100%, use fractions like 0.5 = 50%
- `!temp x` - set the model's temperature (between 0 and 2). Optimal for an optimized model usually hangs at around 1.
- `!alert` - send out a test alert (for testing, debug purposes etc)

# Notes
This is mostly for humorous & testing purposes, although the code works perfectly fine when set up correctly, don't expect miracles out of the bot. Actual performance depends on fine-tuning and model size, etc. -- Feel free to expand on its functionalities and adapt it to your needs.

From [FlyingFathead](https://github.com/FlyingFathead/), in collaboration with ChaosWhisperer.
