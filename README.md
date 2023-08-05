# Discord-GPT2-bot
A simple Discord API chatbot for local GPT-2 TensorFlow models.

# Install
1. **Clone the repository**

```
git clone https://github.com/FlyingFathead/Discord-GPT2-bot
cd Discord-GPT2-bot
```

2. **Install dependencies**
```
pip install -r requirements.txt
```

3. **Get the OpenAI GPT-2 model** and tune it for your purposes, place it into the appropriate subdirectory (i.e. `models/124M/`). The directory must include all the Tensorflow model files for the local model, including the `vocab.bpe`, `encoder.json` and `hparams.json` files.

4. **Set up the bot** (see the code; fix the variables to suit your use scenario):
   - set up your Discord API bot token on a single line to `token.txt`
   - set up the bot's authorized users to `authorized_users.txt`.

6. Feel free to set up all the other possible replacement lists (for i.e. unwanted urls, words, strings, etc, see the code for all possible replacement list files...)

7. Enjoy! (?)

# Notes
This is mostly for humorous & testing purposes, although the code works perfectly fine when set up correctly, don't expect miracles out of the bot. Feel free to make your own functionalities as you deem fit.

From [FlyingFathead](https://github.com/FlyingFathead/), in collaboration with ChaosWhisperer.
