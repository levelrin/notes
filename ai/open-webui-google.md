First, we can generate an API key from [Google AI Studio](https://aistudio.google.com/).

After that, we should configure Open WebUI like this:
1. Go to `Settings`.
2. Go to `Admin Settings`.
3. Go to `Connections`.
4. In the `OpenAI API` section, click the `Add Connection` button.
5. In the `URL` section, put this: https://generativelanguage.googleapis.com/v1beta/openai
6. In the `Auth` section (Bearer is selected), put the API key.

Now, we should be able to see the models provided by Google.

By the way, we noticed that some models (especially cutting-edge ones) hang (getting stuck in response).
Older models didn't have that issue.
