## Agentic Mode

Let's say we want AI to call REST APIs using tools.

If the response from the API was not enough, we want AI to make another API call and repeat until the information is enough.

To enable this kind of progressive function calls, we need the following configuration.

1. Go to `Workspace`.
2. Edit the target model.
3. Click `show` in `Advanced Params`.
4. Change the `Function Calling` value from `Default` to `Native`.

Changing it to native mode would allow the model to call the function directly without generating output, preventing AI from stopping until it has enough information to generate the answer.

It's essentially the agentic mode for Open WebUI.

## Skills

I noticed that it's quite difficult to make AI use skills.

A system prompt like this increased the chance of using skills:
> If there are relevant skills, you must use the `view_skill` function to read them first before doing anything else.
