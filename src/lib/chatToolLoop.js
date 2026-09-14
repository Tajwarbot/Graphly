/** Bounded tool-response loop; never replay mutations after a network failure. */
export async function runChatTools(
    chat,
    parts,
    executeTool,
    { maxRounds = 4, maxActions = 16 } = {}
) {
    const actions = [];
    let response;
    try {
        response = (await chat.sendMessage(parts)).response;
        for (let round = 0; round < maxRounds; round++) {
            const calls = response.functionCalls?.() || [];
            if (!calls.length)
                return {
                    text:
                        response.text?.() || 'No graph changes were requested.',
                    actions
                };
            if (actions.length + calls.length > maxActions)
                return {
                    text: `The request exceeded ${maxActions} graph actions. Break it into smaller scenes.`,
                    actions
                };
            const results = [];
            for (const call of calls) {
                let result;
                try {
                    result = await executeTool(call.name, call.args || {});
                } catch (error) {
                    result = { success: false, message: error.message };
                }
                actions.push({
                    name: call.name,
                    args: call.args,
                    result: result.message,
                    success: result.success
                });
                results.push({
                    functionResponse: { name: call.name, response: result }
                });
            }
            // Return actual results to the model so it can correct a failed expression
            // or complete a scene that requires several dependent operations.
            response = (await chat.sendMessage(results)).response;
        }
        return {
            text: 'Reached the planning limit. Completed graph updates are listed below; ask to continue for additional changes.',
            actions
        };
    } catch (error) {
        if (actions.length)
            return {
                text: 'The connection stopped after the graph updates below. They were kept; no actions were repeated.',
                actions
            };
        throw error;
    }
}
