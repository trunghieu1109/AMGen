async def forward_89(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Extract each language and its corresponding percentage of speakers from the passage provided in taskInfo."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting language-percentage pairs, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    try:
        language_percentages = eval(results1['answer'].content)
    except:
        language_percentages = []
    languages_above_20 = []
    for idx, pair in enumerate(language_percentages, start=2):
        cot_instruction2 = f"Subtask {idx}: For the language-percentage pair {pair}, determine if the percentage of speakers is greater than 20%."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [pair],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_{idx}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, evaluating if percentage > 20%, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask {idx} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        if 'yes' in results2['answer'].content.lower():
            languages_above_20.append(pair[0])
    debate_instruction = "Subtask final: Based on the list of languages spoken by more than 20% of Richmond residents, compile and produce the final answer listing these languages."
    final_decision_instruction = "Subtask final: Make final decision on the list of languages spoken by more than 20% of Richmond residents."
    debate_desc = {
        'instruction': debate_instruction,
        'context': ["user query"],
        'input': [languages_above_20],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc = {
        'instruction': final_decision_instruction,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results_final = await self.debate(
        subtask_id="subtask_final",
        debate_desc=debate_desc,
        final_decision_desc=final_decision_desc,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results_final['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, compiling final answer, thinking: {results_final['list_thinking'][round][idx].content}; answer: {results_final['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating final answer, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Subtask final output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
