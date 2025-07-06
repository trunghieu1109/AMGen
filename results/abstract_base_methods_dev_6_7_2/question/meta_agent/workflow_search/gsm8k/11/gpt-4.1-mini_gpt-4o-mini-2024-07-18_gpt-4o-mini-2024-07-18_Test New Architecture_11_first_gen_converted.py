async def forward_11(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Calculate the probability of rolling a number greater than 3 on a six-sided die using Chain-of-Thought reasoning"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, calculating probability of rolling >3, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_sc_instruction2 = "Subtask 2: Calculate the probability of rolling two even numbers in a row on a six-sided die using Self-Consistency Chain-of-Thought reasoning"
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, calculating probability of two even numbers in a row, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])
    aggregate_instruction4 = "Subtask 3: Compute how much more likely (as a percentage) it is to roll a number greater than 3 than to roll two even numbers in a row by consolidating the two probabilities"
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results1['answer'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "probability rolling >3", "probability rolling two even numbers"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, computing percentage difference, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    review_instruction5 = "Subtask 4: Validate the computed percentage difference for accuracy and correctness"
    review_desc5 = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results5 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, validating percentage difference, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    cot_instruction6 = "Subtask 5: Generate final answer as an integer percentage difference based on validated results"
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results4['answer'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "validated percentage difference"]
    }
    results6 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, generating final integer percentage difference, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
