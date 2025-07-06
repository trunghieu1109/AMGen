async def forward_49(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = (
        "Sub-task 1: Analyze the passage to identify and validate all key dates and events related to the first demarcation line and the stabilization of the front. "
        "Explicitly determine which date (July 29 or August 2) marks the end of hostilities and explain the reasoning."
    )
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': "Please review the analysis of key dates and identify any overlooked information or logical inconsistencies.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, analyzing key dates and validating end of hostilities date, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining analysis, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['list_thinking'][0].content}; answer - {results1['list_answer'][0].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the validated key dates and events from Sub-task 1, calculate the time elapsed in months between the first demarcation line on June 18 and the date when the front stabilized. "
        "Explore multiple reasoning paths considering both July 29 and August 2 as possible stabilization dates, and confirm which is correct."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['list_thinking'][0], results1['list_answer'][0]],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx, _ in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, exploring elapsed time calculations, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    debate_instruction3 = (
        "Sub-task 3: Debate which date (July 29 or August 2) correctly marks the stabilization of the front based on the outputs from Sub-tasks 1 and 2. "
        "Two sub-agents argue for each date and then vote on the correct interval before producing the final answer indicating the number of months elapsed since June 18."
    )
    final_decision_instruction3 = "Sub-task 3: Make final decision on the correct stabilization date and calculate the elapsed months since the first demarcation line."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results1['list_thinking'][0], results1['list_answer'][0], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, arguing stabilization date and elapsed time, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding correct stabilization date and elapsed months, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs
